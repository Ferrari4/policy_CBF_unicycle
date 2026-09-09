import time
import quadprog
import numpy as np
from scipy.integrate import solve_ivp

class backup_filter():
    def __init__(self, policy_class):
        super().__init__()
        self.policy_class = policy_class
        self.obs_class = policy_class.obs_class     # obstacle read at the current time (frozen over the backup horizon)
        self.R_O = policy_class.obs_class.R_O
        self.w_max = policy_class.om_max
        self.v_max = policy_class.v_max
        self.v_min = policy_class.v_min
        self.time_horizion = policy_class.T_rollout
        self.alpha = self.policy_class.alpha
        self.alpha_b = self.policy_class.alpha
        self.inter_input = self.policy_class.inter_input
        self.tspan_b = np.linspace(0.0, 
                                   self.policy_class.T_rollout, 
                                   self.policy_class.horizon + 1)
        self.delta = self.policy_class.cert.delta

    def obs_timeline(self):
        """Obstacle pos / vel / acc at the backup sample times t_now + tspan_b[k].
        Each (H+1, n_obs, 2).  Acceleration by central difference of the velocity."""
        Hp1 = len(self.tspan_b)
        cert = self.policy_class.cert
        obs_pos, obs_vel = cert.obs_timeline(Hp1)
        e = cert.vel_eps
        obs_acc = (cert.obs_vel_timeline(Hp1, t_shift=e)
                   - cert.obs_vel_timeline(Hp1, t_shift=-e)) / (2.0 * e)
        return obs_pos, obs_vel, obs_acc
        
    def safety_Bcbf(self, x, u_nom, d_nom):
        start_time = time.perf_counter()
        M = np.eye(u_nom.shape[0])
        q = u_nom
        G = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        HG = [self.v_min,-self.v_max,-self.w_max, -self.w_max]
        f_x = self.policy_class.dyn.f(x,d_nom)
        g_x = self.policy_class.dyn.G(x,d_nom)
        x_rollout, Q_rollout = self.backup_rollout(x, d_nom)
        obs_pos, obs_vel, obs_acc = self.obs_timeline()          # (H+1, n_obs, 2) each
        K = len(self.tspan_b)
        for i in range(obs_pos.shape[1]):
            for k in range(1, K):
                x_k = x_rollout[k]
                Q_k = Q_rollout[k]
                if k == K - 1:
                    grad_h, h_v, dh_dt = self.compute_filter_hb(x_k, obs_pos[k, i], d_nom, obs_vel[k, i], obs_acc[k, i])
                    constrain_r = -((grad_h @ Q_k) @ f_x + dh_dt + self.alpha_b * h_v)
                else:
                    grad_h, h_v, dh_dt = self.compute_filter_h(x_k, obs_pos[k, i], self.R_O[i], obs_vel[k, i])
                    constrain_r = -((grad_h @ Q_k) @ f_x + dh_dt + self.alpha * h_v)
                constrain_l = (grad_h @ Q_k) @ g_x
                G.append(constrain_l.tolist())
                HG.append(float(constrain_r))
        G = np.asarray(G, dtype=float)
        HG = np.asarray(HG, dtype=float)
        assert G.ndim == 2
        assert G.shape[1] == 2
        assert HG.shape == (G.shape[0],)
        try:
            qp_sol = quadprog.solve_qp(M, q, G.T, HG, 0)
            u_act = qp_sol[0]
        except Exception as e:
            print("QP failed:", e)
            print("x =", x)
            print("u_nom =", u_nom)
            print("G =", G)
            print("HG =", HG)
            u_act = [0,0]
            intervening = "infeasible"
            stop_time = time.perf_counter()
            solve_dt = stop_time - start_time
            return u_act, intervening, solve_dt

        if np.linalg.norm(u_act - u_nom) >=  self.inter_input:
            intervening = True
        else:
            intervening = False

        stop_time = time.perf_counter()
        solve_dt = stop_time - start_time

        return u_act, intervening, solve_dt

    def backup_rollout(self,x,d_nom):
        Q0 = np.eye(3)
        z0 = np.concatenate([x, Q0.reshape(-1)])
        sol = solve_ivp(
            fun=lambda t, z: self.ode(t, z,d_nom),
            t_span=(0.0,  self.time_horizion),
            y0=z0,
            t_eval=self.tspan_b,
            method="RK45"
        )
        if not sol.success:
            raise RuntimeError(
                f"Backup rollout integration failed: {sol.message}"
            )
        x_rollout = sol.y[:3, :].T          
        Q_rollout = sol.y[3:, :].T.reshape(-1, 3, 3)
        return x_rollout, Q_rollout
    
    def ode(self, t, z, d_nom):
        x = z[:3]
        Q = z[3:].reshape(3, 3)
        A_b, v_b, w_b, obs_idx = self.backup_jacobian(x)
        x_dot = self.policy_class.dyn.dynamics(x,[v_b,w_b],
                        d_nom)
        x_dot = np.asarray(x_dot, dtype=float).reshape(3)
        assert self.policy_class.dyn.nx == 3
        Q_dot = A_b @ Q
        return np.concatenate([x_dot, Q_dot.reshape(-1)])

    def backup_jacobian(self, x):
        x = np.asarray(x, dtype=float).reshape(3)
        p = x[:2]
        theta = x[2]
        diff = p[None, :] - self.obs_class.pos_now()
        dists = np.linalg.norm(diff, axis=1)
        obs_idx = int(np.argmin(dists))
        diff_closest = diff[obs_idx]
        D = dists[obs_idx]
        n = diff_closest / D
        q = np.array([
            np.cos(theta),
            np.sin(theta)
        ])
        r = np.array([
            -np.sin(theta),
            np.cos(theta)
        ])

        eps = self.policy_class.policy.eps
        s = n @ r
        v_b = self.v_max
        w_b = self.w_max * np.tanh(s / eps)
        P = np.eye(2) - np.outer(n, n)
        tanh_s = np.tanh(s / eps)
        grad_s_pos = (P @ r) / D
        grad_s_theta = -(n @ q)
        d_w_ds = (self.w_max / eps) * (1.0 - tanh_s**2)
        grad_w = d_w_ds * np.array([
            grad_s_pos[0],
            grad_s_pos[1],
            grad_s_theta
        ])
        A_b = np.array([
            [0.0,       0.0,       -v_b * np.sin(theta)],
            [0.0,       0.0,        v_b * np.cos(theta)],
            [grad_w[0], grad_w[1],  grad_w[2]]
        ])

        return A_b, v_b, w_b, obs_idx

    def compute_filter_h(self, x, obs_pos, r0, obs_vel=None):
        p = x[:2] 
        p_O = obs_pos[:2]
        psi = x[2]
        p = np.asarray(p, dtype=float).reshape(2)
        p_O = np.asarray(p_O, dtype=float).reshape(2)
        v_O = np.zeros(2) if obs_vel is None else np.asarray(obs_vel[:2], dtype=float).reshape(2)
        diff = p - p_O
        D = np.linalg.norm(diff)
        n = diff / D
        P = np.eye(2) - np.outer(n, n)
        q = np.array([np.cos(psi), np.sin(psi)])
        r = np.array([-np.sin(psi), np.cos(psi)])
        h = D - r0 + self.delta * (n @ q)
        grad_h_pos = n + self.delta * (q @ P) / D
        grad_h_psi = self.delta * (n @ r)
        grad_h = np.hstack([
            grad_h_pos,
            grad_h_psi
        ])
        dh_dt = -(n @ v_O) - self.delta * (q @ P @ v_O) / D
        return grad_h, h, dh_dt
    
    def compute_filter_hb(self, x, obs_pos, d_nom, obs_vel=None, obs_acc=None):
        p = x[:2]  
        p_O = obs_pos[:2]
        psi = x[2]
        p = np.asarray(p, dtype=float).reshape(2)
        p_O = np.asarray(p_O, dtype=float).reshape(2)
        v_O = np.zeros(2) if obs_vel is None else np.asarray(obs_vel[:2], dtype=float).reshape(2)
        a_O = np.zeros(2) if obs_acc is None else np.asarray(obs_acc[:2], dtype=float).reshape(2)
        diff = p - p_O
        D = np.linalg.norm(diff)
        n = diff / D
        P = np.eye(2) - np.outer(n, n)
        q = np.array([np.cos(psi), np.sin(psi)])
        r = np.array([-np.sin(psi), np.cos(psi)])
        d_xy = np.asarray(d_nom[:2], dtype=float)
        radial_velocity = q * self.v_max + d_xy - v_O
        h_b = n @ radial_velocity
        grad_hb_pos = (radial_velocity @ P) / D
        grad_hb_psi =  n @ (r * self.v_max)
        grad_hb = np.hstack([
            grad_hb_pos,
            grad_hb_psi
        ])
        dhb_dt = -(radial_velocity @ P @ v_O) / D - (n @ a_O)
        return grad_hb, h_b, dhb_dt
