import time
import numpy as np
import quadprog

from Dynamics import sys_dynm_dd
from Control_policy import policy

class v_certificate:
    def __init__(self ,dynamic_class: sys_dynm_dd, policy_name="constant_policy",alpha=1.0,
                 goal = np.array([4.5,4.5]), ivp_method="manual_RK4", 
                 v_max=1.0 ,om_max=2.0 ,slack=False):

        self.sys_dynm = dynamic_class
        self.use_slack = slack
        self.slack_weight = 100
        self.alpha = alpha
        self.gamma = 0.2 # For CLF
        self.k = 1.0
        self.v_max = v_max
        self.om_max = om_max
        self.inter_input = 1e-3

        self.nx = self.sys_dynm.nx          
        self.goal = goal
        self.nV = 1                        
        self.policy_name = policy_name
        self.ivp_method = ivp_method
  
    def clf_value(self, x, goal):
        px, py, theta = x
        px_goal, py_goal = goal
        dx = px - px_goal
        dy = py - py_goal
        d = np.sqrt(dx**2 + dy**2)
        theta_goal = np.arctan2(py_goal - py, px_goal - px)
        # Wrapped heading error [-pi, pi]
        theta_err = np.arctan2(
            np.sin(theta_goal - theta),
            np.cos(theta_goal - theta)
        )
        # CLF
        V = d**2 * (0.5 + self.k * (1.0 - np.cos(theta_err)))
        return V

    def clf_value_gradient(self, x, goal):
        px, py, theta = x
        px_goal, py_goal = goal
        dx = px - px_goal
        dy = py - py_goal
        d2 = dx**2 + dy**2
        theta_goal = np.arctan2(py_goal - py, px_goal - px)
        # Wrapped heading error [-pi, pi]
        theta_err = np.arctan2(
            np.sin(theta_goal - theta),
            np.cos(theta_goal - theta)
        )
        sin_err = np.sin(theta_err)
        cos_err = np.cos(theta_err)
        A = 0.5 + self.k * (1.0 - cos_err)
        # Analytical gradient
        dV_dpx = 2.0 * dx * A - self.k * dy * sin_err
        dV_dpy = 2.0 * dy * A + self.k * dx * sin_err
        dV_dtheta = -self.k * d2 * sin_err
        grad_V = np.array([
            dV_dpx,
            dV_dpy,
            dV_dtheta
        ])
        return grad_V

    def clf_certficate(self, x, goal=None):
        goal = self.goal if goal is None else goal
        px, py, theta = x
        px_goal, py_goal = goal
        dx = px - px_goal
        dy = py - py_goal
        d = np.sqrt(dx**2 + dy**2)
        theta_goal = np.arctan2(py_goal - py, px_goal - px)
        # Wrapped heading error [-pi, pi]
        theta_err = np.arctan2(
            np.sin(theta_goal - theta),
            np.cos(theta_goal - theta)
        )
        V = 0.5 * d**2 + self.k * (1.0 - np.cos(theta_err))
        return np.array([V])

    def evaluate_v_trajectory(self, trajectory):
        if trajectory.ndim != 2:
            raise ValueError("trajectory must have shape (horizon + 1, nx)")
        if trajectory.shape[1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {trajectory.shape[1]}")

        return np.stack([self.clf_certficate(state) for state in trajectory], axis=0)

    def aggregate_v(self, t_values, v_values):
        return np.max(v_values, axis=0)

    def compute_v_vmax(self, x0, bH_dstb, include_v0=True):

        x0 = self.sys_dynm.chk_x(x0)
        bH_dstb = np.asarray(bH_dstb, dtype=float)
        if bH_dstb.ndim != 3:
            raise ValueError("bH_dstb must have shape " "(n_samples, horizon, nd)")
        n_samples, horizon, nd = bH_dstb.shape
        if nd != self.sys_dynm.nd:
            raise ValueError(f"Expected disturbance dimension " f"{self.sys_dynm.nd}, got {nd}")

        sample_vmax = []
        trajectories = []
        value_histories = []

        for H_dstb in bH_dstb:
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, policy_name=self.policy_name, method=self.ivp_method)
            v_history = self.evaluate_v_trajectory(trajectory)
            t_history = np.linspace(0.0, self.sys_dynm.dt * (v_history.shape[0] - 1), v_history.shape[0])
            v_values = v_history if include_v0 else v_history[1:]
            t_values = t_history if include_v0 else t_history[1:]

            v_agg = self.aggregate_v(t_values, v_values)

            trajectories.append(trajectory)
            value_histories.append(v_history)
            sample_vmax.append(v_agg)

        bHp1_x = np.stack(trajectories, axis=0)
        bHp1v_v = np.stack(value_histories, axis=0)
        bv_vmax = np.stack(sample_vmax, axis=0)

        assert bHp1_x.shape == (n_samples, horizon + 1, self.nx)
        assert bHp1v_v.shape == (n_samples, horizon + 1, self.nV)
        assert bv_vmax.shape == (n_samples, self.nV)

        v_argmax = np.argmax(bv_vmax, axis=0)
        value_indices = np.arange(self.nV)
        v_vmax = bv_vmax[v_argmax, value_indices]
        vH_dstb = bH_dstb[v_argmax]
        vHp1_x = bHp1_x[v_argmax]
        vHp1v_v = bHp1v_v[v_argmax]

        assert v_vmax.shape == (self.nV,)
        assert vH_dstb.shape == (self.nV, horizon, self.sys_dynm.nd)

        info = {
            "v_argmax": v_argmax,
            "bv_vmax": bv_vmax,
            "vHp1_x": vHp1_x,
            "vHp1v_v": vHp1v_v,
            "bHp1_x": bHp1_x,
            "bHp1v_v": bHp1v_v,
            "v_vmax": v_vmax,
        }

        return v_vmax, vH_dstb, info

    def compute_v_diag(self, x0, vH_dstb, include_v0=True):
        v_out = np.zeros(self.nV)
        for j, H_dstb in enumerate(vH_dstb):
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, policy_name=self.policy_name, method=self.ivp_method)
            v_history = self.evaluate_v_trajectory(trajectory)
            t_history = np.linspace(0.0, self.sys_dynm.dt * (v_history.shape[0] - 1), v_history.shape[0])
            v_values = v_history if include_v0 else v_history[1:]
            t_values = t_history if include_v0 else t_history[1:]
            v_out[j] = self.aggregate_v(t_values, v_values)[j]
        return v_out

    def get_value_and_grad(self, x0, bH_dstb, include_v0=True, eps=1e-5):
        v_vmax, vH_dstb, info = self.compute_v_vmax(x0, bH_dstb, include_v0)
        grad_v_vmax = np.zeros((self.nV, self.nx))
        for i in range(self.nx):
            e = np.zeros(self.nx)
            e[i] = eps
            vp = self.compute_v_diag(x0 + e, vH_dstb, include_v0)
            vm = self.compute_v_diag(x0 - e, vH_dstb, include_v0)
            grad_v_vmax[:, i] = (vp - vm) / (2.0 * eps)

        v0_dstb = vH_dstb[:, 0]
        v_f = np.stack([self.sys_dynm.f(x0, d) for d in v0_dstb], axis=0)
        v_G = np.stack([self.sys_dynm.G(x0, d) for d in v0_dstb], axis=0)

        info["vx_gradvmax"] = grad_v_vmax

        return v_vmax, vH_dstb, grad_v_vmax, v_f, v_G, info

    def solve_pclf_qp(self, u_nom, V, grad_V, f_x, g_x):
        start_time = time.perf_counter()
        nu = self.sys_dynm.nu
        n_z = nu + 1 if self.use_slack else nu      # z = [v, om, (delta)]

        M = np.eye(n_z)
        if self.use_slack:
            M[nu, nu] = self.slack_weight
        q = np.zeros(n_z)
        q[:nu] = np.asarray(u_nom, dtype=float)

        def pad(row):
            return list(row) + ([0.0] if self.use_slack else [])

        G = [pad([1.0, 0.0]), pad([-1.0, 0.0]),
            pad([0.0, 1.0]), pad([0.0, -1.0])]
        HG = [0.0, -self.v_max, -self.om_max, -self.om_max]

        # P-CLF row (SOFT when slack on): Vdot <= -gamma*V + delta
        LfV = grad_V @ f_x
        LGV = grad_V @ g_x
        G.append(list(-LGV) + ([1.0] if self.use_slack else []))
        HG.append(LfV + self.gamma * V)

        if self.use_slack:
            G.append([0.0] * nu + [1.0])
            HG.append(0.0)

        try:
            qp_sol = quadprog.solve_qp(M, q, np.array(G, dtype=float).T,
                                    np.array(HG, dtype=float), 0)
            u_act = qp_sol[0][:nu]
            delta = qp_sol[0][nu] if self.use_slack else 0.0
        except Exception as e:
            print("QP failed:", e)
            u_act = np.array([np.clip(u_nom[0], 0.0, self.v_max),
                            np.clip(u_nom[1], -self.om_max, self.om_max)])
            intervening = "infeasible"
            solve_dt = time.perf_counter() - start_time
            return u_act, intervening, solve_dt, V, 0.0

        if np.linalg.norm(u_act - u_nom) >= self.inter_input:
            intervening = True
        else:
            intervening = False

        solve_dt = time.perf_counter() - start_time
        return u_act, intervening, solve_dt, V, delta

    def solve_clf_qp(self, u_nom, x, goal):
        start_time = time.perf_counter()

        V = self.clf_value(x, goal)
        grad_V = self.clf_value_gradient(x, goal)
        f_x = self.sys_dynm.f(x, [0, 0, 0])
        g_x = self.sys_dynm.G(x, [0, 0, 0])

        nu = self.sys_dynm.nu
        n_z = nu + 1 if self.use_slack else nu      # z = [v, om, (delta)]

        # Objective: 0.5*||u - u_nom||^2 + 0.5*slack_weight*delta^2
        M = np.eye(n_z)
        if self.use_slack:
            M[nu, nu] = self.slack_weight           # e.g. 100.0
        q = np.zeros(n_z)
        q[:nu] = np.asarray(u_nom, dtype=float)

        def pad(row):
            # hard constraints get a 0 in the slack column
            return list(row) + ([0.0] if self.use_slack else [])

        # Input box: 0 <= v <= v_max, |om| <= om_max
        # (swap for the wheel-diamond |v|/v_max + |om|/om_max <= 1 later)
        G = [pad([1.0, 0.0]), pad([-1.0, 0.0]),
            pad([0.0, 1.0]), pad([0.0, -1.0])]
        HG = [0.0, -self.v_max, -self.om_max, -self.om_max]

        # CLF row (SOFT when slack on): Vdot <= -gamma*V + delta
        #   =>  -LGV @ u + delta >= LfV + gamma*V
        LfV = grad_V @ f_x                          # = 0 for the unicycle (drift-free)
        LGV = grad_V @ g_x
        G.append(list(-LGV) + ([1.0] if self.use_slack else []))
        HG.append(LfV + self.gamma * V)

        # delta >= 0
        if self.use_slack:
            G.append([0.0] * nu + [1.0])
            HG.append(0.0)

        try:
            qp_sol = quadprog.solve_qp(M, q, np.array(G, dtype=float).T,
                                    np.array(HG, dtype=float), 0)
            u_act = qp_sol[0][:nu]
            delta = qp_sol[0][nu] if self.use_slack else 0.0

        except Exception as e:
            print("QP failed:", e)
            u_act = np.array([np.clip(u_nom[0], 0.0, self.v_max),
                            np.clip(u_nom[1], -self.om_max, self.om_max)])
            intervening = "infeasible"
            solve_dt = time.perf_counter() - start_time
            return u_act, intervening, solve_dt, V, 0.0

        if np.linalg.norm(u_act - u_nom) >= self.inter_input:
            intervening = True
        else:
            intervening = False

        solve_dt = time.perf_counter() - start_time
        return u_act, intervening, solve_dt, V, delta

    def solve_clf_cbf_qp(self, u_nom, x, goal, h_hmax, grad_h_hmax, h_f, h_G):
        """
        Combined QP for result set two:
        - CBF rows HARD    : hdot <= -alpha*h        (h <= 0 safe, RPCBF convention)
        - CLF row  SOFT    : Vdot <= -gamma*V + delta (when use_slack)
        """

        start_time = time.perf_counter()
        V = self.clf_value(x, goal)
        grad_V = self.clf_value_gradient(x, goal)
        f_x = self.sys_dynm.f(x, [0,0,0])  # nominal model in the controller
        g_x = self.sys_dynm.G(x, [0,0,0])

        nu = self.sys_dynm.nu
        n_z = nu + 1 if self.use_slack else nu      # z = [v, om, (delta)]

        # Objective: 0.5*||u - u_nom||^2 + 0.5*slack_weight*delta^2
        M = np.eye(n_z)
        if self.use_slack:
            M[nu, nu] = self.slack_weight
        q = np.zeros(n_z)
        q[:nu] = np.asarray(u_nom, dtype=float)

        def pad(row):
            # hard constraints get a 0 in the slack column
            return list(row) + ([0.0] if self.use_slack else [])

        # Input box: 0 <= v <= v_max, |om| <= om_max
        G = [pad([1.0, 0.0]), pad([-1.0, 0.0]),
            pad([0.0, 1.0]), pad([0.0, -1.0])]
        HG = [0.0, -self.v_max, -self.om_max, -self.om_max]

        # CBF rows (HARD): hdot <= -alpha*h
        #   =>  -LGH @ u >= LfH + alpha*h
        for j in range(len(h_hmax)):
            LfH = grad_h_hmax[j] @ h_f[j]
            LGH = grad_h_hmax[j] @ h_G[j]
            G.append(pad(list(-LGH)))
            HG.append(LfH + self.alpha * h_hmax[j])

        # CLF row (SOFT when slack on): Vdot <= -gamma*V + delta
        #   =>  -LGV @ u + delta >= LfV + gamma*V
        LfV = grad_V @ f_x
        LGV = grad_V @ g_x
        G.append(list(-LGV) + ([1.0] if self.use_slack else []))
        HG.append(LfV + self.gamma * V)

        # delta >= 0
        if self.use_slack:
            G.append([0.0] * nu + [1.0])
            HG.append(0.0)

        try:
            qp_sol = quadprog.solve_qp(M, q, np.array(G, dtype=float).T,
                                    np.array(HG, dtype=float), 0)
            u_act = qp_sol[0][:nu]
            delta = qp_sol[0][nu] if self.use_slack else 0.0

        except Exception as e:
            print("QP failed:", e)
            if len(h_hmax) > 0:
                j = int(np.argmax(h_hmax))
                LGH = grad_h_hmax[j] @ h_G[j]
                u_lim = np.array([self.v_max, self.om_max])
                u_act = np.clip(-np.sign(LGH) * u_lim, -u_lim, u_lim)
                u_act[0] = np.clip(u_act[0], 0.0, self.v_max)
            else:   # no barriers passed -> degenerate to CLF-only fallback
                u_act = np.array([np.clip(u_nom[0], 0.0, self.v_max),
                                np.clip(u_nom[1], -self.om_max, self.om_max)])
            intervening = "infeasible"
            solve_dt = time.perf_counter() - start_time
            return u_act, intervening, solve_dt, h_hmax, V, 0.0

        if np.linalg.norm(u_act - u_nom) >= self.inter_input:
            intervening = True
        else:
            intervening = False

        solve_dt = time.perf_counter() - start_time
        return u_act, intervening, solve_dt, h_hmax, V, delta

    