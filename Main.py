import time
import quadprog
import numpy as np

from Get_obstacles import ObsDyn
from Dynamics import sys_dynm_dd
from Control_policy import policy
from Lyapunov_function import v_certificate
from Barrier_function import h_certificate
from Barrier_function_batch import h_certificate_batch
from Lyapunov_function_batch import v_certificate_batch
from Noise_sampler import noise_train_sampler, noise_test_sampler

class policy_filter:
    def __init__(self, controller,h_controller, v_controller, 
                 obstacles, static_obs,noise_choice, include_h0, include_v0):
        
        # Simulation parameters
        self.T_rollout   = 1.5      # s, CBF certificate lookahead
        self.T_rollout_clf = 0.1    # s, CLF lookahead; 
        self.T_dstb_hold = 0.3      # s, piecewise-constant disturbance interval
        self.T_sim       = 100.0    # s, sim length
        self.dt          = 0.02    # s, sim step size

        # General parameters
        self.v_max = 0.5            # m/s, max linear velocity
        self.v_min = 0.1            # m/s, min linear velocity, applied more so for the QP
        self.om_max = 3.0           # rad/s, max angular velocity

        # obstacles parameters
        radius_to_inflate = 0.0
        amplitude = 0.5
        frequency = 2
        phase = 0.0

        # Noise parameters
        self.n_samples = 50
        self.n_samples_uniform = 25
        self.d_scale = 0.05

        # cbf parameters
        self.alpha = 2.0
        self.inter_input = 1e-3
        cbf_delta = 0.0
        cbf_early_terminate = False

        # clf parameters
        self.slack_weight = 100
        self.gamma = 0.1            # only used by the hand-drawn CLF (clf_qp)
        # Integral policy CLF  J_T(x) = int_0^T l(x_t) dt.  Exact identity along pi:
        #     dJ_T/dt = -l(x_0) + l(x_T)
        # clf_exact_tail=False  -> enforce  Vdot <= -l(x)          (tail dropped, assumes l(x_T) ~ 0)
        # clf_exact_tail=True   -> enforce  Vdot <= -(l(x) - l(x_T)) (exact, always feasible with u = pi)
        self.clf_exact_tail = False
        self.tail_log = []          # l(x_T) / l(x_0) per P-CLF solve: evidence for dropping the tail
        self.cert_valid_log = []

        self.noise_choice = noise_choice
        self.include_h0 = include_h0
        self.include_v0 = include_v0
        self.main_controller = controller
        
        self.horizon       = int(round(self.T_rollout / self.dt))
        self.horizon_clf   = int(round(self.T_rollout_clf / self.dt))
        self.interval_size = int(round(self.T_dstb_hold / self.dt))
        self.n_steps_sim   = int(round(self.T_sim / self.dt))
        
        self.obs_class = ObsDyn(layout=obstacles,
                                static=static_obs,
                                dt=self.dt,
                                barrier_inflate=radius_to_inflate, 
                                amplitude=amplitude, 
                                freq=frequency, 
                                phi=phase)
            
        self.policy = policy(v_max=self.v_max, 
                             om_max=self.om_max,
                             obs_class=self.obs_class, 
                             eps=0.6,  
                             process="batch")
             
        self.dyn = sys_dynm_dd(policy_class=self.policy, 
                               dt=self.dt, 
                               ivp_method="manual_RK4",
                               process="batch")
        
        self.cert = h_certificate(dynamic_class=self.dyn,
                                  obs_class=self.obs_class, 
                                  policy_h="policy_h",
                                  policy_name=h_controller,
                                  delta = cbf_delta)

        self.cert_batch = h_certificate_batch(dynamic_class=self.dyn, 
                                            obs_class=self.obs_class,
                                            hcert_class=self.cert,
                                            terminate_on_hb=cbf_early_terminate)

        self.clf = v_certificate(dynamic_class=self.dyn, 
                                 policy_name=v_controller)

        self.clf_batch = v_certificate_batch(dynamic_class=self.dyn, 
                                             vfun_class=self.clf)

        self.dvdt_log = []          # dV/dt from the moving obstacle, one entry per CBF solve
        self.rng = np.random.default_rng(12345)
        self.test_noise = noise_test_sampler(nd=self.dyn.nd, rng=self.rng)
        self.train_noise = noise_train_sampler(nd=self.dyn.nd, rng=self.rng)

    def u_nominal(self, x, controller, goal):
        if controller == "proportional_policy":
            u = self.policy.proportional_policy(x, goal)

        elif controller == "constant_policy":
            u = self.policy.constant_policy(x, goal)

        elif controller == "backup_policy":
            u = self.policy.backup_policy(x, goal)

        elif controller == "random_policy":
            u = self.policy.random_policy(x, goal)

        else:
            raise ValueError(f"Invalid controller: {controller}.")

        u = np.asarray(u, dtype=float)

        if self.dyn.process == "batch":
            u = u.reshape(-1, self.dyn.nu)

            assert u.shape[0] == 1, \
                "Main simulation expects one nominal control"

            return u[0]

        return u.reshape(self.dyn.nu)

    def noise_selection(self, override=False, horizon=None):
        horizon = self.horizon if horizon is None else horizon
        if override == True:
            sample = 1
            if self.noise_choice=="BangBang": 
                raise ValueError ("For single noise selection with BangBang try test noise function")   
        else:
            sample = self.n_samples

        if self.noise_choice == "BangBang":
            BH_dstb_train, _ = self.train_noise.bangbang_uniform_train(
                        n_samples=self.n_samples,
                        n_samples_uniform=self.n_samples_uniform,
                        horizon=horizon,
                        interval_size=self.interval_size,
                        scale=self.d_scale)

        elif self.noise_choice == "Uniform":
            BH_dstb_train, _ = self.train_noise.uniform_train(
                        n_samples=sample,
                        horizon=horizon,
                        interval_size=self.interval_size,
                        scale=self.d_scale)

        elif self.noise_choice == "Zero":
            BH_dstb_train, _ = self.train_noise.zero_train(
                        n_samples=1,
                        horizon=horizon,
                        interval_size=self.interval_size)

        else:
            raise ValueError(f"Invalid noise_choice: {self.noise_choice}. Must be \
                             'BangBang', 'Uniform', or 'Zero'.")       

        return BH_dstb_train

    def noise_single(self, env_noise, index):
        if env_noise == "Uniform":
            d_env = self.test_noise.uniform_test(rng=self.rng, scale=self.d_scale)
        elif env_noise == "Zero":
            d_env = self.test_noise.zero_test()
        elif env_noise == "BangBang":
            d_env = self.test_noise.bangbang_test(rng=self.rng, k=index, 
                                                  interval_size=self.interval_size, scale=self.d_scale)
        else:
            raise ValueError(f"Unknown environment noise: {env_noise}")

        return d_env

    def propagate(self, x, u, d, goal):
        x_next = self.dyn.solve_ivp_fun(x0=x, d=d, goal=goal, u=u)
        x_next = np.asarray(x_next)
        if self.dyn.process == "batch":
            x_next = x_next.reshape(-1, self.dyn.nx)
            assert x_next.shape[0] == 1, \
                "Main simulation expects one propagated state"
            return x_next[0]
        return x_next.reshape(self.dyn.nx)

    def _init_qp(self, u_nom, use_slack):
        nu = self.dyn.nu
        n_z = nu + 1 if use_slack else nu
        M = np.eye(n_z)
        if use_slack:
            M[nu, nu] = self.slack_weight

        q = np.zeros(n_z)
        if self.main_controller != "clf_nom":       
            q[:nu] = np.asarray(u_nom, dtype=float)
         
        def pad(row):
            return list(row) + ([0.0] if use_slack else [])

        G = [
            pad([1.0, 0.0]),
            pad([-1.0, 0.0]),
            pad([0.0, 1.0]),
            pad([0.0, -1.0])
        ]

        HG = [
             self.v_min,
            -self.v_max,
            -self.om_max,
            -self.om_max
        ]

        return M, q, G, HG, pad

    def _add_clf_constraint(self, G: list, HG: list, V, grad_V, f_x, g_x, use_slack, rate=None):
        # Enforces  grad_V (f + G u) <= -rate + delta.
        # rate=None -> gamma*V (hand-drawn CLF).  Integral P-CLF passes rate = l(x) or l(x) - l(x_T).
        rate = self.gamma * V if rate is None else rate
        LfV = grad_V @ f_x
        LGV = grad_V @ g_x

        row = list(-LGV)

        if use_slack:
            row.append(1.0)

        G.append(row)
        HG.append(LfV + rate)

        if use_slack:
            slack_row = [0.0] * self.dyn.nu + [1.0]
            G.append(slack_row)
            HG.append(0.0)

    def _add_cbf_constraints(self, G: list, HG: list, h_values, grad_h, h_f, h_G, pad, dV_dt=None):
        # Row j:  grad_h (f + G u) + dV_dt + alpha h <= 0   ->   -grad_h G u >= grad_h f + dV_dt + alpha h
        # dV_dt is the explicit time derivative from a moving obstacle (zero when static).
        if dV_dt is None:
            dV_dt = np.zeros(len(h_values))
        for j in range(len(h_values)):
            LfH = grad_h[j] @ h_f[j]
            LGH = grad_h[j] @ h_G[j]

            G.append(pad(list(-LGH)))
            HG.append(LfH + dV_dt[j] + self.alpha * h_values[j])

    def _finalize_constraints(self, G, HG, n_z):
        G = np.asarray(G, dtype=float)
        HG = np.asarray(HG, dtype=float)
        assert G.ndim == 2, (
            f"G must be 2D, got shape {G.shape}."
        )
        assert HG.ndim == 1, (
            f"HG must be 1D, got shape {HG.shape}."
        )
        assert G.shape[1] == n_z, (
            f"Constraint matrix has wrong number of columns: "
            f"G.shape={G.shape}, expected (*, {n_z})."
        )
        assert G.shape[0] == HG.shape[0], (
            f"Number of constraints does not match: "
            f"G has {G.shape[0]} rows but HG has {HG.shape[0]} entries."
        )
        return G, HG

    def _solve_qp(self, M, q, G, HG, u_nom, use_slack):
        nu = self.dyn.nu
        qp_sol = quadprog.solve_qp(M,q,np.asarray(G, dtype=float).T,np.asarray(HG, dtype=float),0)
        u_act = qp_sol[0][:nu]
        delta = (qp_sol[0][nu] if use_slack else 0.0)
        if self.main_controller != "clf_nom":
            intervening = bool(np.linalg.norm(u_act - u_nom) >= self.inter_input)
        else:
            intervening = "Not applicable"
        return u_act, intervening, delta
    
    def _pclf_terms(self, x, goal):
        """Integral P-CLF value/gradient plus the decrease rate the theory prescribes."""
        v_vmax, _, grad_v, v_f, v_G, info = self.clf_batch.get_value_and_grad(
            x, self.noise_selection(horizon=self.horizon_clf), 
            goal, include_v0=self.include_v0)
        V, grad_V = v_vmax[0], grad_v[0]
        ell0 = float(self.clf.clf_certificate(x, goal)[0])       # l(x_0), the integrand now
        ellT = float(info["vHp1v_v"][0, -1, 0])                  # l(x_T) on the worst-case rollout
        decrease_ok = ellT < ell0                       # state inside the set where J_T is a CLF
        if self.clf_exact_tail and decrease_ok:
            rate = ell0 - ellT
        else:
            rate = ell0   
        self.cert_valid_log.append(decrease_ok)
        self.tail_log.append(ellT / max(ell0, 1e-12))

        return V, grad_V, v_f[0], v_G[0], rate

    def clf_qp(self, x, u_nom, d_nom, goal, use_slack):
        start_time = time.perf_counter()
        V = self.clf.clf_value(x, goal) # Hand drawn CLF function
        grad_V = self.clf.clf_value_gradient(x, goal)
        f_x = self.dyn.f(x, d_nom)
        g_x = self.dyn.G(x, d_nom)
        M, q, G, HG, _ = self._init_qp(u_nom, use_slack=use_slack)
        self._add_clf_constraint(G, HG, V, grad_V, f_x, g_x, use_slack=use_slack)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            delta = 0.0

        solve_dt = time.perf_counter() - start_time

        Vdot = np.nan
        alV = np.nan
        if intervening != "infeasible":
            Vdot = grad_V @ (f_x + g_x @ u_act)
            alV = -self.gamma * V 
            assert Vdot <= alV + delta + 1e-7, (
                f"CLF violated at "
                f"Vdot={Vdot:.4f}")
            
        return u_act, intervening, solve_dt, V, delta, Vdot, alV

    def rpcbf_qp(self, x, u_nom, goal, use_slack):
        start_time = time.perf_counter()
        h_hmax, _, grad_h, h_f, h_G, info = self.cert_batch.get_value_and_grad(x, self.noise_selection(),
                                                                        goal, include_h0=self.include_h0)
        dV_dt = info["dV_dt"]
        self.dvdt_log.append(dV_dt.copy())
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=use_slack)
        self._add_cbf_constraints(G, HG, h_hmax, grad_h, h_f, h_G, pad, dV_dt)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, _ = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"

        solve_dt = time.perf_counter() - start_time

        return u_act, intervening, solve_dt, h_hmax, info["h_stop_step"]

    def pclf_qp(self, x, u_nom, goal ,use_slack):
        start_time = time.perf_counter()
        V, grad_V, v_f0, v_G0, rate = self._pclf_terms(x, goal)
        M, q, G, HG, _ = self._init_qp(u_nom, use_slack=use_slack)
        self._add_clf_constraint(G, HG, V, grad_V, v_f0, v_G0, use_slack=use_slack, rate=rate)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            delta = 0.0

        solve_dt = time.perf_counter() - start_time

        Vdot = np.nan
        alV = np.nan
        if intervening != "infeasible":
            Vdot = grad_V @ (v_f0 + v_G0 @ u_act)
            alV = -rate
            assert Vdot <= alV + delta + 1e-7, (
                f"P-CLF violated at "
                f"Vdot={Vdot:.4f}")

        return u_act, intervening, solve_dt, V, delta, Vdot, alV

    def pclf_rpcbf_qp(self, x, u_nom, d_nom, goal, use_slack):
        start_time = time.perf_counter()
        h_hmax, _, grad_h, h_f, h_G, info_h = self.cert_batch.get_value_and_grad(x, self.noise_selection(), 
                                                                            goal, include_h0=self.include_h0)
        dV_dt = info_h["dV_dt"]
        self.dvdt_log.append(dV_dt.copy())
        V, grad_V, _, _, rate = self._pclf_terms(x, goal)
        f_x = self.dyn.f(x, d_nom)
        g_x = self.dyn.G(x, d_nom)
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=use_slack)
        self._add_cbf_constraints(G, HG, h_hmax, grad_h, h_f, h_G, pad, dV_dt)
        self._add_clf_constraint(G, HG, V, grad_V, f_x, g_x, use_slack=use_slack, rate=rate)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            delta = 0.0

        solve_dt = time.perf_counter() - start_time

        Vdot = np.nan
        alV = np.nan
        if intervening != "infeasible":
            Vdot = grad_V @ (f_x + g_x @ u_act)
            alV = -rate 
            assert Vdot <= alV + delta + 1e-7, (
                f"CLF violated at "
                f"Vdot={Vdot:.4f}")

            for j in range(len(h_hmax)):
                hdot = grad_h[j] @ (h_f[j] + h_G[j] @ u_act) + dV_dt[j]
                assert hdot <= -self.alpha * h_hmax[j] + 1e-7, (
                    f"CBF row {j} "
                    f"violated at "
                    f"hdot={hdot:.4f}"
                )

        return u_act, intervening, solve_dt, V, delta, h_hmax, Vdot, alV

    def clf_cbf_qp(self, x, u_nom, d_nom, goal, use_slack):
        start_time = time.perf_counter()
        h_hmax, _, grad_h, h_f, h_G, info_h = self.cert_batch.get_value_and_grad(x, self.noise_selection(), 
                                                                    goal, include_h0=self.include_h0)
        dV_dt = info_h["dV_dt"]
        self.dvdt_log.append(dV_dt.copy())
        V = self.clf.clf_value(x, goal)
        grad_V = self.clf.clf_value_gradient(x, goal)
        f_x = self.dyn.f(x, d_nom)
        g_x = self.dyn.G(x, d_nom)
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=use_slack)
        self._add_cbf_constraints(G, HG, h_hmax, grad_h, h_f, h_G, pad, dV_dt)
        self._add_clf_constraint(G, HG, V, grad_V, f_x, g_x, use_slack=use_slack)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            delta = 0.0

        solve_dt = time.perf_counter() - start_time

        Vdot = np.nan
        alV = np.nan
        if intervening != "infeasible":
            Vdot = grad_V @ (f_x + g_x @ u_act)
            alV = -self.gamma * V 
            assert Vdot <= alV + delta + 1e-7, (
                f"CLF violated at "
                f"Vdot={Vdot:.4f}")

            for j in range(len(h_hmax)):
                hdot = grad_h[j] @ (h_f[j] + h_G[j] @ u_act) + dV_dt[j]
                assert hdot <= -self.alpha * h_hmax[j] + 1e-7, (
                    f"CBF row {j} "
                    f"violated at "
                    f"hdot={hdot:.4f}"
                )

        return u_act, intervening, solve_dt, V, delta, h_hmax, Vdot, alV

    def pclf_goals_qp(self, x, u_nom, goals, use_slack):
        start_time = time.perf_counter()
        W, _, grad_W, v_f, v_G, info = self.clf_batch.get_value_and_grad_goals(
            x, goals, self.horizon, include_v0=self.include_v0, k_cert=1.0)
        V = W[0] - info["c_floor"]            # certificate on W_A - c  (>= 0)
        grad_V = grad_W[0]
        M, q, G, HG, _ = self._init_qp(u_nom, use_slack=use_slack)
        self._add_clf_constraint(G, HG, V, grad_V, v_f[0], v_G[0], use_slack=use_slack)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            delta = 0.0

        solve_dt = time.perf_counter() - start_time

        Vdot = np.nan; alV = np.nan
        if intervening != "infeasible":
            Vdot = grad_V @ (v_f[0] + v_G[0] @ u_act)
            alV = -self.gamma * V
            assert Vdot <= alV + delta + 1e-7, f"P-CLF violated: Vdot={Vdot:.4f}"

        return u_act, intervening, solve_dt, V, delta, Vdot, alV

    def two_step_pclf_pcbf(self, x, u_nom, d_nom, goal, use_slack):
        start_time = time.perf_counter()
        V, grad_V, _, _, rate = self._pclf_terms(x, goal)
        f_x = self.dyn.f(x, d_nom)
        g_x = self.dyn.G(x, d_nom)
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=use_slack)
        self._add_clf_constraint(G, HG, V, grad_V, f_x, g_x, use_slack=use_slack, rate=rate)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        # First QP block:
        try:
            u_nom, intervening, delta_clf = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            delta = 0.0
            solve_dt = time.perf_counter() - start_time

            return u_act, intervening, solve_dt, V, delta, np.full(self.cert.nh, np.nan) , np.nan, np.nan 

        h_hmax, _, grad_h, h_f, h_G, info_h = self.cert_batch.get_value_and_grad(x, self.noise_selection(), 
                                                                            goal, include_h0=self.include_h0)
        dV_dt = info_h["dV_dt"]
        self.dvdt_log.append(dV_dt.copy())
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=use_slack)
        q[:self.dyn.nu] = np.asarray(u_nom, dtype=float)
        self._add_cbf_constraints(G, HG, h_hmax, grad_h, h_f, h_G, pad, dV_dt)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        # Second QP block:
        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
            intervening = bool(np.linalg.norm(u_act - u_nom) >= self.inter_input)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            delta = 0.0

        solve_dt = time.perf_counter() - start_time
        
        Vdot = np.nan
        alV = np.nan

        if intervening != "infeasible":
            Vdot = grad_V @ (f_x + g_x @ u_act)
            alV = -rate 
            for j in range(len(h_hmax)):
                hdot = grad_h[j] @ (h_f[j] + h_G[j] @ u_act) + dV_dt[j]
                assert hdot <= -self.alpha * h_hmax[j] + 1e-7, (
                    f"CBF row {j} "
                    f"violated at "
                    f"hdot={hdot:.4f}"
                )

        return u_act, intervening, solve_dt, V, delta_clf, h_hmax, Vdot, alV