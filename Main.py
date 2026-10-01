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
    def __init__(self, controller, h_controller, v_controller,
                obstacles, static_obs, noise_choice_cbf, noise_choice_clf,
                include_h0, include_v0, T_rollout=None, T_rollout_clf=None):

        # Simulation parameters
        self.T_rollout     = 1.5 if T_rollout is None else float(T_rollout) # CBF
        self.T_rollout_clf = 3.0 if T_rollout_clf is None else float(T_rollout_clf) # CLF
        self.T_dstb_hold = 0.3      # s, piecewise-constant disturbance interval
        self.T_sim       = 100.0    # s, sim length
        self.dt          = 0.01    # s, sim step size

        # General parameters
        self.v_max = 0.5            # m/s, max linear velocity
        self.v_min = 0.1            # m/s, min linear velocity, applied more so for the QP
        self.om_max = 3.0           # rad/s, max angular velocity

        # obstacles parameters
        self.radius_to_inflate = 0.0
        self.obs_amplitude = 0.5
        self.obs_frequency = 8
        self.obs_phase = 0.0

        # Noise parameters
        self.n_samples = 50
        self.n_samples_uniform = 25
        self.rho = 0.2
        self.d_scale = self.rho * np.array([self.v_max, self.v_max, self.om_max])

        # cbf parameters
        self.alpha = 2.0
        self.inter_input = 1e-3
        self.cbf_delta = 0.5
        self.cbf_early_terminate = False
        self.hdot_log = []   # grad_h (f + G u_act) + dV_dt, one (nh,) row per CBF solve
        self.alh_log  = []   # -alpha * h_hmax, the rate the constraint demanded
        self.cbf_active_log = []   # bool per CBF solve: CBF multiplier > 0
        self.lam_cbf_log    = []   # (nh,) CBF multipliers per CBF solve

        # clf parameters
        self.gamma = 1.0 # Unused
        self.slack_weight = 100
        self.clf_exact_tail = True
        self.tail_log = []          # l(x_T) / l(x_0) per P-CLF solve: evidence for dropping the tail
        self.ell0_log = []
        self.ellT_log = []
        self.cert_valid_log = []

        # policy / dynamics settings
        self.policy_eps = 0.6
        self.ivp_method = "manual_RK4"
        self.process    = "batch"

        # inputs (stored for logging)
        self.noise_choice_cbf = noise_choice_cbf
        self.noise_choice_clf = noise_choice_clf
        self.include_h0 = include_h0
        self.include_v0 = include_v0
        self.main_controller = controller
        self.h_controller = h_controller
        self.v_controller = v_controller
        self.obstacles_layout = obstacles
        self.static_obs = static_obs

        self.horizon   = int(round(self.T_rollout / self.dt))
        self.horizon_clf   = int(round(self.T_rollout_clf / self.dt))
        self.interval_size = int(round(self.T_dstb_hold / self.dt))
        self.n_steps_sim   = int(round(self.T_sim / self.dt))

        self.obs_class = ObsDyn(layout=obstacles,
                                static=static_obs,
                                dt=self.dt,
                                barrier_inflate=self.radius_to_inflate,
                                amplitude=self.obs_amplitude,
                                freq=self.obs_frequency,
                                phi=self.obs_phase)

        self.policy = policy(v_max=self.v_max,
                             om_max=self.om_max,
                             obs_class=self.obs_class,
                             eps=self.policy_eps,
                             process=self.process)

        self.dyn = sys_dynm_dd(policy_class=self.policy,
                               dt=self.dt,
                               ivp_method=self.ivp_method,
                               process=self.process)

        self.cert = h_certificate(dynamic_class=self.dyn,
                                  obs_class=self.obs_class,
                                  policy_h="policy_h",
                                  policy_name=h_controller,
                                  delta=self.cbf_delta)

        self.cert_batch = h_certificate_batch(dynamic_class=self.dyn,
                                            obs_class=self.obs_class,
                                            hcert_class=self.cert,
                                            terminate_on_hb=self.cbf_early_terminate)

        self.clf = v_certificate(dynamic_class=self.dyn,
                                 policy_name=v_controller,
                                 obs_class=self.obs_class)

        self.clf_batch = v_certificate_batch(dynamic_class=self.dyn,
                                             vfun_class=self.clf)

        self.dvdt_log = []          # dV/dt from the moving obstacle, one entry per CBF solve
        self.seed_cert = 12345
        self.seed_env  = 54321
        self.rng = np.random.default_rng(self.seed_cert)        # rollouts / certificates (unchanged)
        self.env_rng = np.random.default_rng(self.seed_env)     # environment disturbance only
        self.test_noise = noise_test_sampler(nd=self.dyn.nd, rng=self.env_rng)
        self.train_noise = noise_train_sampler(nd=self.dyn.nd, rng=self.rng)

    def params(self):
        """Every setting this filter ran with, as plain scalars/strings, for logging next to results."""
        return dict(
            # timing / horizons
            T_rollout=self.T_rollout, T_rollout_clf=self.T_rollout_clf, T_dstb_hold=self.T_dstb_hold,
            T_sim=self.T_sim, dt=self.dt, horizon=self.horizon, horizon_clf=self.horizon_clf,
            interval_size=self.interval_size, n_steps_sim=self.n_steps_sim,
            # input limits
            v_max=self.v_max, v_min=self.v_min, om_max=self.om_max,
            # CBF
            alpha=self.alpha, cbf_delta=self.cbf_delta, cbf_early_terminate=self.cbf_early_terminate,
            inter_input=self.inter_input, include_h0=self.include_h0,
            # CLF
            gamma=self.gamma, slack_weight=self.slack_weight, clf_exact_tail=self.clf_exact_tail,
            include_v0=self.include_v0,
            # disturbance model
            n_samples=self.n_samples, n_samples_uniform=self.n_samples_uniform, rho=self.rho,
            d_scale_x=float(self.d_scale[0]), d_scale_y=float(self.d_scale[1]), d_scale_th=float(self.d_scale[2]),
            noise_choice_cbf=self.noise_choice_cbf, noise_choice_clf=self.noise_choice_clf,
            # policies
            main_controller=self.main_controller, h_controller=self.h_controller, v_controller=self.v_controller,
            policy_eps=self.policy_eps, ivp_method=self.ivp_method, process=self.process,
            # obstacles
            obstacles_layout=self.obstacles_layout, static_obs=self.static_obs,
            radius_to_inflate=self.radius_to_inflate, obs_amplitude=self.obs_amplitude,
            obs_frequency=self.obs_frequency, obs_phase=self.obs_phase,
            n_obs=int(len(self.obs_class.R_O)),
            obs_centres=";".join(f"{c[0]:.4f},{c[1]:.4f}" for c in np.asarray(self.obs_class.pos_now())),
            obs_radii=";".join(f"{r:.4f}" for r in np.asarray(self.obs_class.R_O)),
            # seeds
            seed_cert=self.seed_cert, seed_env=self.seed_env,
        )

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

    def _noise_selection(self, choice, sampler, horizon, override=False):
        sample = 1 if override else self.n_samples
        if override and choice == "BangBang":
            raise ValueError("For single noise selection with BangBang try test noise function")
        if choice == "BangBang":
            d, _ = sampler.bangbang_uniform_train(n_samples=self.n_samples,
                    n_samples_uniform=self.n_samples_uniform, horizon=horizon,
                    interval_size=self.interval_size, scale=self.d_scale)
        elif choice == "Uniform":
            d, _ = sampler.uniform_train(n_samples=sample, horizon=horizon,
                    interval_size=self.interval_size, scale=self.d_scale)
        elif choice == "Zero":
            d, _ = sampler.zero_train(n_samples=1, horizon=horizon,
                    interval_size=self.interval_size)
        else:
            raise ValueError(f"Invalid noise_choice: {choice}")
        return d

    def noise_selection_cbf(self, override=False, horizon=None):
        return self._noise_selection(self.noise_choice_cbf, self.train_noise,
                                    self.horizon if horizon is None else horizon, override)

    def noise_selection_clf(self, override=False, horizon=None):
        return self._noise_selection(self.noise_choice_clf, self.train_noise,
                                    self.horizon_clf if horizon is None else horizon, override)

    def noise_single(self, env_noise, index):
        if env_noise == "Uniform":
            d_env = self.test_noise.uniform_test(rng=self.env_rng, scale=self.d_scale)
        elif env_noise == "Zero":
            d_env = self.test_noise.zero_test()
        elif env_noise == "BangBang":
            d_env = self.test_noise.bangbang_test(rng=self.env_rng, k=index, 
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

    def _solve_qp(self, M, q, G, HG, u_nom, use_slack, n_cbf=0):
        nu = self.dyn.nu
        G = np.asarray(G, dtype=float)
        HG = np.asarray(HG, dtype=float)
        z, _, _, _, lagr, _ = quadprog.solve_qp(M, q, G.T, HG, 0)
        u_act = z[:nu]
        delta = z[nu] if use_slack else 0.0
        if self.main_controller != "clf_nom":
            intervening = bool(np.linalg.norm(u_act - u_nom) >= self.inter_input)
        else:
            intervening = "Not applicable"
        if n_cbf > 0:                              # CBF rows are always the last n_cbf rows
            lam_cbf = lagr[-n_cbf:]
            self.lam_cbf_log.append(lam_cbf.copy())
            self.cbf_active_log.append(bool(np.any(lam_cbf > 1e-9)))

        return u_act, intervening, delta

    def _pclf_terms(self, x, goal):
        """Integral P-CLF value/gradient plus the decrease rate the theory prescribes."""
        v_vmax, _, grad_v, v_f, v_G, info = self.clf_batch.get_value_and_grad(
            x, self.noise_selection_clf(horizon=self.horizon_clf),
            goal, include_v0=self.include_v0)
        V_max, grad_V = v_vmax[0], grad_v[0]                     # worst-case cumulative cost (gradient source)
        V = float(self.clf.clf_certificate(x, goal)[0])          # instantaneous V(x_t), like h_now
        ell0 = V                                                 # l(x_0) == V(x_t): the integrand now
        ellT = float(info["vHp1v_v"][0, -1, 0])                  # l(x_T) on the worst-case rollout
        decrease_ok = ellT < ell0
        if self.clf_exact_tail and decrease_ok:
            rate = ell0 - ellT
        else:
            rate = ell0
        self.cert_valid_log.append(decrease_ok)
        self.tail_log.append(ellT / max(ell0, 1e-12))
        self.ell0_log.append(ell0)
        self.ellT_log.append(ellT)

        return V_max, grad_V, v_f[0], v_G[0], rate, V

    def _pcbf_terms(self, h_values, grad_h, h_f, h_G, dV_dt, u_act, check=True):
        """hdot along u_act and the required rate -alpha*h, both (nh,). Always logs one row.
        Constraint:  hdot <= -alpha h   (h <= 0 safe).   margin = -alpha h - hdot  >= 0.
        """
        h_values = np.atleast_1d(np.asarray(h_values, dtype=float))
        nh = h_values.shape[0]
        if u_act is None:                              # QP infeasible
            hdot = np.full(nh, np.nan)
            alh  = np.full(nh, np.nan)
        else:
            hdot = np.array([grad_h[j] @ (h_f[j] + h_G[j] @ u_act) + dV_dt[j]
                             for j in range(nh)])
            alh  = -self.alpha * h_values
        self.hdot_log.append(hdot)
        self.alh_log.append(alh)
        if u_act is not None and check:
            for j in range(nh):
                assert hdot[j] <= alh[j] + 1e-7, (
                    f"CBF row {j} violated at hdot={hdot[j]:.4f} > -alpha*h={alh[j]:.4f}")
        return hdot, alh

    def _log_cbf_nan(self):
        """Placeholder row when a step ends before the CBF certificate is evaluated."""
        self.hdot_log.append(np.full(self.cert.nh, np.nan))
        self.alh_log.append(np.full(self.cert.nh, np.nan))
        self.lam_cbf_log.append(np.full(self.cert.nh, np.nan))
        self.cbf_active_log.append(False)

    def rpcbf_qp(self, x, u_nom, goal, use_slack):
        start_time = time.perf_counter()
        h_hmax, _, grad_h, h_f, h_G, info = self.cert_batch.get_value_and_grad(x, self.noise_selection_cbf(),
                                                                        goal, include_h0=self.include_h0)

        dV_dt = info["dV_dt"]
        self.dvdt_log.append(dV_dt.copy())
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=use_slack)
        self._add_cbf_constraints(G, HG, h_hmax, grad_h, h_f, h_G, pad, dV_dt)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, _ = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack,
                                                n_cbf=len(h_hmax))
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            self.lam_cbf_log.append(np.full(self.cert.nh, np.nan))
            self.cbf_active_log.append(False)

        # hdot / -alpha*h along the applied input (NaN row if infeasible). Outside the try on purpose.
        self._pcbf_terms(h_hmax, grad_h, h_f, h_G, dV_dt,
                        None if intervening == "infeasible" else u_act)

        solve_dt = time.perf_counter() - start_time

        return u_act, intervening, solve_dt, h_hmax, info["h_stop_step"]

    def pclf_qp(self, x, u_nom, goal ,use_slack):
        start_time = time.perf_counter()
        V_max, grad_V, v_f0, v_G0, rate, V = self._pclf_terms(x, goal)
        M, q, G, HG, _ = self._init_qp(u_nom, use_slack=use_slack)
        self._add_clf_constraint(G, HG, V_max, grad_V, v_f0, v_G0, use_slack=use_slack, rate=rate)
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

        return u_act, intervening, solve_dt, V_max, delta, Vdot, alV, V

    def pclf_rpcbf_qp(self, x, u_nom, goal, use_slack):
        start_time = time.perf_counter()
        h_hmax, _, grad_h, h_f, h_G, info_h = self.cert_batch.get_value_and_grad(x, self.noise_selection_cbf(),
                                                                            goal, include_h0=self.include_h0)
        V_max, grad_V, v_f0, v_G0, rate, V = self._pclf_terms(x, goal)
        dV_dt = info_h["dV_dt"]
        self.dvdt_log.append(dV_dt.copy())
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=use_slack)
        self._add_clf_constraint(G, HG, V_max, grad_V, v_f0, v_G0, use_slack=use_slack, rate=rate)
        self._add_cbf_constraints(G, HG, h_hmax, grad_h, h_f, h_G, pad, dV_dt)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack,
                                                    n_cbf=len(h_hmax))
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            delta = 0.0
            self.lam_cbf_log.append(np.full(self.cert.nh, np.nan))
            self.cbf_active_log.append(False)

        # hdot / -alpha*h along the applied input (NaN row if infeasible). Replaces the old assert loop.
        self._pcbf_terms(h_hmax, grad_h, h_f, h_G, dV_dt,
                        None if intervening == "infeasible" else u_act)

        solve_dt = time.perf_counter() - start_time

        Vdot = np.nan
        alV = np.nan
        if intervening != "infeasible":
            Vdot = grad_V @ (v_f0 + v_G0 @ u_act)
            alV = -rate
            assert Vdot <= alV + delta + 1e-7, (
                f"CLF violated at "
                f"Vdot={Vdot:.4f}")

        return u_act, intervening, solve_dt, V_max, delta, h_hmax, Vdot, alV, V

    def two_step_pclf_pcbf(self, x, u_nom, goal, use_slack):
        start_time = time.perf_counter()
        V_max, grad_V, v_f0, v_G0, rate, V = self._pclf_terms(x, goal)
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=use_slack)
        self._add_clf_constraint(G, HG, V_max, grad_V, v_f0, v_G0, use_slack=use_slack, rate=rate)
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
            self._log_cbf_nan()          # CBF never evaluated this step: keep logs aligned

            return (u_act, intervening, solve_dt, V_max, delta,
                    np.full(self.cert.nh, np.nan), np.nan, np.nan, V, u_nom)

        h_hmax, _, grad_h, h_f, h_G, info_h = self.cert_batch.get_value_and_grad(x, self.noise_selection_cbf(),
                                                                            goal, include_h0=self.include_h0)
        dV_dt = info_h["dV_dt"]
        self.dvdt_log.append(dV_dt.copy())
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=False)
        q[:self.dyn.nu] = np.asarray(u_nom, dtype=float)
        self._add_cbf_constraints(G, HG, h_hmax, grad_h, h_f, h_G, pad, dV_dt)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        # Second QP block:
        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=False,
                                                    n_cbf=len(h_hmax))
            intervening = bool(np.linalg.norm(u_act - u_nom) >= self.inter_input)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.zeros(self.dyn.nu)
            intervening = "infeasible"
            delta = 0.0
            self.lam_cbf_log.append(np.full(self.cert.nh, np.nan))
            self.cbf_active_log.append(False)

        # hdot / -alpha*h along the applied input (NaN row if infeasible). Replaces the old assert loop.
        self._pcbf_terms(h_hmax, grad_h, h_f, h_G, dV_dt,
                        None if intervening == "infeasible" else u_act)

        solve_dt = time.perf_counter() - start_time
        Vdot = np.nan
        alV = np.nan

        if intervening != "infeasible":
            Vdot = grad_V @ (v_f0 + v_G0 @ u_act)
            alV = -rate

        return u_act, intervening, solve_dt, V_max, delta_clf, h_hmax, Vdot, alV, V, u_nom