import os
from re import U
import time
from matplotlib.pyplot import sca
import numpy as np
import quadprog

from Backup_pure import backup_filter
from Dynamics import sys_dynm_dd
from Control_policy import policy
from Lyapunov_function import v_certificate
from Barrier_function import h_certificate
from Barrier_function_batch import h_certificate_batch
from Noise_sampler import noise_train_sampler, noise_test_sampler
from Plot_results import (plot_trajectories, plot_h_history, plot_v_history)

class policy_filter:
    def __init__(self, controller, h_controller, v_controller, obstacles, noise_choice):
        self.noise_choice = noise_choice

        # Simulation parameters
        self.T_rollout   = 1.5      # s, certificate lookahead
        self.T_dstb_hold = 0.3      # s, piecewise-constant disturbance interval
        self.T_sim       = 100.0    # s, sim length
        self.dt          = 0.005    # s, sim step size

        # General parameters
        self.barrier_inflate = 0.0  # margin for safety (to avoid numerical issues)
        self.v_max = 1.0            # m/s, max linear velocity
        self.v_min = 0.0          # m/s, min linear velocity, applied more so for the QP
        self.om_max = 3.0           # rad/s, max angular velocity
        self.goal = np.array([4.5, 4.5]) # goal position in the plane in meters [x,y]

        # Noise parameters
        self.n_samples = 50
        self.n_samples_uniform = 25
        self.d_scale = 0.05

        # cbf parameters
        self.alpha = 2.0
        self.inter_input = 1e-3

        # clf parameters
        self.slack_weight = 100
        self.gamma = 0.2 
        
        self.horizon       = int(round(self.T_rollout / self.dt))
        self.interval_size = int(round(self.T_dstb_hold / self.dt))
        self.n_steps_sim   = int(round(self.T_sim / self.dt))
        
        if obstacles == "single":
            self.obs_pos = np.array([[2.0, 2.5]])
            self.R_O = np.array([0.3]) + self.barrier_inflate
        elif obstacles == "multi":
            self.obs_pos = np.array([[2.0, 2.5], [3.0, 3.5], [1.5, 1.8]])
            self.R_O = np.array([0.3, 0.2, 0.1]) + self.barrier_inflate
        else:
            raise ValueError("obstacles must be 'single' or 'multi'")

        self.policy = policy(v_max=self.v_max, 
                             om_max=self.om_max,
                             obs_pos=self.obs_pos, 
                             eps=0.6, 
                             goal=self.goal, 
                             process="batch")
             
        self.dyn = sys_dynm_dd(policy_class=self.policy, 
                               dt=self.dt, 
                               ivp_method="manual_RK4",
                               process="batch")
        
        self.cert = h_certificate(dynamic_class=self.dyn,
                                  obs_pos=self.obs_pos, 
                                  R_O=self.R_O,
                                  policy_h="policy_h",
                                  policy_name=h_controller,
                                  delta = 0.0)

        self.cert_batch = h_certificate_batch(dynamic_class=self.dyn,
                                              obs_pos=self.obs_pos, 
                                              R_O=self.R_O,
                                              hcert_class=self.cert)

        self.clf = v_certificate(dynamic_class=self.dyn, 
                                 policy_name=v_controller, 
                                 goal=self.goal)

        self.rng = np.random.default_rng(12345)
        self.test_noise = noise_test_sampler(nd=self.dyn.nd, rng=self.rng)
        self.train_noise = noise_train_sampler(nd=self.dyn.nd, rng=self.rng)

    def u_nominal(self, x, controller):
        if controller == "proportional_policy":
            u = self.policy.proportional_policy(x)

        elif controller == "constant_policy":
            u = self.policy.constant_policy(x)

        elif controller == "backup_policy":
            u = self.policy.backup_policy(x)

        elif controller == "random_policy":
            u = self.policy.random_policy(x)

        else:
            raise ValueError(f"Invalid controller: {controller}.")

        u = np.asarray(u, dtype=float)

        if self.dyn.process == "batch":
            u = u.reshape(-1, self.dyn.nu)

            assert u.shape[0] == 1, \
                "Main simulation expects one nominal control"

            return u[0]

        return u.reshape(self.dyn.nu)

    def noise_selection(self, override=False):
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
                        horizon=self.horizon,
                        interval_size=self.interval_size,
                        scale=self.d_scale)

        elif self.noise_choice == "Uniform":
            BH_dstb_train, _ = self.train_noise.uniform_train(
                        n_samples=sample,
                        horizon=self.horizon,
                        interval_size=self.interval_size,
                        scale=self.d_scale)

        elif self.noise_choice == "Zero":
            BH_dstb_train, _ = self.train_noise.zero_train(
                        n_samples=1,
                        horizon=self.horizon,
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

    def propagate(self, x, u, d):
        x_next = self.dyn.solve_ivp_fun(x0=x, d=d, u=u)
        x_next = np.asarray(x_next)
        if self.dyn.process == "batch":
            x_next = x_next.reshape(-1, self.dyn.nx)
            assert x_next.shape[0] == 1, \
                "Main simulation expects one propagated state"
            return x_next[0]
        return x_next.reshape(self.dyn.nx)

    def _init_qp(self, u_nom, use_slack=False):
        nu = self.dyn.nu
        n_z = nu + 1 if use_slack else nu
        M = np.eye(n_z)
        if use_slack:
            M[nu, nu] = self.slack_weight
        q = np.zeros(n_z)
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

    def _add_clf_constraint(self, G: list, HG: list, V, grad_V, f_x, g_x, use_slack=False):
        LfV = grad_V @ f_x
        LGV = grad_V @ g_x

        row = list(-LGV)

        if use_slack:
            row.append(1.0)

        G.append(row)
        HG.append(LfV + self.gamma * V)

        if use_slack:
            slack_row = [0.0] * self.dyn.nu + [1.0]
            G.append(slack_row)
            HG.append(0.0)

    def _add_cbf_constraints(self, G: list, HG: list, h_values, grad_h, h_f, h_G, pad):
        for j in range(len(h_values)):
            LfH = grad_h[j] @ h_f[j]
            LGH = grad_h[j] @ h_G[j]

            G.append(pad(list(-LGH)))
            HG.append(LfH + self.alpha * h_values[j])

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

    def _solve_qp(self, M, q, G, HG, u_nom, use_slack=False):
        nu = self.dyn.nu
        qp_sol = quadprog.solve_qp(M,q,np.asarray(G, dtype=float).T,np.asarray(HG, dtype=float),0)
        u_act = qp_sol[0][:nu]
        delta = (qp_sol[0][nu] if use_slack else 0.0)
        intervening = (np.linalg.norm(u_act - u_nom)>= self.inter_input)
        return u_act, intervening, delta
    
    def clf_qp(self, x, u_nom, d_nom, use_slack=False):
        start_time = time.perf_counter()
        V = self.clf.clf_value(x)
        grad_V = self.clf.clf_value_gradient(x)
        f_x = self.dyn.f(x, d_nom)
        g_x = self.dyn.G(x, d_nom)
        M, q, G, HG, _ = self._init_qp(u_nom, use_slack=use_slack)
        self._add_clf_constraint(G, HG, V, grad_V, f_x, g_x, use_slack=use_slack)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
        except Exception as e:
            print("QP failed:", e)
            u_act = np.array([np.clip(u_nom[0], 0.0, self.v_max), np.clip(u_nom[1], -self.om_max, self.om_max)])
            intervening = "infeasible"
            delta = 0.0
        solve_dt = time.perf_counter() - start_time

        if not isinstance(intervening, str):
            Vdot = grad_V @ (f_x + g_x @ u_act)
            assert Vdot <= -self.gamma * V + delta + 1e-7, (
                f"CLF violated at "
                f"Vdot={Vdot:.4f}")
            
        return u_act, intervening, solve_dt, V, delta

    def rpcbf_qp(self, x, u_nom):
        start_time = time.perf_counter()
        h_hmax, _, grad_h, h_f, h_G, _ = self.cert_batch.get_value_and_grad(x, self.noise_selection(), include_h0=False)
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=False)
        self._add_cbf_constraints(G, HG, h_hmax, grad_h, h_f, h_G, pad)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, _ = self._solve_qp(M, q, G, HG, u_nom, use_slack=False)
        except Exception as e:
            print("QP failed:", e)
            j = int(np.argmax(h_hmax))
            LGH = grad_h[j] @ h_G[j]
            u_lim = np.array([self.v_max, self.om_max])
            u_act = np.clip(-np.sign(LGH) * u_lim, -u_lim, u_lim)
            u_act[0] = np.clip(u_act[0], 0.0, self.v_max)
            intervening = "infeasible"
        solve_dt = time.perf_counter() - start_time

        return u_act, intervening, solve_dt, h_hmax

    def pclf_qp(self, x, u_nom, use_slack=False):
        start_time = time.perf_counter()
        v_vmax, _, grad_v, v_f, v_G, _ = self.clf.get_value_and_grad(x, self.noise_selection(override=True), include_v0=True)
        V = v_vmax[0]
        grad_V = grad_v[0]
        M, q, G, HG, _ = self._init_qp(u_nom, use_slack=use_slack)
        self._add_clf_constraint(G, HG, V, grad_V, v_f[0], v_G[0], use_slack=use_slack)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])
        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)

        except Exception as e:
            print("QP failed:", e)
            u_act = np.array([np.clip(u_nom[0], 0.0, self.v_max), np.clip(u_nom[1], -self.om_max, self.om_max)])
            intervening = "infeasible"
            delta = 0.0

        solve_dt = time.perf_counter() - start_time

        if not isinstance(intervening, str):
            Vdot = grad_V @ (v_f[0] + v_G[0] @ u_act)
            assert Vdot <= -self.gamma * V + delta + 1e-7, f"P-CLF violated: Vdot={Vdot:.4f}"

        return u_act, intervening, solve_dt, V, delta

    def clf_cbf_qp(self, x, u_nom, d_nom, use_slack=False):
        start_time = time.perf_counter()
        h_hmax, _, grad_h, h_f, h_G, _ = self.cert_batch.get_value_and_grad(x, self.noise_selection(), include_h0=False)
        V = self.clf.clf_value(x)
        grad_V = self.clf.clf_value_gradient(x)
        f_x = self.dyn.f(x, d_nom)
        g_x = self.dyn.G(x, d_nom)
        M, q, G, HG, pad = self._init_qp(u_nom, use_slack=use_slack)
        self._add_cbf_constraints(G, HG, h_hmax, grad_h, h_f, h_G, pad)
        self._add_clf_constraint(G, HG, V, grad_V, f_x, g_x, use_slack=use_slack)
        G, HG = self._finalize_constraints(G, HG, M.shape[0])

        try:
            u_act, intervening, delta = self._solve_qp(M, q, G, HG, u_nom, use_slack=use_slack)
        except Exception as e:
            print("QP failed:", e)

            if len(h_hmax) > 0:
                j = int(np.argmax(h_hmax))
                LGH = grad_h[j] @ h_G[j]
                u_lim = np.array([self.v_max, self.om_max])
                u_act = np.clip(-np.sign(LGH) * u_lim, -u_lim, u_lim)
                u_act[0] = np.clip(u_act[0], 0.0, self.v_max)

            else:
                u_act = np.array([np.clip(u_nom[0], 0.0, self.v_max), np.clip(u_nom[1], -self.om_max, self.om_max)])

            intervening = "infeasible"
            delta = 0.0
        solve_dt = time.perf_counter() - start_time

        if not isinstance(intervening, str):
                Vdot = grad_V @ (f_x + g_x @ u_act)
                assert Vdot <= -self.gamma * V + delta + 1e-7
                for j in range(len(h_hmax)):
                    hdot = grad_h[j] @ (h_f[j] + h_G[j] @ u_act)
                    assert hdot <= -self.alpha * h_hmax[j] + 1e-7, (
                        f"CBF row {j} "
                        f"violated at "
                        f"hdot={hdot:.4f}"
                    )

        return u_act, intervening, solve_dt, V, delta, h_hmax

    def clf_cbf_backup_qp(self, x, u_nom, d_nom, use_slack=False):
        # assert self.cert.policy_h == "backup_h"
        assert self.cert.policy_name == "backup_policy"
        return self.clf_cbf_qp(x, u_nom, d_nom, use_slack)
       

def print_step_summary(kk,x,u_nom,u_act,solve_dt,intervening,
    goal=None,h_values=None,V=None,delta=None,value_name="CLF"):
    status = (
        "INTERVENE" if intervening is True
        else intervening if isinstance(intervening, str)
        else "NOMINAL"
    )
    lines = [
        f"\n[{kk:2d}] Step Summary",
        f"  State          : x = [{x[0]:7.4f}, {x[1]:7.4f}, {x[2]:7.4f}]",
        f"  Nominal control: u_nom = [{u_nom[0]:6.3f}, {u_nom[1]:6.3f}]",
        f"  Actual control : u_act = [{u_act[0]:6.3f}, {u_act[1]:6.3f}]",
    ]
    if h_values is not None:
        h_str = ", ".join(f"{h:7.4f}" for h in np.atleast_1d(h_values))
        lines.append(f"  Barrier values : h = [{h_str}]")

    if V is not None:
        if delta is not None:
            lines.append(f"  {value_name:<15}: V = {V:8.4f}   slack = {delta:8.4f}")
        else:
            lines.append(f"  {value_name:<15}: V = {V:8.4f}")
    lines.append(f"  Solve time     : {solve_dt * 1000:.2f} ms")
    lines.append(f"  Status         : {status}")
    if goal is not None:
        distance = np.linalg.norm(goal - x[:2])
        lines.append(f"  Distance goal  : {distance:.4f}")
    print("\n".join(lines))

def run_simulation(method, x_s, controller,h_controller ,v_controller, rollout_noise, env_noise, no_obs):
    safety = policy_filter(controller=controller,
                           h_controller=h_controller,
                           v_controller=v_controller ,
                           obstacles=no_obs, 
                           noise_choice=rollout_noise)
    backup_safety = backup_filter(policy_class=safety)
    valid_methods = {"rpcbf", "clf", "clf_cbf", "pclf", "clf_cbf_backup", "pure_backup", "None"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}")
    x_s = np.array(x_s)
    trajectory_actual = [x_s.copy()]
    applied_u = []
    h_now_log = []
    h_hmax_log = []
    V_log = []
    delta_log = []
    print_summary = True
    for kk in range(safety.n_steps_sim):
        x_control = x_s.copy()
        d_env = safety.noise_single(env_noise, kk)
        u_nom = safety.u_nominal(x_control, controller)
        h_hmax = None
        V = None
        delta = None
      
        if method == "rpcbf":
            u_act, intervening, solve_dt, h_hmax = safety.rpcbf_qp(x=x_control, u_nom=u_nom)
            h_now = safety.cert.h_function(x_control)
            h_now_log.append(h_now)
            h_hmax_log.append(h_hmax)

        elif method == "clf":
            u_act, intervening, solve_dt, V, delta = safety.clf_qp(u_nom=u_nom, 
                                                                   x=x_control, 
                                                                   d_nom=d_env, 
                                                                   use_slack=False)
            V_log.append(V)
            delta_log.append(delta if not isinstance(intervening, str) else np.nan)

        elif method == "clf_cbf":
            u_act, intervening, solve_dt, V, delta, h_hmax = safety.clf_cbf_qp(x=x_control, 
                                                                               u_nom=u_nom, 
                                                                               d_nom=d_env, 
                                                                               use_slack=True)
            h_now_log.append(safety.cert.h_function(x_control))
            h_hmax_log.append(h_hmax)
            V_log.append(V)
            delta_log.append(delta if not isinstance(intervening, str) else np.nan)

        elif method == "pclf":
            u_act, intervening, solve_dt, V, delta = safety.pclf_qp(x=x_control, 
                                                                    u_nom=u_nom,
                                                                    use_slack=True)
            V_log.append(V)
            delta_log.append(delta if not isinstance(intervening, str) else np.nan)

        elif method == "clf_cbf_backup":
            u_act, intervening, solve_dt, V, delta, h_hmax = safety.clf_cbf_backup_qp(x=x_control, 
                                                                                      u_nom=u_nom, 
                                                                                      d_nom=d_env, 
                                                                                      use_slack=True)
            h_now_log.append(safety.cert.h_function(x_control))
            
            
            delta_log.append(delta if not isinstance(intervening, str) else np.nan)

        elif method == "pure_backup":
            u_act, intervening, solve_dt = backup_safety.safety_Bcbf(x=x_control,
                                                                     u_nom=u_nom, 
                                                                     d_nom=d_env)
            h_now = safety.cert.h_function(x_control)
            h_now_log.append(h_now)
            h_hmax = np.zeros_like(h_now)
            V = 0.0
            delta = 0.0
            delta_log.append(
                delta if not isinstance(intervening, str) else np.nan
            )
            h_hmax_log.append(h_hmax)
            V_log.append(V)

        elif method == "None":
            u_act=u_nom
            solve_dt = 0.0
            intervening = "None"
            h_hmax = 0.0
            V = 0.0
            delta = 0.0

        x_s = safety.propagate(x=x_control, u=u_act, d=d_env)
        trajectory_actual.append(x_s.copy())
        applied_u.append(np.asarray(u_act).copy())

        value_name = {"clf": "CLF value", "pclf": "P-CLF value", "clf_cbf": "CLF value"}.get(method, "CLF")
        if print_summary == True:
            print_step_summary(kk=kk, x=x_s, u_nom=u_nom, u_act=u_act, solve_dt=solve_dt, 
                            intervening=intervening, goal=safety.goal, h_values=h_hmax, 
                            V=V, delta=delta, value_name=value_name)
        if np.linalg.norm(safety.goal - x_s[:2]) < 0.1:
            print("Goal reached!")
            break

        if kk >= 10000:
            print(f"Early stop at step:{kk}")
            break

        if intervening == "infeasible":
            print(f"QP stopped due to infeasibility at step {kk}")
            break

    # Plotting
    trajectory_actual = np.asarray(trajectory_actual)
    applied_u = np.asarray(applied_u)
    h_now_log = np.asarray(h_now_log)
    h_hmax_log = np.asarray(h_hmax_log)
    V_log = np.asarray(V_log)
    delta_log = np.asarray(delta_log)
    obstacles = [(safety.obs_pos[i, 0], safety.obs_pos[i, 1], safety.R_O[i]) for i in range(len(safety.R_O))]
    plot_trajectories(states_list=trajectory_actual, inputs_list=applied_u, goal = safety.goal,
                      obstacles=obstacles, dt=safety.dt, title=method.upper(), results_dir="Results")
    if len(h_now_log) > 0:
        plot_h_history(h_now=h_now_log, h_hmax=h_hmax_log, 
                       dt=safety.dt, path=os.path.join("Results", f"{method}_h_history.png"))

    if len(V_log) > 0:
        label = "V_pclf(x)" if method == "pclf" else "V(x)"
        title = "P-CLF value" if method == "pclf" else "CLF value"
        plot_v_history(V_log=V_log, delta_log=delta_log, dt=safety.dt, 
                       path=os.path.join("Results", f"{method}_V_history.png"), 
                       value_label=label, title=title)
    return {"states": trajectory_actual, "inputs": applied_u, "h_now": h_now_log, 
            "h_hmax": h_hmax_log, "V": V_log, "delta": delta_log, "safety": safety}


if __name__ == "__main__":
    results = run_simulation(method="clf_cbf", 
                             x_s=[0.5, 2.5, 0.0], 
                             controller="proportional_policy",
                             h_controller="backup_policy", 
                             v_controller="proportional_policy",
                             rollout_noise="Uniform",
                             env_noise="Uniform", 
                             no_obs="multi")

