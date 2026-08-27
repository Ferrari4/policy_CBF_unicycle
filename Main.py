import os
import time
import numpy as np
import quadprog

from Dynamics import sys_dynm_dd
from Control_policy import policy
from Lyapunov_function import v_certificate
from Barrier_function import h_certificate
from Barrier_function_batch import h_certificate_batch
from Noise_sampler import noise_train_sampler, noise_test_sampler
from Plot_results import (plot_trajectories, plot_h_history, plot_v_history)

class policy_filter:
    def __init__(self, controller="proportional_policy", obstacles="single", noise_choice="BangBang"):
        self.controller = controller
        self.noise_choice = noise_choice

        # Simulation parameters
        self.T_rollout   = 1.5      # s, certificate lookahead
        self.T_dstb_hold = 0.3      # s, piecewise-constant disturbance interval
        self.T_sim       = 100.0    # s, sim length
        self.dt          = 0.005    # s, sim step size

        # General parameters
        self.barrier_inflate = 0.0 # margin for safety (to avoid numerical issues)
        self.v_max = 1.0            # m/s, max linear velocity
        self.om_max = 3.0           # rad/s, max angular velocity
        self.goal = np.array([4.5, 4.5]) # goal position in the plane in meters [x,y]

        # Noise parameters
        self.n_samples = 50
        self.n_samples_uniform = 25
        self.d_scale = 0.1

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
                                  policy_name="proportional_policy",
                                  delta = 0.0)

        self.cert_batch = h_certificate_batch(dynamic_class=self.dyn,
                                              obs_pos=self.obs_pos, 
                                              R_O=self.R_O,
                                              hcert_class=self.cert)

        self.rng = np.random.default_rng(12345)
        self.test_noise = noise_test_sampler(nd=self.dyn.nd, rng=self.rng)
        self.train_noise = noise_train_sampler(nd=self.dyn.nd, rng=self.rng)

    def u_nominal(self, x):
        if self.controller == "proportional_policy":
            u = self.policy.proportional_policy(x)

        elif self.controller == "constant_policy":
            u = self.policy.constant_policy(x)

        elif self.controller == "backup_policy":
            u = self.policy.backup_policy(x)

        else:
            raise ValueError(f"Invalid controller: {self.controller}.")

        u = np.asarray(u, dtype=float)

        if self.dyn.process == "batch":
            u = u.reshape(-1, self.dyn.nu)

            assert u.shape[0] == 1, \
                "Main simulation expects one nominal control"

            return u[0]

        return u.reshape(self.dyn.nu)

    def noise_selection(self):
        if self.noise_choice == "BangBang":
            BH_dstb_train, _ = self.train_noise.bangbang_uniform_train(
                        n_samples=self.n_samples,
                        n_samples_uniform=self.n_samples_uniform,
                        horizon=self.horizon,
                        interval_size=self.interval_size,
                        scale=self.d_scale)

        elif self.noise_choice == "Uniform":
            BH_dstb_train, _ = self.train_noise.uniform_train(
                        n_samples=self.n_samples,
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

    def propagate(self, x, u, d):

        x_next = self.dyn.solve_ivp_fun(
            x0=x,
            d=d,
            u=u
        )

        x_next = np.asarray(x_next)

        if self.dyn.process == "batch":

            x_next = x_next.reshape(
                -1,
                self.dyn.nx
            )

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
            0.0,
            -self.v_max,
            -self.om_max,
            -self.om_max
        ]

        return M, q, G, HG, pad

    def _add_clf_constraint(self, G, HG, V, grad_V, f_x, g_x, use_slack=False):
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

    def _add_cbf_constraints(self, G, HG, h_values, grad_h, h_f, h_G, pad):
        for j in range(len(h_values)):
            LfH = grad_h[j] @ h_f[j]
            LGH = grad_h[j] @ h_G[j]

            G.append(
                pad(list(-LGH))
            )

            HG.append(
                LfH + self.alpha * h_values[j]
            )

    def _solve_qp(self, M, q, G, HG, u_nom, use_slack=False):
        nu = self.dyn.nu

        qp_sol = quadprog.solve_qp(
            M,
            q,
            np.asarray(G, dtype=float).T,
            np.asarray(HG, dtype=float),
            0
        )

        u_act = qp_sol[0][:nu]

        delta = (
            qp_sol[0][nu]
            if use_slack
            else 0.0
        )

        intervening = (
            np.linalg.norm(u_act - u_nom)
            >= self.inter_input
        )

        return u_act, intervening, delta
    
    def clf_qp(
        self,
        u_nom,
        V,
        grad_V,
        f_x,
        g_x,
        use_slack=False
    ):

        start_time = time.perf_counter()

        M, q, G, HG, _ = self._init_qp(
            u_nom,
            use_slack=use_slack
        )

        self._add_clf_constraint(
            G,
            HG,
            V,
            grad_V,
            f_x,
            g_x,
            use_slack=use_slack
        )

        try:

            u_act, intervening, delta = \
                self._solve_qp(
                    M,
                    q,
                    G,
                    HG,
                    u_nom,
                    use_slack=use_slack
                )

        except Exception as e:

            print("QP failed:", e)

            u_act = np.array([
                np.clip(
                    u_nom[0],
                    0.0,
                    self.v_max
                ),
                np.clip(
                    u_nom[1],
                    -self.om_max,
                    self.om_max
                )
            ])

            intervening = "infeasible"
            delta = 0.0

        solve_dt = (
            time.perf_counter()
            - start_time
        )

        return (
            u_act,
            intervening,
            solve_dt,
            V,
            delta
        )

    def rpcbf_qp(
        self,
        x,
        u_nom
    ):

        start_time = time.perf_counter()

        h_hmax, _, grad_h, h_f, h_G, _ = \
            self.cert_batch.get_value_and_grad(
                x,
                self.noise_selection(),
                include_h0=False
            )

        M, q, G, HG, pad = \
            self._init_qp(
                u_nom,
                use_slack=False
            )

        self._add_cbf_constraints(
            G,
            HG,
            h_hmax,
            grad_h,
            h_f,
            h_G,
            pad
        )

        try:

            u_act, intervening, _ = \
                self._solve_qp(
                    M,
                    q,
                    G,
                    HG,
                    u_nom,
                    use_slack=False
                )

        except Exception as e:

            print("QP failed:", e)

            # Same CBF-oriented fallback you used before.
            j = int(
                np.argmax(h_hmax)
            )

            LGH = (
                grad_h[j]
                @ h_G[j]
            )

            u_lim = np.array([
                self.v_max,
                self.om_max
            ])

            u_act = np.clip(
                -np.sign(LGH) * u_lim,
                -u_lim,
                u_lim
            )

            u_act[0] = np.clip(
                u_act[0],
                0.0,
                self.v_max
            )

            intervening = "infeasible"

        solve_dt = (
            time.perf_counter()
            - start_time
        )

        return (
            u_act,
            intervening,
            solve_dt,
            h_hmax
        )

    def clf_cbf_qp(
        self,
        u_nom,
        V,
        grad_V,
        f_x,
        g_x,
        h_hmax,
        grad_h,
        h_f,
        h_G,
        use_slack=False
    ):

        start_time = time.perf_counter()

        M, q, G, HG, pad = \
            self._init_qp(
                u_nom,
                use_slack=use_slack
            )

        # Hard CBF rows
        self._add_cbf_constraints(
            G,
            HG,
            h_hmax,
            grad_h,
            h_f,
            h_G,
            pad
        )

        # CLF row, optionally soft
        self._add_clf_constraint(
            G,
            HG,
            V,
            grad_V,
            f_x,
            g_x,
            use_slack=use_slack
        )

        try:

            u_act, intervening, delta = \
                self._solve_qp(
                    M,
                    q,
                    G,
                    HG,
                    u_nom,
                    use_slack=use_slack
                )

        except Exception as e:

            print("QP failed:", e)

            if len(h_hmax) > 0:

                j = int(
                    np.argmax(h_hmax)
                )

                LGH = (
                    grad_h[j]
                    @ h_G[j]
                )

                u_lim = np.array([
                    self.v_max,
                    self.om_max
                ])

                u_act = np.clip(
                    -np.sign(LGH) * u_lim,
                    -u_lim,
                    u_lim
                )

                u_act[0] = np.clip(
                    u_act[0],
                    0.0,
                    self.v_max
                )

            else:

                u_act = np.array([
                    np.clip(
                        u_nom[0],
                        0.0,
                        self.v_max
                    ),
                    np.clip(
                        u_nom[1],
                        -self.om_max,
                        self.om_max
                    )
                ])

            intervening = "infeasible"
            delta = 0.0

        solve_dt = (
            time.perf_counter()
            - start_time
        )

        return (
            u_act,
            intervening,
            solve_dt,
            V,
            delta
        )

   
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
            lines.append(
                f"  {value_name:<15}: V = {V:8.4f}   slack = {delta:8.4f}"
            )
        else:
            lines.append(
                f"  {value_name:<15}: V = {V:8.4f}"
            )

    lines.append(f"  Solve time     : {solve_dt * 1000:.2f} ms")
    lines.append(f"  Status         : {status}")

    if goal is not None:
        distance = np.linalg.norm(goal - x[:2])
        lines.append(f"  Distance goal  : {distance:.4f}")

    print("\n".join(lines))


def run_simulation(
    method,
    robust_pclf=False
):

    valid_methods = {
        "rpcbf",
        "clf",
        "clf_cbf",
        "pclf"
    }

    if method not in valid_methods:
        raise ValueError(
            f"Unknown method: {method}"
        )

    x_s = np.array([
        0.5,
        2.5,
        0.0
    ])

    controller = (
        "proportional_policy"
        if method == "rpcbf"
        else "constant_policy"
    )

    safety = policy_filter(
        controller=controller,
        obstacles="multi",
        noise_choice="Zero"
    )

    clf = v_certificate(
        dynamic_class=safety.dyn,
        policy_name="proportional_policy",
        goal=safety.goal
    )

    cert = safety.cert_batch

    d_nom = np.zeros(
        safety.dyn.nd
    )

    trajectory_actual = [
        x_s.copy()
    ]

    applied_u = []

    h_now_log = []
    h_hmax_log = []

    V_log = []
    delta_log = []

    for kk in range(
        safety.n_steps_sim
    ):

        # State on which this control was computed
        x_control = x_s.copy()

        u_nom = safety.u_nominal(
            x_control
        )

        h_hmax = None
        V = None
        delta = None

        # =====================================================
        # RP-CBF
        # =====================================================
        if method == "rpcbf":

            (
                u_act,
                intervening,
                solve_dt,
                h_hmax
            ) = safety.rpcbf_qp(
                x=x_control,
                u_nom=u_nom
            )

            h_now = \
                safety.cert.h_function(
                    x_control
                )

            h_now_log.append(
                h_now
            )

            h_hmax_log.append(
                h_hmax
            )

        # =====================================================
        # Handcrafted CLF
        # =====================================================
        elif method == "clf":

            V = clf.clf_value(
                x_control
            )

            grad_V = \
                clf.clf_value_gradient(
                    x_control
                )

            f_x = safety.dyn.f(
                x_control,
                d_nom
            )

            g_x = safety.dyn.G(
                x_control,
                d_nom
            )

            (
                u_act,
                intervening,
                solve_dt,
                V,
                delta
            ) = safety.clf_qp(
                u_nom=u_nom,
                V=V,
                grad_V=grad_V,
                f_x=f_x,
                g_x=g_x,
                use_slack=False
            )

            # Regression check
            if not isinstance(
                intervening,
                str
            ):

                Vdot = (
                    grad_V
                    @ (
                        f_x
                        + g_x @ u_act
                    )
                )

                assert (
                    Vdot
                    <=
                    -safety.gamma * V
                    + delta
                    + 1e-7
                ), (
                    f"CLF violated at "
                    f"step {kk}: "
                    f"Vdot={Vdot:.4f}"
                )

            V_log.append(V)

            delta_log.append(
                delta
                if not isinstance(
                    intervening,
                    str
                )
                else np.nan
            )

        # =====================================================
        # CLF + RP-CBF
        # =====================================================
        elif method == "clf_cbf":

            (
                h_hmax,
                _,
                grad_h,
                h_f,
                h_G,
                _
            ) = cert.get_value_and_grad(
                x_control,
                safety.noise_selection(),
                include_h0=False
            )

            V = clf.clf_value(
                x_control
            )

            grad_V = \
                clf.clf_value_gradient(
                    x_control
                )

            f_x = safety.dyn.f(
                x_control,
                d_nom
            )

            g_x = safety.dyn.G(
                x_control,
                d_nom
            )

            (
                u_act,
                intervening,
                solve_dt,
                V,
                delta
            ) = safety.clf_cbf_qp(
                u_nom=u_nom,
                V=V,
                grad_V=grad_V,
                f_x=f_x,
                g_x=g_x,
                h_hmax=h_hmax,
                grad_h=grad_h,
                h_f=h_f,
                h_G=h_G,
                use_slack=False
            )

            if not isinstance(
                intervening,
                str
            ):

                Vdot = (
                    grad_V
                    @ (
                        f_x
                        + g_x @ u_act
                    )
                )

                assert (
                    Vdot
                    <=
                    -safety.gamma * V
                    + delta
                    + 1e-7
                )

                for j in range(
                    len(h_hmax)
                ):

                    hdot = (
                        grad_h[j]
                        @ (
                            h_f[j]
                            + h_G[j] @ u_act
                        )
                    )

                    assert (
                        hdot
                        <=
                        -safety.alpha
                        * h_hmax[j]
                        + 1e-7
                    ), (
                        f"CBF row {j} "
                        f"violated at "
                        f"step {kk}: "
                        f"hdot={hdot:.4f}"
                    )

            h_now_log.append(
                safety.cert.h_function(
                    x_control
                )
            )

            h_hmax_log.append(
                h_hmax
            )

            V_log.append(V)

            delta_log.append(
                delta
                if not isinstance(
                    intervening,
                    str
                )
                else np.nan
            )

        # =====================================================
        # Policy-rollout CLF
        # =====================================================
        elif method == "pclf":

            (
                v_vmax,
                _,
                grad_v,
                v_f,
                v_G,
                _
            ) = clf.get_value_and_grad(
                x_control,
                safety.noise_selection(),
                include_v0=True
            )

            if robust_pclf:

                f_x = v_f[0]
                g_x = v_G[0]

            else:

                f_x = safety.dyn.f(
                    x_control,
                    d_nom
                )

                g_x = safety.dyn.G(
                    x_control,
                    d_nom
                )

            (
                u_act,
                intervening,
                solve_dt,
                V,
                delta
            ) = safety.clf_qp(
                u_nom=u_nom,
                V=v_vmax[0],
                grad_V=grad_v[0],
                f_x=f_x,
                g_x=g_x,

                # P-CLF used slack in your
                # previous experiment.
                use_slack=True
            )

            V_log.append(V)

            delta_log.append(
                delta
                if not isinstance(
                    intervening,
                    str
                )
                else np.nan
            )

        # =====================================================
        # Environment disturbance
        #
        # Preserve your earlier experiments:
        # handcrafted CLF propagated with uniform disturbance;
        # the others used zero disturbance.
        # =====================================================
        if method == "clf":

            d_env = \
                safety.test_noise.uniform_test(
                    safety.rng,
                    x=x_control,
                    k=kk,
                    scale=safety.d_scale
                )

        else:

            d_env = \
                safety.test_noise.zero_test(
                    safety.rng,
                    x=x_control,
                    k=kk,
                    scale=safety.d_scale
                )

        # =====================================================
        # New dynamics interface
        # =====================================================
        x_s = safety.propagate(
            x=x_control,
            u=u_act,
            d=d_env
        )

        trajectory_actual.append(
            x_s.copy()
        )

        applied_u.append(
            np.asarray(u_act).copy()
        )

        # =====================================================
        # Summary
        # =====================================================
        value_name = {
            "clf": "CLF value",
            "pclf": "P-CLF value",
            "clf_cbf": "CLF value",
        }.get(
            method,
            "CLF"
        )

        print_step_summary(
            kk=kk,
            x=x_s,
            u_nom=u_nom,
            u_act=u_act,
            solve_dt=solve_dt,
            intervening=intervening,
            goal=safety.goal,
            h_values=h_hmax,
            V=V,
            delta=delta,
            value_name=value_name
        )

        # =====================================================
        # Goal stop
        # =====================================================
        if np.linalg.norm(
            safety.goal - x_s[:2]
        ) < 0.1:

            print("Goal reached!")
            break

    # =========================================================
    # Convert logs
    # =========================================================
    trajectory_actual = np.asarray(
        trajectory_actual
    )

    applied_u = np.asarray(
        applied_u
    )

    h_now_log = np.asarray(
        h_now_log
    )

    h_hmax_log = np.asarray(
        h_hmax_log
    )

    V_log = np.asarray(
        V_log
    )

    delta_log = np.asarray(
        delta_log
    )

    # =========================================================
    # Plot trajectory + controls
    # =========================================================
    obstacles = [
        (
            safety.obs_pos[i, 0],
            safety.obs_pos[i, 1],
            safety.R_O[i]
        )
        for i in range(
            len(safety.R_O)
        )
    ]

    plot_trajectories(
        states_list=trajectory_actual,
        inputs_list=applied_u,
        obstacles=obstacles,
        dt=safety.dt,
        title=method.upper(),
        results_dir="Results"
    )

    # =========================================================
    # Barrier plots
    # =========================================================
    if len(h_now_log) > 0:

        plot_h_history(
            h_now=h_now_log,
            h_hmax=h_hmax_log,
            dt=safety.dt,
            path=os.path.join(
                "Results",
                f"{method}_h_history.png"
            )
        )

    # =========================================================
    # CLF / P-CLF plots
    # =========================================================
    if len(V_log) > 0:

        label = (
            "V_pclf(x)"
            if method == "pclf"
            else "V(x)"
        )

        title = (
            "P-CLF value"
            if method == "pclf"
            else "CLF value"
        )

        plot_v_history(
            V_log=V_log,
            delta_log=delta_log,
            dt=safety.dt,
            path=os.path.join(
                "Results",
                f"{method}_V_history.png"
            ),
            value_label=label,
            title=title
        )

    return {
        "states": trajectory_actual,
        "inputs": applied_u,
        "h_now": h_now_log,
        "h_hmax": h_hmax_log,
        "V": V_log,
        "delta": delta_log,
        "safety": safety
    }

if __name__ == "__main__":

    method = "pclf"

    results = run_simulation(
        method=method,
        robust_pclf=False
    )