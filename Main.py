import os
import time
import numpy as np
import quadprog
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

from Dynamics import sys_dynm_dd
from Control_policy import policy
from Lyapunov_function import v_certificate
from Value_function import h_certificate
from Value_function_batch import h_certificate_batch
from Sim_unicycle import UnicycleSim
from Noise_sampler import noise_train_sampler, noise_test_sampler

class policy_filter:
    def __init__(self, controller="proportional_policy", obstacles="single", noise_choice="BangBang"):
        self.controller = controller
        self.noise_choice = noise_choice

        # Simulation parameters
        self.T_rollout   = 1.5      # s, certificate lookahead
        self.T_dstb_hold = 0.3      # s, piecewise-constant disturbance interval
        self.T_sim       = 100.0    # s, sim length
        self.dt          = 0.005     # s, sim step size

        # General parameters
        self.barrier_inflate = 0.03 # margin for safety (to avoid numerical issues)
        self.v_max = 1.0            # m/s, max linear velocity
        self.om_max = 3.0           # rad/s, max angular velocity
        self.goal = np.array([4.0, 4.0]) # goal position in the plane in meters [x,y]

        # Noise parameters
        self.grid_size = 12 # only for plotting the mesh grid, not for training
        self.n_samples = 50
        self.n_samples_uniform = 25
        self.d_scale = 0.1

        # cbf parameters
        self.alpha = 2.0
        self.inter_input = 1e-3
        self.slack_penalty = 1e4
        
        self.horizon       = int(round(self.T_rollout / self.dt))
        self.interval_size = int(round(self.T_dstb_hold / self.dt))
        self.n_steps_sim   = int(round(self.T_sim / self.dt))
        
        if obstacles == "single":
            self.obs_pos = np.array([[2.0, 2.5]])
            self.R_O = np.array([0.3]) + self.barrier_inflate
        else:
            self.obs_pos = np.array([[2.0, 2.5], [3.0, 3.5], [1.5, 1.8]])
            self.R_O = np.array([0.3, 0.2, 0.1]) + self.barrier_inflate

        self.visual_sim = UnicycleSim()
        for i,obp in enumerate(self.obs_pos):
            self.visual_sim.add_obstacle(obp[0], obp[1], self.R_O[i])

        self.policy = policy(v_max=self.v_max, om_max=self.om_max,
                             obs_pos=self.obs_pos, eps=0.6, goal=self.goal)
        
        self.dyn = sys_dynm_dd(policy_class=self.policy, dt=self.dt)
        
        self.cert = h_certificate(dynamic_class=self.dyn,
                                  obs_pos=self.obs_pos, R_O=self.R_O,
                                  policy_h="policy_h",
                                  policy_name="proportional_policy",
                                  ivp_method="manual_RK4",
                                  delta = 0.0)

        self.cert_batch = h_certificate_batch(dynamic_class=self.dyn,
                                    obs_pos=self.obs_pos, R_O=self.R_O,
                                    hcert_class=self.cert,
                                    policy_h="policy_h",
                                    policy_name="proportional_policy",
                                    ivp_method="manual_RK4",
                                    delta=0.0)

        self.rng = np.random.default_rng(12345)
        self.test_noise = noise_test_sampler(nd=self.dyn.nd, rng=self.rng)
        self.train_noise = noise_train_sampler(nd=self.dyn.nd, rng=self.rng)
  
    def create_mesh_grid(self, xy_range=(0.0, 5.0), theta_fixed=None):
        lo, hi = xy_range
        b_x = np.linspace(lo, hi, num=self.grid_size)
        b_y = np.linspace(lo, hi, num=self.grid_size)
        XX, YY = np.meshgrid(b_x, b_y, indexing="ij")
        if theta_fixed is None:
            theta_fixed = 0.0
        TT = np.full_like(XX, theta_fixed)
        grid_state = np.stack([XX, YY, TT], axis=-1)
        return grid_state.reshape(-1, self.dyn.nx), b_x, b_y

    def u_nominal(self, x):
        if self.controller == "proportional_policy":
            return self.policy.proportional_policy(state=x)
        elif self.controller == "constant_policy":
            return self.policy.constant_policy(state=x)
        elif self.controller == "backup_policy":
            return self.policy.backup_policy(state=x)
        else:
            raise ValueError(f"Invalid controller: {self.controller}. Must be 'proportional_policy', 'constant_policy', or 'backup_policy'.")

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
                        n_samples=1.0,
                        horizon=self.horizon,
                        interval_size=self.interval_size)

        else:
            raise ValueError(f"Invalid noise_choice: {self.noise_choice}. Must be 'BangBang', 'Uniform', or 'Zero'.")       

        return BH_dstb_train
            
    def safety_rpcbf(self, x, u_nom):
        start_time = time.perf_counter()

        self.cert = self.cert_batch  # Use the batch version of the certificate for efficiency

        h_hmax, hH_dstb, grad_h_hmax, h_f, h_G, info = self.cert.get_value_and_grad(
        x, self.noise_selection(), include_h0=False)

        M = np.eye(self.dyn.nu)
        q = np.asarray(u_nom, dtype=float)

        # Input box: |v| <= v_max, |om| <= om_max
        # (swap for the wheel-diamond |v|/v_max + |om|/om_max <= 1 later)
        G = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        HG = [0.0, -self.v_max, -self.om_max, -self.om_max]

        for j in range(self.cert.nh):
            LfV = grad_h_hmax[j] @ h_f[j]
            LGV = grad_h_hmax[j] @ h_G[j]

            constrain_r = LfV + self.alpha * h_hmax[j]
            constrain_l = list(-LGV)

            G.append(constrain_l)
            HG.append(constrain_r)

        try:
            qp_sol = quadprog.solve_qp(M, q, np.array(G).T, np.array(HG), 0)
            u_act = qp_sol[0][:self.dyn.nu]

        except Exception as e:
            print("QP failed:", e)
            j = int(np.argmax(h_hmax))
            LGV = grad_h_hmax[j] @ h_G[j]
            u_lim = np.array([self.v_max, self.om_max])
            u_act = np.clip(-np.sign(LGV) * u_lim, -u_lim, u_lim)
            intervening = "infeasible"
            solve_dt = time.perf_counter() - start_time

            return u_act, intervening, solve_dt, h_hmax

        if np.linalg.norm(u_act - u_nom) >= self.inter_input:
            intervening = True
        else:
            intervening = False

        solve_dt = time.perf_counter() - start_time

        return u_act, intervening, solve_dt, h_hmax

    def evaluate_mesh(self, theta_fixed=0.0, include_h0=True):
        x0, b_x, b_y = self.create_mesh_grid(theta_fixed=theta_fixed)
        xs = x0.shape[0]
        side = self.grid_size

        # one shared disturbance batch for the whole grid, so V is deterministic
        bH_dstb, _ = self.train_noise.bangbang_uniform_train(
            n_samples=self.n_samples,
            n_samples_uniform=self.n_samples_uniform,
            horizon=self.horizon,
            interval_size=self.interval_size,
            scale=self.d_scale,
        )

        h_hmax = np.zeros((xs, self.cert.nh))
        for s in range(xs):
            h_hmax[s] = self.cert.compute_h_hmax_from_dstb(x0[s], bH_dstb, 
                        include_h0=include_h0, max_type="cubic_spline")
        out = {
            "b_x":     b_x,
            "b_y":     b_y,
            "theta":   theta_fixed,
            "h_hmax":  h_hmax.reshape(side, side, self.cert.nh),
            "h_total": h_hmax.max(axis=1).reshape(side, side),
            "obs_pos": self.obs_pos,
            "R_O":     self.R_O,
        }

        return out

def plot_h_history(h_now, h_hmax, dt, obs_pos=None, path=None):
    """h_now, h_hmax: (T, nh).  Convention: h > 0 unsafe, safe set = {h <= 0}."""
    if path is None:
        path = os.path.join("Results", "h_history.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    h_now = np.asarray(h_now)
    h_hmax = np.asarray(h_hmax)
    t = np.arange(h_now.shape[0]) * dt
    nh = h_now.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, (V, title) in zip(axes, [(h_now,  "h(x_t)  -  current state"),
                                     (h_hmax, "h_hmax  -  worst-case lookahead")]):
        for j in range(nh):
            ax.plot(t[:V.shape[0]], V[:, j], lw=1.6, label=f"obs {j}")
        ax.axhline(0.0, color="k", ls="--", lw=1.2)
        ax.axhspan(0.0, max(1e-3, np.nanmax(V)), color="red", alpha=0.06)
        ax.set_ylabel("h")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    viol = np.where(h_now.max(axis=1) > 0.0)[0]
    if viol.size:
        print(f"SAFETY VIOLATION: h > 0 at {viol.size} steps, "
              f"first t = {viol[0] * dt:.2f}s, max h = {h_now.max():.4f}")
    else:
        print(f"No violation. Closest approach: h_max = {h_now.max():.4f} "
              f"(margin {-h_now.max():.4f})")

def plot_mesh(out, traj=None, path=None):
    if path is None:
        path = os.path.join("Results", "rpcbf_contour.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    b_x, b_y = out["b_x"], out["b_y"]
    XX, YY = np.meshgrid(b_x, b_y, indexing="ij")
    nh = out["h_hmax"].shape[2]

    panels = [("Total", out["h_total"])]
    for j in range(nh):
        panels.append((f"obs {j}", out["h_hmax"][:, :, j]))

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=2.0)

    handles = [
        Line2D([0], [0], color="magenta", lw=2, label="h = 0 (safe-set boundary)"),
        Line2D([0], [0], color="k", ls="--", label="obstacle"),
        Line2D([0], [0], color="lime", lw=2, label="trajectory"),
        Line2D([0], [0], marker="o", color="lime", mec="k", ls="", label="start"),
        Line2D([0], [0], marker="s", color="lime", mec="k", ls="", label="end"),
    ]
    if traj is None:
        handles = handles[:2]

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (title, V) in zip(axes, panels):
        cf = ax.contourf(XX, YY, np.clip(V, -1, 2),
                         levels=np.linspace(-1, 2, 25), cmap="RdBu_r", norm=norm)
        ax.contour(XX, YY, V, levels=[0.0], colors="magenta", linewidths=2.0)
        for i in range(len(out["R_O"])):
            circ = plt.Circle(out["obs_pos"][i], out["R_O"][i],
                              fill=False, color="k", ls="--", lw=1.2)
            ax.add_patch(circ)
        if traj is not None:
            ax.plot(traj[:, 0], traj[:, 1], color="lime", lw=2.0, zorder=6)
            ax.plot(traj[0, 0], traj[0, 1], "o", color="lime", ms=8, mec="k", zorder=7)
            ax.plot(traj[-1, 0], traj[-1, 1], "s", color="lime", ms=8, mec="k", zorder=7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title}  (theta = {out['theta']:.2f})")
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax)
    axes[0].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main_RP_CBF_QP():
    x_s = np.array([0.5, 2.5, 0.0])
    safety = policy_filter(controller="proportional_policy", 
                           obstacles="multi",
                           noise_choice="Zero")
    bypass = False  # IMPORTANT: set to FALSE for CBF safety intervention, TRUE to bypass and use nominal control only
    trajectory_actual = [x_s]
    applied_u = []
    h_now_log = []
    h_hmax_log = []
    for kk in range(safety.n_steps_sim):
        u_nom = safety.u_nominal(x_s)
        u_act, intervening, dt, h_hmax = safety.safety_rpcbf(x=x_s, u_nom=u_nom)

        d_env = safety.test_noise.zero_test(safety.rng, x=x_s, k=kk, scale=safety.d_scale) # Robot applied noise

        h_now_log.append(safety.cert.h_function_batch(x_s)
                         if hasattr(safety.cert, "h_function_batch")
                         else safety.cert.h_function(x_s))
        h_hmax_log.append(h_hmax)

        if bypass == True:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_nom, d=d_env, method="manual_RK4")
        else:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_act, d=d_env, method="manual_RK4")
        trajectory_actual.append(x_s)
        applied_u.append(u_act)
        status = (
            "INTERVENE" if intervening is True
            else intervening if isinstance(intervening, str)
            else "NOMINAL"
        )

        h_str = ", ".join(f"{h:7.4f}" for h in h_hmax)
        print(
            f"\n[{kk:2d}] Step Summary\n"
            f"  State          : x = [{x_s[0]:7.4f}, {x_s[1]:7.4f}, {x_s[2]:7.4f}]\n"
            f"  Nominal control: u_nom = [{u_nom[0]:6.3f}, {u_nom[1]:6.3f}]\n"
            f"  Actual control : u_act = [{u_act[0]:6.3f}, {u_act[1]:6.3f}]\n"
            f"  Barrier values : h = [{h_str}]\n"
            f"  Solve time     : {dt*1000:.2f} ms\n"
            f"  Status      : {status}" )

        print(f"Distance to goal: {np.linalg.norm(safety.goal - x_s[:2]):.4f}")
        if np.linalg.norm(safety.goal - x_s[:2]) < 0.1:
            print("Goal reached!")
            break

    safety.visual_sim.plot_external_trajectories(states_list=np.array(trajectory_actual), 
                                                 inputs_list=np.array(applied_u))
    
    plot_h_history(np.array(h_now_log), np.array(h_hmax_log), safety.dt,
                path=os.path.join("Results", "h_history.png"))
    
    # out = safety.evaluate_mesh(theta_fixed=0.0,include_h0=False)
    # plot_mesh(out, traj=np.array(trajectory_actual),
    #           path=os.path.join("Results", "rpcbf_check.png"))
    # print("saved rpcbf_check.png")

def main_clf(): 
    # Is using Zero noise model for CLF compute, progating with a noisy model
    x_s = np.array([0.5, 2.5, 0.0])
    safety = policy_filter(controller="constant_policy",
                           obstacles="multi",        # unused this run, keeps init happy
                           noise_choice="Zero")
    
    clf = v_certificate(dynamic_class=safety.dyn,
                        alpha=safety.alpha,              # 2.0 — same gain as result set one
                        v_max=safety.v_max, 
                        om_max=safety.om_max,
                        slack=False)                 # slack ON for the result runs
    bypass = False  # IMPORTANT: set to FALSE 

    trajectory_actual = [x_s]
    applied_u = []
    V_log = []
    delta_log = []

    for kk in range(safety.n_steps_sim):
        
        u_nom = safety.u_nominal(x_s)
        u_act, intervening, dt, V, delta = clf.solve_clf_qp(u_nom=u_nom, x=x_s, goal=safety.goal)
        t0 = time.perf_counter()
        d_env = safety.test_noise.uniform_test(safety.rng, x=x_s, k=kk, scale=safety.d_scale)

        # Optional but cheap: decrease-condition regression check
        grad_V = clf.clf_value_gradient(x_s, safety.goal)
        Vdot = grad_V @ (clf.sys_dynm.f(x_s, [0,0,0]) + clf.sys_dynm.G(x_s, [0,0,0]) @ u_act)
        assert Vdot <= -clf.gamma * V + delta + 1e-7, \
            f"CLF condition violated at step {kk}: Vdot={Vdot:.4f}"

        V_log.append(V)
        delta_log.append(delta if not isinstance(intervening, str) else np.nan)

        if bypass == True:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_nom, d=d_env, method="manual_RK4")
        else:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_act, d=d_env, method="manual_RK4")
        trajectory_actual.append(x_s)
        applied_u.append(u_act)

        status = (
            "INTERVENE" if intervening is True
            else intervening if isinstance(intervening, str)
            else "NOMINAL"
        )
        cert_dt = time.perf_counter() - t0
        print(
            f"\n[{kk:2d}] Step Summary\n"
            f"  State          : x = [{x_s[0]:7.4f}, {x_s[1]:7.4f}, {x_s[2]:7.4f}]\n"
            f"  Nominal control: u_nom = [{u_nom[0]:6.3f}, {u_nom[1]:6.3f}]\n"
            f"  Actual control : u_act = [{u_act[0]:6.3f}, {u_act[1]:6.3f}]\n"
            f"  CLF value      : V = {V:8.4f}   slack = {delta:8.4f}\n"
            f"  Solve time     : {dt+cert_dt*1000:.2f} ms\n"
            f"  Status      : {status}")

        print(f"Distance to goal: {np.linalg.norm(safety.goal - x_s[:2]):.4f}")
        if np.linalg.norm(safety.goal - x_s[:2]) < 0.1:
            print("Goal reached!")
            break

    safety.visual_sim.plot_external_trajectories(states_list=np.array(trajectory_actual),
                                                 inputs_list=np.array(applied_u))

    # V / slack history (analogue of plot_h_history)
    t = np.arange(len(V_log)) * safety.dt
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t, V_log, label="V(x)")
    ax1.set_yscale("log")           # exponential decay -> straight line, slope = -gamma
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("V (log)")
    ax2 = ax1.twinx()
    ax2.plot(t, delta_log, color="tab:red", alpha=0.6, label="slack")
    ax2.set_ylabel("slack")
    fig.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(os.path.join("Results", "clf_only_history.png"), dpi=150)
    print("saved clf_history.png")

def main_clf_cbf():
    # Is using Zero noise model for CLF compute, progating with a noisy model
    x_s = np.array([0.5, 2.5, 0.0])
    safety = policy_filter(controller="constant_policy",
                           obstacles="multi",
                           noise_choice="Zero")          # Zero first; BangBang after
        
    clf = v_certificate(dynamic_class=safety.dyn,
                        alpha=safety.alpha,              # 2.0 — same gain as result set one
                        v_max=safety.v_max, om_max=safety.om_max,
                        slack=False)                      # safety hard, stability soft
    cert = safety.cert_batch                             # use batch cert directly
    bypass = False

    trajectory_actual = [x_s]
    applied_u = []
    V_log, delta_log = [], []
    h_now_log, h_hmax_log = [], []

    for kk in range(safety.n_steps_sim):
        u_nom = safety.u_nominal(x_s)

        t0 = time.perf_counter()
        h_hmax, hH_dstb, grad_h, h_f, h_G, info = cert.get_value_and_grad(
            x_s, safety.noise_selection(), include_h0=False)
        cert_dt = time.perf_counter() - t0

        u_act, intervening, dt, h_hmax, V, delta = clf.solve_clf_cbf_qp(
            u_nom=u_nom, x=x_s, goal=safety.goal,
            h_hmax=h_hmax, grad_h_hmax=grad_h, h_f=h_f, h_G=h_G)

        d_env = safety.test_noise.zero_test(safety.rng, x=x_s, k=kk,
                                               scale=safety.d_scale)

        # Regression checks: CLF soft, CBF hard
        if not isinstance(intervening, str):
            grad_V = clf.clf_value_gradient(x_s, safety.goal)
            f0 = clf.sys_dynm.f(x_s, [0, 0, 0])
            G0 = clf.sys_dynm.G(x_s, [0, 0, 0])
            Vdot = grad_V @ (f0 + G0 @ u_act)
            assert Vdot <= -clf.gamma * V + delta + 1e-7, \
                f"CLF violated at step {kk}: Vdot={Vdot:.4f}"
            for j in range(len(h_hmax)):
                hdot = grad_h[j] @ (h_f[j] + h_G[j] @ u_act)
                assert hdot <= -clf.alpha * h_hmax[j] + 1e-7, \
                    f"CBF row {j} violated at step {kk}: hdot={hdot:.4f}"

        h_now_log.append(cert.h_function_batch(x_s)
                         if hasattr(cert, "h_function_batch")
                         else cert.h_function(x_s))
        h_hmax_log.append(h_hmax)
        V_log.append(V)
        delta_log.append(delta if not isinstance(intervening, str) else np.nan)

        if bypass == True:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_nom, d=d_env, method="manual_RK4")
        else:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_act, d=d_env, method="manual_RK4")
        trajectory_actual.append(x_s)
        applied_u.append(u_act)

        status = (
            "INTERVENE" if intervening is True
            else intervening if isinstance(intervening, str)
            else "NOMINAL"
        )

        h_str = ", ".join(f"{h:7.4f}" for h in h_hmax)
        print(
            f"\n[{kk:2d}] Step Summary\n"
            f"  State          : x = [{x_s[0]:7.4f}, {x_s[1]:7.4f}, {x_s[2]:7.4f}]\n"
            f"  Nominal control: u_nom = [{u_nom[0]:6.3f}, {u_nom[1]:6.3f}]\n"
            f"  Actual control : u_act = [{u_act[0]:6.3f}, {u_act[1]:6.3f}]\n"
            f"  Barrier values : h = [{h_str}]\n"
            f"  CLF value      : V = {V:8.4f}   slack = {delta:8.4f}\n"
            f"  Solve time     : {dt+cert_dt*1000:.2f} ms\n"
            f"  Status      : {status}")

        print(f"Distance to goal: {np.linalg.norm(safety.goal - x_s[:2]):.4f}")
        if np.linalg.norm(safety.goal - x_s[:2]) < 0.1:
            print("Goal reached!")
            break

    safety.visual_sim.plot_external_trajectories(states_list=np.array(trajectory_actual),
                                                 inputs_list=np.array(applied_u))

    plot_h_history(np.array(h_now_log), np.array(h_hmax_log), safety.dt,
                   path=os.path.join("Results", "clf_cbf_h_history.png"))

    # V / slack history alongside the h plots
    t = np.arange(len(V_log)) * safety.dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, V_log)
    ax1.set_yscale("log"); ax1.set_ylabel("V (log)"); ax1.grid(alpha=0.3)
    ax1.set_title("CLF value")
    ax2.plot(t, delta_log, color="tab:red")
    ax2.set_ylabel("slack"); ax2.set_xlabel("time [s]"); ax2.grid(alpha=0.3)
    ax2.set_title("CLF relaxation (spikes = safety overriding stability)")
    fig.tight_layout()
    fig.savefig(os.path.join("Results", "clf_cbf_V_history.png"), dpi=150)
    print("saved clf_cbf_V_history.png")

def main_P_CLF_QP():
    x_s = np.array([0.5, 2.5, 0.0])
    safety = policy_filter(controller="constant_policy",
                        obstacles="multi",       # obstacles unused; keeps init happy
                        noise_choice="Zero")
    
    clf = v_certificate(dynamic_class=safety.dyn,
                        policy_name="proportional_policy",   # rollout policy = goal-seeker, ALWAYS
                        alpha=safety.alpha,
                        goal=safety.goal,
                        v_max=safety.v_max, om_max=safety.om_max,
                        slack=True)

    d_nom = np.zeros(3)
    
    robust_row = False  # False: nominal f,G in the QP row (isolate the rollout-V effect)
                        # True : worst-case first-step f,G (full RP-CLF)
    bypass = False
    trajectory_actual = [x_s]
    applied_u = []
    V_log = []
    delta_log = []
    for kk in range(safety.n_steps_sim):
        u_nom = safety.u_nominal(x_s)

        v_vmax, vH_dstb, grad_v, v_f, v_G, info = clf.get_value_and_grad(
            x_s, safety.noise_selection(), include_v0=True)

        if robust_row:
            f_row, g_row = v_f[0], v_G[0]
        else:
            f_row = clf.sys_dynm.f(x_s, d_nom)
            g_row = clf.sys_dynm.G(x_s, d_nom)

        u_act, intervening, dt, V, delta = clf.solve_pclf_qp(
            u_nom=u_nom, V=v_vmax[0], grad_V=grad_v[0],
            f_x=f_row, g_x=g_row)

        d_env = safety.test_noise.zero_test(safety.rng, x=x_s, k=kk, scale=safety.d_scale)

        V_log.append(V)
        delta_log.append(delta if not isinstance(intervening, str) else np.nan)

        if bypass == True:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_nom, d=d_env, method="manual_RK4")
        else:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_act, d=d_env, method="manual_RK4")
        trajectory_actual.append(x_s)
        applied_u.append(u_act)
        status = (
            "INTERVENE" if intervening is True
            else intervening if isinstance(intervening, str)
            else "NOMINAL"
        )

        print(
            f"\n[{kk:2d}] Step Summary\n"
            f"  State          : x = [{x_s[0]:7.4f}, {x_s[1]:7.4f}, {x_s[2]:7.4f}]\n"
            f"  Nominal control: u_nom = [{u_nom[0]:6.3f}, {u_nom[1]:6.3f}]\n"
            f"  Actual control : u_act = [{u_act[0]:6.3f}, {u_act[1]:6.3f}]\n"
            f"  P-CLF value    : V = {V:8.4f}   slack = {delta:8.4f}\n"
            f"  Solve time     : {dt*1000:.2f} ms\n"
            f"  Status      : {status}")

        print(f"Distance to goal: {np.linalg.norm(safety.goal - x_s[:2]):.4f}")
        if np.linalg.norm(safety.goal - x_s[:2]) < 0.1:
            print("Goal reached!")
            break

    safety.visual_sim.plot_external_trajectories(states_list=np.array(trajectory_actual),
                                                inputs_list=np.array(applied_u))

    t = np.arange(len(V_log)) * safety.dt
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t, V_log, label="V_pclf(x)")
    ax1.set_yscale("log")
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("V (log)")
    ax2 = ax1.twinx()
    ax2.plot(t, delta_log, color="tab:red", alpha=0.6, label="slack")
    ax2.set_ylabel("slack")
    fig.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(os.path.join("Results", "pclf_history.png"), dpi=150)
    print("saved pclf_history.png")

if __name__ == "__main__":
    option = 4
    if option == 1:
        main_RP_CBF_QP()
    elif option == 2:
        main_clf()
    elif option == 3:
        main_clf_cbf()
    elif option == 4:
        main_P_CLF_QP()