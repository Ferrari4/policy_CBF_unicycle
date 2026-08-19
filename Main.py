import os
import time
import numpy as np
import quadprog
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

from Dynamics import sys_dynm_dd
from Control_policy import policy
from Value_function import h_certificate
from Sim_unicycle import UnicycleSim
from Noise_sampler import noise_train_sampler, noise_test_sampler

class policy_filter:
    def __init__(self, controller="proportional_policy", obstacles="single"):
        self.controller = controller

        # Simulation parameters
        self.T_rollout   = 1.5      # s, certificate lookahead
        self.T_dstb_hold = 0.3      # s, piecewise-constant disturbance interval
        self.T_sim       = 100.0    # s, sim length
        self.dt          = 0.05     # s, sim step size

        # General parameters
        self.barrier_inflate = 0.03 # margin for safety (to avoid numerical issues)
        self.v_max = 1.0            # m/s, max linear velocity
        self.om_max = 3.0           # rad/s, max angular velocity
        self.goal = np.array([4.0, 2.0]) # goal position in the plane in meters [x,y]

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

    def safety_rpcbf(self, x, u_nom):
        start_time = time.perf_counter()
        BH_dstb_train, _ = self.train_noise.bangbang_uniform_train(
            n_samples=self.n_samples,
            n_samples_uniform=self.n_samples_uniform,
            horizon=self.horizon,
            interval_size=self.interval_size,
            scale=self.d_scale)

        h_hmax, hH_dstb, grad_h_hmax, h_f, h_G, info = self.cert.get_value_and_grad(
            x, BH_dstb_train, include_h0=False)

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
            h_hmax[s] = self.cert.compute_h_hmax_from_dstb(x0[s], bH_dstb, include_h0=include_h0, max_type="cubic_spline")
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

def main():
    x_s = np.array([0.5, 2.5, 0.0])
    safety = policy_filter(controller="proportional_policy", obstacles="multi")
    bypass = False  # IMPORTANT: set to FALSE for CBF safety intervention, TRUE to bypass and use nominal control only
    trajectory_actual = [x_s]
    applied_u = []
    for kk in range(safety.n_steps_sim):
        u_nom = safety.u_nominal(x_s)
        u_act, intervening, dt, h_hmax = safety.safety_rpcbf(x=x_s, u_nom=u_nom)
        d_env = safety.test_noise.uniform_test(safety.rng, x=x_s, k=kk, scale=safety.d_scale)
        if bypass == True:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_nom, d=d_env, method="manual_RK4")
        else:
            x_s = safety.dyn.one_step_u(x0=x_s, u=u_act, d=d_env, method="manual_RK4")
        trajectory_actual.append(x_s)
        applied_u.append(u_act)

        h_str = ", ".join(f"{h:7.4f}" for h in h_hmax)
        print(
            f"\n[{kk:2d}] Step Summary\n"
            f"  State          : x = [{x_s[0]:7.4f}, {x_s[1]:7.4f}, {x_s[2]:7.4f}]\n"
            f"  Nominal control: u_nom = [{u_nom[0]:6.3f}, {u_nom[1]:6.3f}]\n"
            f"  Actual control : u_act = [{u_act[0]:6.3f}, {u_act[1]:6.3f}]\n"
            f"  Barrier values : h = [{h_str}]\n"
            f"  Status         : {'INTERVENE' if intervening is True else intervening if isinstance(intervening, str) else 'NOMINAL'}")

        print(f"Distance to goal: {np.linalg.norm(safety.goal - x_s[:2]):.4f}")
        if np.linalg.norm(safety.goal - x_s[:2]) < 0.1:
            print("Goal reached!")
            break

    safety.visual_sim.plot_external_trajectories(states_list=np.array(trajectory_actual), inputs_list=np.array(applied_u))
    out = safety.evaluate_mesh(theta_fixed=0.0,include_h0=False)
    plot_mesh(out, traj=np.array(trajectory_actual),
              path=os.path.join("Results", "rpcbf_check.png"))
    print("saved rpcbf_check.png")

if __name__ == "__main__":
    main()