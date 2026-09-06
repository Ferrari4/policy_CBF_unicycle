import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_h_history(h_now, h_hmax, dt, path=None):
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

def plot_v_history(V_log, delta_log, dt, path, value_label="V(x)", title="CLF value"):
    t = np.arange(len(V_log)) * dt

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(t, V_log, label=value_label)
    ax1.set_yscale("log")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("V (log)")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(t, delta_log, alpha=0.6, label="slack")
    ax2.set_ylabel("slack")

    fig.legend(loc="upper right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def as_traj_list(data, name, dim):
    arr = np.asarray(data)

    if arr.ndim == 2:
        if arr.shape[1] != dim:
            raise ValueError(
                f"{name} must have shape (N, {dim}), got {arr.shape}"
            )
        return [arr]

    if arr.ndim == 3:
        if arr.shape[2] != dim:
            raise ValueError(
                f"{name} must have shape (B, N, {dim}), got {arr.shape}"
            )
        return [arr[i] for i in range(arr.shape[0])]

    raise ValueError(
        f"{name} must be 2D or 3D, got shape {arr.shape}"
    )

def plot_vdot_history(Vdot_log, V_log, delta_log, dt, gamma, path,
                      Vdot_actual=None, value_label="V", title="CLF decrease condition"):
    """
    Three-band view of the CLF decrease condition along one run.

        Vdot_log    (N,)  analytic  grad_V @ (f + G u_act)      -- what the QP saw
        V_log       (N,)  V(x_k)
        delta_log   (N,)  QP slack (nan where infeasible)
        Vdot_actual (N,)  optional, e.g. np.diff(V_log)/dt      -- what actually happened
        gamma             CLF rate used in the constraint

    Bands:  Vdot <= -gamma V        exponential decrease (slack == 0)
            -gamma V < Vdot < 0     decreasing, slower than gamma (slack active)
            Vdot >= 0               not decreasing (CLF condition failed here)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    Vdot  = np.asarray(Vdot_log, dtype=float)
    V     = np.asarray(V_log, dtype=float)
    delta = np.asarray(delta_log, dtype=float)
    N     = min(Vdot.shape[0], V.shape[0], delta.shape[0])
    Vdot, V, delta = Vdot[:N], V[:N], delta[:N]
    t     = np.arange(N) * dt
    req   = -gamma * V                                   # what the constraint demanded

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # slack-active intervals
    active = np.nan_to_num(delta, nan=0.0) > 1e-9
    if active.any():
        edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t[s], t[min(e, N - 1)], color="orange", alpha=0.15, lw=0,
                       label="slack active" if s == np.where(edges == 1)[0][0] else None)

    ax.axhline(0.0, color="k", ls="--", lw=1.2, label=r"$\dot V = 0$")
    ax.plot(t, req,  color="tab:red",  lw=1.4, ls=":", label=fr"$-\gamma {value_label}$  ($\gamma$={gamma})")
    ax.plot(t, Vdot, color="tab:blue", lw=1.6, label=fr"$\dot {value_label}$ (QP, analytic)")
    if Vdot_actual is not None:
        Va = np.asarray(Vdot_actual, dtype=float)[:N]
        ax.plot(t[:Va.shape[0]], Va, color="tab:green", lw=1.0, alpha=0.8,
                label=fr"$\dot {value_label}$ (actual, finite diff)")

    # infeasible steps, if any
    infeas = np.isnan(delta)
    if infeas.any():
        ax.plot(t[infeas], np.zeros(infeas.sum()), "x", color="red", ms=6, label="QP infeasible")

    ax.set_xlabel("time [s]")
    ax.set_ylabel(fr"$\dot {value_label}$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # summary: how much of the run sat in each band
    exp_band  = Vdot <= req + 1e-7
    slow_band = (~exp_band) & (Vdot < 0.0)
    fail_band = Vdot >= 0.0
    print(f"[{title}]  exponential {exp_band.mean():6.1%} | "
          f"decreasing-but-slow {slow_band.mean():6.1%} | "
          f"not decreasing {fail_band.mean():6.1%} | "
          f"infeasible steps {infeas.sum()}")

def robot_triangle(pose, size=0.15):
    x, y, theta = pose

    local = np.array([
        [ size, 0.0],
        [-size * 0.6,  size * 0.5],
        [-size * 0.6, -size * 0.5],
    ])

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    return local @ R.T + np.array([x, y])

def plot_trajectories(states_list, inputs_list, goal, obstacles=None, labels=None, dt=0.1, robot_size=0.15, 
                      triangle_every=None, title="Unicycle trajectories", show=False, results_dir="Results"):

        obstacles = [] if obstacles is None else obstacles
        states_list = as_traj_list(states_list, "states_list", 3)
        inputs_list = as_traj_list(inputs_list, "inputs_list", 2)

        if len(states_list) != len(inputs_list):
            raise ValueError(
                f"Got {len(states_list)} state trajectories but "
                f"{len(inputs_list)} input trajectories.")

        n_traj = len(states_list)

        if labels is None:
            labels = [f"traj {i}" for i in range(n_traj)]
        elif isinstance(labels, str):
            labels = [labels]
        if len(labels) != n_traj:
            raise ValueError("labels must have the same length as states_list.")

        dt = float(dt)

        # ---------------- figure 1: XY trajectories ----------------
        fig_traj, ax_traj = plt.subplots(figsize=(8, 8))

        for (cx, cy, r_obs) in obstacles:
            circ = Circle((cx, cy), r_obs, facecolor="lightcoral",
                              edgecolor="darkred", alpha=0.5, zorder=1)
            ax_traj.add_patch(circ)
            ax_traj.plot(cx, cy, "x", color="darkred", markersize=8, zorder=2)
        if obstacles:
            ax_traj.plot([], [], "s", color="lightcoral",
                         markeredgecolor="darkred", label="obstacle")
            ax_traj.plot([], [], "x", color="darkred", label="obstacle center")

        colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))

        # ---------------- figure 2: control inputs ----------------
        fig_u, (ax_v, ax_omega) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        for i, (states, inputs, label) in enumerate(
                zip(states_list, inputs_list, labels)):

            if states.ndim != 2 or states.shape[1] != 3:
                raise ValueError(f"states_list[{i}] must have shape (N+1, 3), "
                                 f"got {states.shape}")
            if inputs.ndim != 2 or inputs.shape[1] != 2:
                raise ValueError(f"inputs_list[{i}] must have shape (N, 2), "
                                 f"got {inputs.shape}")

            # accept N inputs (strict ZOH) or N+1 (input logged at every state)
            if states.shape[0] == inputs.shape[0]:
                print(f"[plot_external_trajectories] traj {i}: inputs has same "
                      f"length as states; dropping last input (assumed unused).")
                inputs = inputs[:-1]
            elif states.shape[0] != inputs.shape[0] + 1:
                raise ValueError(
                    f"Trajectory {i}: expected len(states) == len(inputs)+1, "
                    f"got {states.shape[0]} states and {inputs.shape[0]} inputs.")

            c = colors[i % 10]

            # XY path
            ax_traj.plot(states[:, 0], states[:, 1], "-", color=c,
                         linewidth=1.5, label=label, zorder=3)
            ax_traj.plot(states[0, 0], states[0, 1], "o", color="green",
                         markersize=9, markeredgecolor="black", zorder=5)
            ax_traj.plot(goal[0], goal[1], "*", color="gold",
                         markersize=15, markeredgecolor="black", zorder=5)

            # robot triangles
            poses = [states[0], states[-1]]
            if triangle_every is not None:
                poses = list(states[::triangle_every])
                if not np.array_equal(poses[-1], states[-1]):
                    poses.append(states[-1])
            for pose in poses:
                tri = robot_triangle(pose, size=robot_size)
                ax_traj.fill(tri[:, 0], tri[:, 1], color=c, alpha=0.6,
                             edgecolor="black", linewidth=0.8, zorder=4)

            # input signals: inputs[k] is held over [t_k, t_{k+1}) -> ZOH steps
            t_u = np.arange(inputs.shape[0] + 1) * dt
            v_sig = np.append(inputs[:, 0], inputs[-1, 0])
            w_sig = np.append(inputs[:, 1], inputs[-1, 1])
            ax_v.plot(t_u, v_sig, linewidth=1.5, color=c, label=label,
                      drawstyle="steps-post")
            ax_omega.plot(t_u, w_sig, linewidth=1.5, color=c, label=label,
                          drawstyle="steps-post")

        # ---------------- formatting ----------------
        ax_traj.plot([], [], "o", color="green", markeredgecolor="black",
                     label="start")
        ax_traj.plot([], [], "*", color="gold", markeredgecolor="black",
                     label="end")
        ax_traj.set_xlabel("x [m]")
        ax_traj.set_ylabel("y [m]")
        ax_traj.set_title(title)
        ax_traj.set_aspect("equal", adjustable="datalim")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.legend(loc="best", fontsize=9)

        ax_v.set_ylabel("v [m/s]")
        ax_v.set_title("Control inputs")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc="best", fontsize=9)
        ax_omega.set_xlabel("Time [s]")
        ax_omega.set_ylabel(r"$\omega$ [rad/s]")
        ax_omega.grid(True, alpha=0.3)

        fig_traj.tight_layout()
        fig_u.tight_layout()

        # ---------------- save figures ----------------
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
        os.makedirs(results_dir, exist_ok=True)

        fig_traj.savefig(
            os.path.join(results_dir, "trajectory_plot.png"),
            dpi=300,
            bbox_inches="tight"
        )

        fig_u.savefig(
            os.path.join(results_dir, "inputs_plot.png"),
            dpi=300,
            bbox_inches="tight"
        )

        if show:
            plt.show()

        return ax_traj, (ax_v, ax_omega)
