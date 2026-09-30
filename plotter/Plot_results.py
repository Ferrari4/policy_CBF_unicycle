import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_h_history(h_now, h_hmax, dt, path=None):
    if path is None:
        path = os.path.join("plotter/Results", "h_history.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    h_now = np.asarray(h_now)*(-1)
    h_hmax = np.asarray(h_hmax)*(-1)
    t = np.arange(h_now.shape[0]) * dt
    nh = h_now.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, (V, title) in zip(axes, [(h_now,  "h(x_t)  -  current state"),
                                     (h_hmax, "h_hmax  -  worst-case lookahead")]):
        for j in range(nh):
            ax.plot(t[:V.shape[0]], V[:, j], lw=1.6, label=f"obs {j}")
        ax.axhline(0.0, color="k", ls="--", lw=1.2)
        ax.axhspan(min(-1e-3, np.nanmin(V)), 0.0, color="red", alpha=0.06)
        ax.set_ylabel("h")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    viol = np.where(h_now.max(axis=1) < 0.0)[0]
    if viol.size:
        print(f"SAFETY VIOLATION: h < 0 at {viol.size} steps, "
              f"first t = {viol[0] * dt:.2f}s, max h = {h_now.max():.4f}")
    else:
        print(f"No violation. Closest approach: h_max = {h_now.max():.4f} "
              f"(margin {-h_now.max():.4f})")
        
def plot_v_history(V_log, V_max_log, dt, path=None,
                   value_label="V(x_t)", title="instantaneous P-CLF value",
                   max_label="V_max",
                   max_title=r"cumulative cost, worst case  $\max_d \int_0^T V(x_t)\,dt$  (gradient source)"):
    """
    Two panels, same layout as plot_h_history (h_now / h_hmax):

        V_log      (N,)  V(x_t)   instantaneous certificate at the current state
                                  (clf_certificate(x_t))                    -- like h_now
        V_max_log  (N,)  V_max    worst-case horizon integral of V along the rollout
                                  (v_vmax from get_value_and_grad); this is the
                                  quantity the QP gradient grad_V is computed from  -- like h_hmax

    Both are >= 0 by construction, so each panel gets a dashed line at 0 and the
    y-axis is pinned so that line is always visible.

    Two files are written:
        <path>          linear y-axis (as before)
        <path>_log.png  log y-axis, so the goal-approach phase (V << 1) is readable.
                        The 0 line cannot exist on a log axis, so it is replaced by a
                        dotted line at the smallest positive value reached.
    """
    if path is None:
        path = os.path.join("plotter/Results", "v_history.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    V_log     = np.asarray(V_log, dtype=float)
    V_max_log = np.asarray(V_max_log, dtype=float)
    N = min(V_log.shape[0], V_max_log.shape[0])
    V_log, V_max_log = V_log[:N], V_max_log[:N]
    t = np.arange(N) * dt

    def _pin_zero(ax, y):
        lo = min(0.0, np.nanmin(y)) if N else 0.0
        hi = np.nanmax(y) if N else 1.0
        pad = 0.05 * max(hi - lo, 1e-12)
        ax.set_ylim(lo - pad, hi + pad)             # keep the 0 line on screen

    def _positive_floor(y):
        pos = y[np.isfinite(y) & (y > 0)]
        return pos.min() if pos.size else 1e-12

    def _draw(log_scale):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax_V, ax_I = axes
        scale_tag = "  [log scale]" if log_scale else ""

        # ---- panel 1: V(x_t), instantaneous ----
        ax_V.plot(t, V_log, lw=1.6, color="tab:blue", label="V")
        ax_V.set_ylabel("V")
        ax_V.set_title(f"{value_label}  -  {title}{scale_tag}")

        # ---- panel 2: V_max, cumulative cost over the horizon (what the gradient is taken of) ----
        ax_I.plot(t, V_max_log, lw=1.6, color="tab:purple", label=max_label)
        ax_I.set_ylabel(max_label)
        ax_I.set_title(f"{max_label}  -  {max_title}{scale_tag}")

        for ax, y, lab in ((ax_V, V_log, "V"), (ax_I, V_max_log, max_label)):
            if log_scale:
                ax.set_yscale("log")
                floor = _positive_floor(y)
                ax.axhline(floor, color="k", ls=":", lw=1.0,
                           label=f"min {lab} = {floor:.2e}")
            else:
                ax.axhline(0.0, color="k", ls="--", lw=1.2,
                           label=f"{lab} = 0  (lower bound)" if ax is ax_I else None)
                _pin_zero(ax, y)
            ax.grid(alpha=0.3, which="both" if log_scale else "major")
            ax.legend(fontsize=8, loc="upper right")

        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()
        return fig

    fig = _draw(log_scale=False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    root, ext = os.path.splitext(path)
    fig = _draw(log_scale=True)
    fig.savefig(f"{root}_log{ext or '.png'}", dpi=300, bbox_inches="tight")
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
                      Vdot_actual=None, value_label="V", title="CLF decrease condition",
                      req_log=None, ell0_log=None, ellT_log=None):
    """
    Three-band view of the CLF decrease condition along one run.

        Vdot_log    (N,)  analytic  grad_V @ (f + G u_act)      -- what the QP saw
        V_log       (N,)  V(x_k)
        delta_log   (N,)  QP slack (nan where infeasible)
        Vdot_actual (N,)  optional, e.g. np.diff(V_log)/dt      -- what actually happened
        gamma             rate of the hand-drawn constraint      (-gamma V)
        req_log     (N,)  optional, the bound the QP actually enforced, i.e. -rate
                          (alV_log from the sim).  When given it replaces -gamma V as the
                          reference in the band statistics.
        ell0_log    (N,)  optional, l(x_0) per P-CLF solve  (safety.ell0_log)
        ellT_log    (N,)  optional, l(x_T) per P-CLF solve  (safety.ellT_log)
                          When both are given the two candidate P-CLF bounds are drawn:
                              tail dropped :  -l(x_0)
                              tail kept    :  -(l(x_0) - l(x_T))   (positive when l(x_T) > l(x_0))

    Bands:  Vdot <= req             constraint met (slack == 0)
            req  < Vdot < 0         decreasing, slower than required (slack active)
            Vdot >= 0               not decreasing (CLF condition failed here)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    Vdot  = np.asarray(Vdot_log, dtype=float)
    V     = np.asarray(V_log, dtype=float)
    delta = np.asarray(delta_log, dtype=float)
    N     = min(Vdot.shape[0], V.shape[0], delta.shape[0])

    have_req  = req_log is not None
    have_tail = ell0_log is not None and ellT_log is not None
    if have_req:
        req_enf = np.asarray(req_log, dtype=float)
        N = min(N, req_enf.shape[0])
    if have_tail:
        ell0 = np.asarray(ell0_log, dtype=float)
        ellT = np.asarray(ellT_log, dtype=float)
        N = min(N, ell0.shape[0], ellT.shape[0])

    Vdot, V, delta = Vdot[:N], V[:N], delta[:N]
    t         = np.arange(N) * dt
    req_gamma = -gamma * V                               # hand-drawn exponential bound
    if have_req:
        req_enf = req_enf[:N]
    if have_tail:
        ell0, ellT   = ell0[:N], ellT[:N]
        req_notail   = -ell0                             # tail dropped
        req_tail     = -(ell0 - ellT)                    # tail kept

    # the bound the statistics are measured against
    if have_req:
        req, req_name = req_enf, "-rate (enforced by QP)"
    else:
        req, req_name = req_gamma, fr"$-\gamma {value_label}$  ($\gamma$={gamma})"

    fig, (ax, ax_s) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.0, 1.0]})

    # slack-active intervals (shaded on both panels so they line up)
    active = np.nan_to_num(delta, nan=0.0) > 1e-9
    if active.any():
        edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
        first = np.where(edges == 1)[0][0]
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            for a in (ax, ax_s):
                a.axvspan(t[s], t[min(e, N - 1)], color="orange", alpha=0.15, lw=0,
                          label="slack active" if (s == first and a is ax) else None)

    ax.axhline(0.0, color="k", ls="--", lw=1.2, label=r"$\dot V = 0$")

    # candidate bounds
    if have_tail:
        ax.plot(t, req_notail, color="tab:purple", lw=1.2, ls="-.",
                label=r"$-l(x_0)$  (tail dropped)")
        ax.plot(t, req_tail, color="tab:orange", lw=1.2, ls="-.",
                label=r"$-(l(x_0)-l(x_T))$  (tail kept)")
    ax.plot(t, req_gamma, color="gray" if (have_req or have_tail) else "tab:red",
            lw=1.0 if (have_req or have_tail) else 1.4, ls=":",
            alpha=0.6 if (have_req or have_tail) else 1.0,
            label=fr"$-\gamma {value_label}$  ($\gamma$={gamma})")
    if have_req:
        ax.plot(t, req_enf, color="tab:red", lw=1.8, ls=":", label=req_name)

    ax.plot(t, Vdot, color="tab:blue", lw=1.6, label=fr"$\dot {value_label}$ (QP, analytic)")
    if Vdot_actual is not None:
        Va = np.asarray(Vdot_actual, dtype=float)[:N]
        ax.plot(t[:Va.shape[0]], Va, color="tab:green", lw=1.0, alpha=0.8,
                label=fr"$\dot {value_label}$ (actual, finite diff)")

    # infeasible steps, if any
    infeas = np.isnan(delta)
    if infeas.any():
        ax.plot(t[infeas], np.zeros(infeas.sum()), "x", color="red", ms=6, label="QP infeasible")

    ax.set_ylabel(fr"$\dot {value_label}$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    # ---- panel 2: slack (CLF relaxation) ----
    ax_s.plot(t, delta, lw=1.6, color="tab:blue", label="slack")
    ax_s.axhline(0.0, color="k", ls="--", lw=1.2)
    if infeas.any():
        ax_s.plot(t[infeas], np.zeros(infeas.sum()), "x", color="red", ms=6, label="QP infeasible")
    ax_s.set_xlabel("time [s]")
    ax_s.set_ylabel("slack")
    ax_s.set_title("slack  -  CLF relaxation")
    ax_s.grid(alpha=0.3)
    ax_s.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # summary: how much of the run sat in each band (nan rows, i.e. infeasible, are skipped)
    ok        = ~np.isnan(Vdot) & ~np.isnan(req)
    n_ok      = max(ok.sum(), 1)
    met_band  = ok & (Vdot <= req + 1e-7)
    slow_band = ok & ~met_band & (Vdot < 0.0)
    fail_band = ok & (Vdot >= 0.0)
    band_name = "rate met" if have_req else "exponential"
    print(f"[{title}]  {band_name} {met_band.sum()/n_ok:6.1%} | "
          f"decreasing-but-slow {slow_band.sum()/n_ok:6.1%} | "
          f"not decreasing {fail_band.sum()/n_ok:6.1%} | "
          f"infeasible steps {infeas.sum()}")

    # which bounds would have been satisfied by the Vdot the QP delivered
    if have_req or have_tail:
        pct = lambda b: (ok & (Vdot <= b + 1e-7)).sum() / n_ok
        line = f"[{title}]  satisfied:  -gamma V {pct(req_gamma):6.1%}"
        if have_tail:
            tail_valid = ok & (ellT < ell0)
            line += (f" | -l(x0) {pct(req_notail):6.1%}"
                     f" | -(l(x0)-l(xT)) {pct(req_tail):6.1%}"
                     f"   [tail valid (l(xT)<l(x0)) on {tail_valid.sum()/n_ok:6.1%} of steps]")
        print(line)

def plot_hdot_history(hdot_log, h_hmax_log, dt, alpha, path,
                      hdot_actual=None, title="CBF condition  (h <= 0 safe)"):
    """
    CBF counterpart of plot_vdot_history. Two panels per obstacle.

        hdot_log    (N, nh)  analytic grad_h (f + G u_act) + dV_dt   -- what the QP saw
        h_hmax_log  (N, nh)  worst-case lookahead h (same certificate the gradient belongs to)
        hdot_actual (N, nh)  optional, e.g. np.diff(h_hmax_log, axis=0)/dt  -- what actually happened
        alpha                CBF rate used in the constraint

    Constraint:  hdot <= -alpha h.      margin = -alpha h - hdot
        margin > 0   constraint inactive (nominal input feasible)
        margin = 0   constraint active   (filter is shaping the input)
        margin < 0   violated            (should never happen; assert in Main catches it)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    hdot = np.asarray(hdot_log, dtype=float)
    h    = np.asarray(h_hmax_log, dtype=float)
    if hdot.ndim == 1: hdot = hdot[:, None]
    if h.ndim == 1:    h    = h[:, None]
    N    = min(hdot.shape[0], h.shape[0])
    hdot, h = hdot[:N], h[:N]
    nh   = hdot.shape[1]
    t    = np.arange(N) * dt
    req  = -alpha * h                                    # what the constraint demanded
    margin = req - hdot

    fig, axes = plt.subplots(2 * nh, 1, figsize=(10, 3.2 * 2 * nh), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for j in range(nh):
        # ---- panel 1: hdot vs required rate ----
        ax = axes[2 * j]
        ax.axhline(0.0, color="k", ls="--", lw=1.0, label=r"$\dot h = 0$")
        ax.plot(t, req[:, j],  color="tab:red",  ls=":", lw=1.4,
                label=fr"$-\alpha h$  ($\alpha$={alpha})")
        ax.plot(t, hdot[:, j], color="tab:blue", lw=1.6, label=r"$\dot h$ (QP, analytic)")
        if hdot_actual is not None:
            ha = np.asarray(hdot_actual, dtype=float)
            if ha.ndim == 1: ha = ha[:, None]
            ha = ha[:N, j]
            ax.plot(t[:ha.shape[0]], ha, color="tab:green", lw=1.0, alpha=0.8,
                    label=r"$\dot h$ (actual, finite diff)")
        infeas = np.isnan(hdot[:, j])
        if infeas.any():
            ax.plot(t[infeas], np.zeros(infeas.sum()), "x", color="red", ms=6, label="QP infeasible")
        ax.set_ylabel(r"$\dot h$")
        ax.set_title(f"obs {j}:  {title}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

        # ---- panel 2: margin ----
        axm = axes[2 * j + 1]
        axm.axhline(0.0, color="k", ls="--", lw=1.0)
        axm.plot(t, margin[:, j], color="tab:purple", lw=1.6, label=r"margin $= -\alpha h - \dot h$")
        axm.fill_between(t, 0.0, margin[:, j], where=margin[:, j] >= 0,
                         color="tab:purple", alpha=0.12, lw=0)
        axm.fill_between(t, 0.0, margin[:, j], where=margin[:, j] < 0,
                         color="red", alpha=0.25, lw=0, label="violated")
        active = np.nan_to_num(margin[:, j], nan=np.inf) <= 1e-6
        if active.any():
            axm.plot(t[active], np.zeros(active.sum()), ".", color="tab:orange", ms=4,
                     label="constraint active")
        axm.set_ylabel("margin")
        axm.grid(alpha=0.3)
        axm.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # summary
    active_any = (np.nan_to_num(margin, nan=np.inf) <= 1e-6).any(axis=1)
    viol       = np.nan_to_num(margin, nan=0.0) < -1e-7
    infeas_any = np.isnan(hdot).any(axis=1)
    print(f"[{title}]  constraint active {active_any.mean():6.1%} of steps | "
          f"min margin {np.nanmin(margin):+.4f} | "
          f"violations {int(viol.sum())} | "
          f"infeasible steps {int(infeas_any.sum())}")

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