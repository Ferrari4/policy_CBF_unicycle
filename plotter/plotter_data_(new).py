"""
Comparison plots for unicycle RPCBF / CLF-CBF runs.

Change from plotter_data_1: methods are no longer sheets inside one
workbook.  Each run is its own .xlsx, and the method label comes from
the "Method" row of the settings block, not the sheet name (which is
now the controller and is identical across files).

Usage
-----
    plot_results_from_excel(
        ["run_a.xlsx", "run_b.xlsx", "run_c.xlsx"],
        dt=0.005,
        goal=[4.5, 4.5],
        obstacles=[(2.0, 2.5, 0.3)],
    )
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# =============================================================
# Reading
# =============================================================

def _read_run(path, label=None):
    """Return (label, settings dict, DataFrame) for one workbook."""
    raw = pd.read_excel(path, header=None)

    # settings block: col 0 = key, col 1 = value, until the header row
    hdr = raw.index[raw[0].astype(str) == "step"]
    if len(hdr) == 0:
        raise ValueError(f"{path}: no 'step' header row found")
    hdr = int(hdr[0])

    settings = {}
    for i in range(hdr):
        key = raw.iloc[i, 0]
        if pd.notna(key) and str(key) != "Simulation Settings":
            settings[str(key)] = raw.iloc[i, 1]

    df = pd.read_excel(path, header=hdr)
    if label is None:
        label = str(settings.get("method", settings.get("Method", os.path.basename(path))))
    return label, settings, df


def _column(df, name):
    """
    Return the column as float, or None if it carries no information.

    Two ways a column is uninformative:
      * all NaN                      -- writer left it out (rpcbf: V, delta)
      * identically zero, no NaN     -- writer padded it (pure_backup:
                                        h_hmax, V, delta are literal 0)

    The second case is the dangerous one: np.isfinite(0) is True, so the
    original code plotted a flat line on the safety boundary for a method
    that has no h_max at all.  V == 0 for a whole run would mean sitting
    on the goal from t=0, which never happens.
    """
    if name not in df.columns:
        return None
    v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    if not np.any(np.isfinite(v)):
        return None
    if np.all(v == 0.0):
        return None
    return v


def _check_dt(df, dt, label):
    """Warn if the supplied dt disagrees with the integration step."""
    try:
        v0 = float(df["input_0"][0])
        th0 = float(df["state_2"][0])
        dx = float(df["state_0"][1]) - float(df["state_0"][0])
        dt_hat = dx / (v0 * np.cos(th0))
    except Exception:
        return
    if not np.isfinite(dt_hat) or dt_hat <= 0:
        return
    if abs(dt_hat - dt) / dt_hat > 0.05:
        print(
            f"  !! {label}: dt={dt} but the data implies dt={dt_hat:.5g}. "
            f"Time axes are off by {dt / dt_hat:.1f}x."
        )


# =============================================================
# Main
# =============================================================

def plot_results_from_excel(
    excel_paths,
    dt=0.005,
    goal=None,
    obstacles=None,
    show=True,
    results_dir="Results",
    order=None,
    labels=None,
):
    if isinstance(excel_paths, str):
        excel_paths = [excel_paths]
    obstacles = [] if obstacles is None else obstacles

    labels = list(labels) if labels else [None] * len(excel_paths)
    assert len(labels) == len(excel_paths), "one label per file"

    runs = {}
    for p, lab in zip(excel_paths, labels):
        label, settings, df = _read_run(p, lab)
        if label in runs:
            label = f"{label} ({os.path.basename(p)})"
        runs[label] = (settings, df)
        print(f"Loaded {os.path.basename(p)}  ->  method '{label}', "
              f"{len(df)} steps")
        _check_dt(df, dt, label)

    if order:
        methods = [m for m in order if m in runs] + \
                  [m for m in runs if m not in order]
    else:
        methods = list(runs)

    cmap = plt.get_cmap("tab10").colors
    color = {m: cmap[i % 10] for i, m in enumerate(methods)}
    style = {m: ["-", "--", "-.", ":"][i % 4] for i, m in enumerate(methods)}

    t_max = max(len(runs[m][1]) for m in methods) * dt

    # ---------------------------------------------------------
    # Figure 1: trajectories
    # ---------------------------------------------------------
    fig_traj, ax = plt.subplots(figsize=(8, 8))

    for cx, cy, r in obstacles:
        ax.add_patch(Circle((cx, cy), r, facecolor="lightcoral",
                            edgecolor="darkred", alpha=0.5, zorder=1))
        ax.plot(cx, cy, "x", color="darkred", markersize=8, zorder=2)

    # draw in reverse so the first method ends up on top rather than
    # buried under every method plotted after it
    for i, m in reversed(list(enumerate(methods))):
        df = runs[m][1]
        x = _column(df, "state_0")
        y = _column(df, "state_1")
        if x is None or y is None:
            continue
        ok = np.isfinite(x) & np.isfinite(y)
        ax.plot(x[ok], y[ok], style[m], color=color[m],
                linewidth=3.0 - 0.5 * i, alpha=0.9,
                label=m, zorder=3 + (len(methods) - i))
        ax.plot(x[ok][0], y[ok][0], "o", color="green", markersize=8,
                markeredgecolor="black", zorder=20)
        ax.plot(x[ok][-1], y[ok][-1], "s", color=color[m], markersize=6,
                markeredgecolor="black", zorder=20)

    if goal is not None:
        g = np.asarray(goal).reshape(-1)
        ax.plot(g[0], g[1], "*", color="gold", markersize=16,
                markeredgecolor="black", label="Goal", zorder=21)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory comparison")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ordered = sorted(range(len(labels)),
                     key=lambda j: methods.index(labels[j])
                     if labels[j] in methods else 99)
    ax.legend([handles[j] for j in ordered], [labels[j] for j in ordered],
              loc="best", fontsize=9)
    fig_traj.tight_layout()

    # ---------------------------------------------------------
    # Figure 2: inputs -- all runs overlaid, one axis per input
    # ---------------------------------------------------------
    fig_u, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j, (col, ylab, ttl) in enumerate([("input_0", r"$v$ [m/s]", "Linear velocity"),
                                          ("input_1", r"$\omega$ [rad/s]", "Angular velocity")]):
        a = axes[j]
        for i, m in reversed(list(enumerate(methods))):
            u = _column(runs[m][1], col)
            if u is None:
                continue
            t = np.arange(len(u)) * dt
            ok = np.isfinite(u)
            a.plot(t[ok], u[ok], style[m], color=color[m], linewidth=2.0 - 0.3 * i,
                   alpha=0.9, drawstyle="steps-post", label=m if j == 0 else None)
        a.set_ylabel(ylab)
        a.set_title(ttl, fontsize=10, loc="left")
        a.grid(True, alpha=0.3)
        a.set_xlim(0, t_max)
    axes[0].legend(fontsize=9, loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig_u.suptitle("Control input comparison", fontsize=14)
    fig_u.tight_layout()

    # ---------------------------------------------------------
    # Figure 3: barrier -- only methods that actually have one
    # ---------------------------------------------------------
    def keys(df, prefix):
        return sorted(k for k in df.columns
                      if str(k).startswith(prefix)
                      and _column(df, k) is not None)

    h_methods = [m for m in methods
                 if keys(runs[m][1], "h_now_") or keys(runs[m][1], "h_hmax_")]

    fig_h = None
    if h_methods:
        fig_h, axes = plt.subplots(len(h_methods), 2,
                                   figsize=(12, 2.6 * len(h_methods)),
                                   squeeze=False, sharex=True)
        for i, m in enumerate(h_methods):
            df = runs[m][1]
            for j, prefix in enumerate(["h_now_", "h_hmax_"]):
                a = axes[i, j]
                ks = keys(df, prefix)
                for k in ks:
                    h = _column(df, k)
                    t = np.arange(len(h)) * dt
                    ok = np.isfinite(h)
                    a.plot(t[ok], h[ok], linewidth=1.4,
                           label=k.replace(prefix, "h "))
                if not ks:
                    a.text(0.5, 0.5, "not computed by this method",
                           ha="center", va="center", fontsize=9,
                           color="0.5", transform=a.transAxes)
                a.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
                a.grid(True, alpha=0.3)
                a.set_xlim(0, t_max)
                if len(ks) > 1:
                    a.legend(fontsize=7, loc="best")
            axes[i, 0].set_ylabel(f"{m}\nh")
        axes[0, 0].set_title(r"$h(x_t)$")
        axes[0, 1].set_title(r"$h_{hmax}$")
        axes[-1, 0].set_xlabel("Time [s]")
        axes[-1, 1].set_xlabel("Time [s]")
        fig_h.suptitle("Safety certificate comparison", fontsize=14)
        fig_h.tight_layout()

    # ---------------------------------------------------------
    # Figure 4: CLF value and slack (was missing entirely)
    # ---------------------------------------------------------
    v_methods = [m for m in methods
                 if _column(runs[m][1], "V_0") is not None
                 or _column(runs[m][1], "delta_0") is not None]

    fig_v = None
    if v_methods:
        fig_v, axes = plt.subplots(len(v_methods), 2,
                                   figsize=(12, 2.6 * len(v_methods)),
                                   squeeze=False, sharex=True)
        for i, m in enumerate(v_methods):
            df = runs[m][1]
            aV, aD = axes[i, 0], axes[i, 1]

            V = _column(df, "V_0")
            if V is not None:
                t = np.arange(len(V)) * dt
                ok = np.isfinite(V)
                aV.plot(t[ok], V[ok], color=color[m], linewidth=1.4)
                aV.set_yscale("symlog", linthresh=1e-3)
            aV.set_ylabel(f"{m}\n$V$")
            aV.grid(True, alpha=0.3)

            D = _column(df, "delta_0")
            if D is not None:
                t = np.arange(len(D)) * dt
                ok = np.isfinite(D)
                aD.plot(t[ok], D[ok], color=color[m], linewidth=1.4)
                act = ok & (D > 1e-9)
                if np.any(act):
                    # shade the window where the CLF is being relaxed
                    aD.fill_between(t, 0, D, where=act, alpha=0.3,
                                    color=color[m], step=None)
                    aD.set_title(
                        f"slack active {100 * act.mean():.0f}% of run, "
                        f"t $\\in$ [{t[act].min():.2f}, {t[act].max():.2f}] s",
                        fontsize=9)
            aD.set_ylabel(r"$\delta$")
            aD.grid(True, alpha=0.3)
            for a in (aV, aD):
                a.set_xlim(0, t_max)
        axes[0, 0].set_title("CLF value $V(x_t)$   (symlog)")
        axes[-1, 0].set_xlabel("Time [s]")
        axes[-1, 1].set_xlabel("Time [s]")
        fig_v.suptitle("Stability certificate and slack", fontsize=14)
        fig_v.tight_layout()

    # ---------------------------------------------------------
    # Numeric summary
    # ---------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"{'method':<14}{'T [s]':>8}{'min h':>10}{'max h':>11}"
          f"{'clearance':>11}{'v<vmin':>9}{'delta on':>10}")
    print("-" * 78)
    for m in methods:
        df = runs[m][1]
        T = len(df) * dt
        h = _column(df, "h_now_0")
        hmin = f"{np.nanmin(h):.4f}" if h is not None else "--"
        hmax = f"{np.nanmax(h):+.2e}" if h is not None else "--"
        clr = "--"
        if obstacles:
            x, y = _column(df, "state_0"), _column(df, "state_1")
            cx, cy, r = obstacles[0]
            clr = f"{np.nanmin(np.hypot(x - cx, y - cy)) - r:+.4f}"
        v = _column(df, "input_0")
        below = f"{100 * np.mean(v < 0.1 - 1e-9):.1f}%" if v is not None else "--"
        D = _column(df, "delta_0")
        don = f"{100 * np.mean(D > 1e-9):.1f}%" if D is not None else "--"
        print(f"{m:<14}{T:>8.3f}{hmin:>10}{hmax:>11}{clr:>11}{below:>9}{don:>10}")
    print("-" * 78)
    print("clearance = min distance to obstacle minus its radius; "
          "negative means penetration")
    print("v<vmin    = fraction of steps below the input box, i.e. the QP "
          "fallback firing")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)
    out = []
    for fig, name in [(fig_traj, "trajectory_comparison"),
                      (fig_u, "input_comparison"),
                      (fig_h, "h_comparison"),
                      (fig_v, "clf_slack_comparison")]:
        if fig is None:
            continue
        p = os.path.join(results_dir, f"{name}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        out.append(p)
        print(f"Saved {p}")

    if show:
        plt.show()
    else:
        for fig in (fig_traj, fig_u, fig_h, fig_v):
            if fig is not None:
                plt.close(fig)

    return out


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob("simulation_*.xlsx"))   # or an explicit list
    # plot_results_from_excel(
    #     files,
    #     dt=0.005,
    #     goal=[4.5, 4.5],
    #     # obstacles=[(2.0, 2.5, 0.3)],
    #     obstacles = [[2.0, 2.5, 0.3], [3.0, 3.5, 0.2], [1.5, 1.8, 0.1]],
    #     order=["pure_backup", "rpcbf", "clf_cbf"],
    #     show=True,
    #     results_dir="Results",
    # )

    plot_results_from_excel(
    ["plotter/sim_pure_backup.xlsx", "plotter/sim_rpcbf_full.xlsx", "plotter/sim_rpcbf_terminated.xlsx"],
    dt=0.005, goal=[3.0, 2.7], obstacles=[(2.0, 2.5, 0.3)],
    labels=["pure_backup", "rpcbf (full horizon)", "rpcbf (backup-terminated)"],
    show=True, results_dir="Results")