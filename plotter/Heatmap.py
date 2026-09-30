import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def heatmap_V(safety, goal, theta=0.0, xlim=(0.0, 6.0),
    ylim=(0.0, 6.0), n=256, path=None,):
    
    """Instantaneous certificate at a fixed robot heading theta. No rollouts."""
    if path is None:
        path = os.path.join("plotter/Results", "heatmap_V.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    goal = np.asarray(goal, dtype=float)
    
    """Instantaneous certificate at a fixed robot heading theta. No rollouts."""
    goal = np.asarray(goal, dtype=float)

    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)

    states = np.stack(
        [X.ravel(), Y.ravel(), np.full(X.size, theta)],
        axis=-1,
    )  # (B, 3)

    goals = np.broadcast_to(goal, (states.shape[0], goal.size))

    # Match the certificate's existing (batch, time, state) interface.
    values = safety.clf.clf_certificate(
        states[:, None, :],   # (B, 1, 3)
        goals[:, None, :],    # (B, 1, 2)
    )
    Z = np.asarray(values).reshape(X.shape)

    obs_pos = np.asarray(safety.obs_class.pos_now()).reshape(-1, 2)
    radii = np.broadcast_to(
        np.asarray(safety.obs_class.R_O).reshape(-1),
        (len(obs_pos),),
    )

    inside = np.zeros(X.shape, dtype=bool)
    for (cx, cy), radius in zip(obs_pos, radii):
        inside |= np.hypot(X - cx, Y - cy) <= radius

    Z = np.ma.masked_where(inside, Z)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(
        X, Y, Z,
        cmap="viridis",
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label=r"$V(x)$")

    contours = ax.contour(
        X, Y, Z,
        levels=50,
        colors="black",
        linewidths=0.6,
        alpha=0.7,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    
    obs_pos = np.asarray(safety.obs_class.pos_now()).reshape(-1, 2)
    radii = np.broadcast_to(
        np.asarray(safety.obs_class.R_O).reshape(-1),
        (len(obs_pos),),
    )

    for (cx, cy), radius in zip(obs_pos, radii):
        ax.add_patch(
            Circle(
                (cx, cy), radius,
                facecolor="lightcoral",
                edgecolor="darkred",
                alpha=0.6,
            )
        )

        if getattr(safety.clf, "obs_class", None) is not None:
            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius + safety.clf.rho_margin,
                    fill=False,
                    edgecolor="darkred",
                    linestyle=":",
                )
            )

    ax.plot(
        goal[0], goal[1], "*",
        color="gold",
        markeredgecolor="black",
        markersize=14,
    )

    ax.set(
        xlabel="x [m]",
        ylabel="y [m]",
        title=rf"Instantaneous $V(x)$ — $\theta={theta:.2f}$ rad",
        xlim=xlim,
        ylim=ylim,
    )
    ax.set_aspect("equal")
    fig.tight_layout()

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return X, Y, Z