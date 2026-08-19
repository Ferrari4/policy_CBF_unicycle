"""
unicycle_sim.py

Basic unicycle model simulator + plotter.

State:  q = [x, y, theta]
Input:  u = [v, omega]
Dynamics:
    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = omega

Integration:
    - Default: built-in fixed-step manual RK4 (no adaptive stepping).
    - Optional: pass your own solver via `integrator` in the constructor.
      Expected signature:
          integrator(f, state, u, dt) -> next_state (np.ndarray shape (3,))
      where f(state, u) -> state_dot. This lets you plug in the manual
      solver from the Policy-CBF framework without touching this class.

Input handling in simulate():
    - u as tuple/list/array of shape (2,)   -> constant input held for whole horizon (needs T)
    - u as array-like of shape (N, 2)       -> per-step input sequence, horizon = N * dt
    - u as callable u(t, state) -> (v, w)   -> feedback / time-varying policy (needs T)
"""

import numpy as np
import matplotlib.pyplot as plt


class UnicycleSim:
    def __init__(self, dt=0.01, integrator=None):
        self.dt = float(dt)
        self.integrator = integrator          # external solver hook (or None -> built-in RK4)
        self.trajectories = []                # list of dicts: {t, states, inputs, label}
        self.obstacles = []                   # list of (cx, cy, radius)

    # ------------------------------------------------------------------ dynamics
    @staticmethod
    def dynamics(state, u):
        """state = [x, y, theta], u = [v, omega] -> state_dot"""
        _, _, th = state
        v, w = u
        return np.array([v * np.cos(th), v * np.sin(th), w])

    # ------------------------------------------------------------------ integration
    def _rk4_step(self, state, u, dt):
        f = self.dynamics
        k1 = f(state, u)
        k2 = f(state + 0.5 * dt * k1, u)
        k3 = f(state + 0.5 * dt * k2, u)
        k4 = f(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _step(self, state, u, dt):
        if self.integrator is not None:
            return np.asarray(self.integrator(self.dynamics, state, u, dt), dtype=float)
        return self._rk4_step(state, u, dt)

    # ------------------------------------------------------------------ input normalization
    def _resolve_input(self, u, T):
        """
        Returns (u_func, n_steps) where u_func(k, t, state) -> np.ndarray (2,).
        Handles: constant input, per-step sequence, or callable policy.
        """
        if callable(u):
            if T is None:
                raise ValueError("T (horizon in seconds) required when u is a callable policy.")
            n_steps = int(round(T / self.dt))
            return (lambda k, t, s: np.asarray(u(t, s), dtype=float)), n_steps

        u_arr = np.asarray(u, dtype=float)

        if u_arr.ndim == 1:
            if u_arr.shape != (2,):
                raise ValueError(f"Single input must have shape (2,), got {u_arr.shape}")
            if T is None:
                raise ValueError("T (horizon in seconds) required for a constant input.")
            n_steps = int(round(T / self.dt))
            return (lambda k, t, s: u_arr), n_steps

        if u_arr.ndim == 2:
            if u_arr.shape[1] != 2:
                raise ValueError(f"Input sequence must have shape (N, 2), got {u_arr.shape}")
            n_steps = u_arr.shape[0]
            return (lambda k, t, s: u_arr[k]), n_steps

        raise ValueError(f"Unsupported input format with ndim={u_arr.ndim}")

    # ------------------------------------------------------------------ simulation
    def simulate(self, x0, u, T=None, label=None):
        """
        Simulate one trajectory and store it internally.

        x0    : array-like (3,) initial state [x, y, theta]
        u     : constant input (2,), sequence (N, 2), or callable u(t, state)
        T     : horizon in seconds (required for constant / callable inputs)
        label : optional legend label

        Returns dict with keys: t (n+1,), states (n+1, 3), inputs (n, 2), label
        """
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (3,):
            raise ValueError(f"x0 must have shape (3,), got {x0.shape}")

        u_func, n_steps = self._resolve_input(u, T)

        states = np.zeros((n_steps + 1, 3))
        inputs = np.zeros((n_steps, 2))
        t_grid = np.arange(n_steps + 1) * self.dt

        states[0] = x0
        for k in range(n_steps):
            uk = u_func(k, t_grid[k], states[k])
            inputs[k] = uk
            states[k + 1] = self._step(states[k], uk, self.dt)

        traj = {
            "t": t_grid,
            "states": states,
            "inputs": inputs,
            "label": label if label is not None else f"traj {len(self.trajectories)}",
        }
        self.trajectories.append(traj)
        return traj

    def simulate_batch(self, x0_list, u_list, T=None, labels=None):
        """
        Simulate multiple trajectories.

        x0_list : list of initial states
        u_list  : single input spec applied to all, OR list of input specs (one per x0)
        """
        n = len(x0_list)
        # If u_list is not a per-trajectory list, broadcast it.
        if not (isinstance(u_list, (list, tuple)) and len(u_list) == n
                and not np.isscalar(u_list[0])):
            u_list = [u_list] * n
        if labels is None:
            labels = [None] * n

        return [self.simulate(x0, u, T=T, label=lb)
                for x0, u, lb in zip(x0_list, u_list, labels)]

    # ------------------------------------------------------------------ environment
    def add_obstacle(self, cx, cy, radius):
        self.obstacles.append((float(cx), float(cy), float(radius)))

    def clear(self):
        self.trajectories = []
        self.obstacles = []

    # ------------------------------------------------------------------ plotting
    @staticmethod
    def _robot_triangle(state, size=0.15):
        """Triangle vertices for the robot at a given pose (nose points along theta)."""
        x, y, th = state
        pts_body = np.array([
            [ size,        0.0        ],   # nose
            [-0.6 * size,  0.5 * size ],
            [-0.6 * size, -0.5 * size ],
        ])
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
        return pts_body @ R.T + np.array([x, y])

    def plot(self, ax=None, robot_size=0.15, triangle_every=None,
             title="Unicycle trajectories", show=True):
        """
        Plot all stored trajectories with start/end markers, obstacles,
        and robot triangles at start and end poses.

        triangle_every : if an int N, also draw a triangle every N steps along
                         each trajectory (useful to see heading evolution).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        # obstacles
        for (cx, cy, r) in self.obstacles:
            circ = plt.Circle((cx, cy), r, facecolor="lightcoral",
                              edgecolor="darkred", alpha=0.5, zorder=1)
            ax.add_patch(circ)
            ax.plot(cx, cy, "x", color="darkred", markersize=8, zorder=2)
        # single legend entries for obstacle shape/center
        if self.obstacles:
            ax.plot([], [], "s", color="lightcoral", markeredgecolor="darkred",
                    label="obstacle")
            ax.plot([], [], "x", color="darkred", label="obstacle center")

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for i, traj in enumerate(self.trajectories):
            s = traj["states"]
            c = colors[i % 10]

            ax.plot(s[:, 0], s[:, 1], "-", color=c, linewidth=1.5,
                    label=traj["label"], zorder=3)

            # start / end markers
            ax.plot(s[0, 0], s[0, 1], "o", color="green", markersize=9,
                    markeredgecolor="black", zorder=5)
            ax.plot(s[-1, 0], s[-1, 1], "*", color="gold", markersize=15,
                    markeredgecolor="black", zorder=5)

            # robot triangles: start pose, end pose, optional intermediate
            poses = [s[0], s[-1]]
            if triangle_every is not None:
                poses = list(s[::triangle_every]) + [s[-1]]
            for p in poses:
                tri = self._robot_triangle(p, size=robot_size)
                ax.fill(tri[:, 0], tri[:, 1], color=c, alpha=0.6,
                        edgecolor="black", linewidth=0.8, zorder=4)

        # single legend entries for start/end
        ax.plot([], [], "o", color="green", markeredgecolor="black", label="start")
        ax.plot([], [], "*", color="gold", markeredgecolor="black", label="end")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        if show:
            plt.show()
        return ax

    # ------------------------------------------------------------------ external plotting
    @staticmethod
    def _as_traj_list(x, name, width):
        """
        Normalize a single trajectory or a list of trajectories into a list
        of float arrays. A single trajectory is anything 2-D shaped (*, width);
        a list of trajectories is a list/tuple of such arrays.
        """
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                return [x.astype(float)]
            if x.ndim == 3:
                return [xi.astype(float) for xi in x]
            raise ValueError(f"{name}: expected 2-D or 3-D array, got ndim={x.ndim}")

        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError(f"{name} is empty.")
            first = np.asarray(x[0], dtype=float)
            if first.ndim == 1:
                # single trajectory given as a list of rows
                return [np.asarray(x, dtype=float)]
            return [np.asarray(xi, dtype=float) for xi in x]

        raise ValueError(f"{name}: unsupported type {type(x)}")

    def plot_external_trajectories(self, states_list, inputs_list, labels=None,
                                   dt=None, robot_size=0.15, triangle_every=None,
                                   title="Unicycle trajectories", show=True):
        """
        Plot externally supplied state trajectories (XY figure) and input
        signals (v / omega vs time figure).

        states_list : one trajectory of shape (N+1, 3), or a list of them
        inputs_list : one input history of shape (N, 2), or a list of them.
                      Length N+1 (one input per state) is also accepted;
                      the trailing input is dropped with a warning.
        labels      : str or list of str, optional
        dt          : timestep used to build the input time axis. Defaults to
                      self.dt -- pass explicitly if the external data was
                      generated with a different step.

        Returns (ax_traj, (ax_v, ax_omega)).
        """
        states_list = self._as_traj_list(states_list, "states_list", 3)
        inputs_list = self._as_traj_list(inputs_list, "inputs_list", 2)

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

        dt = self.dt if dt is None else float(dt)

        # ---------------- figure 1: XY trajectories ----------------
        fig_traj, ax_traj = plt.subplots(figsize=(8, 8))

        for (cx, cy, r_obs) in self.obstacles:
            circ = plt.Circle((cx, cy), r_obs, facecolor="lightcoral",
                              edgecolor="darkred", alpha=0.5, zorder=1)
            ax_traj.add_patch(circ)
            ax_traj.plot(cx, cy, "x", color="darkred", markersize=8, zorder=2)
        if self.obstacles:
            ax_traj.plot([], [], "s", color="lightcoral",
                         markeredgecolor="darkred", label="obstacle")
            ax_traj.plot([], [], "x", color="darkred", label="obstacle center")

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

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
            ax_traj.plot(states[-1, 0], states[-1, 1], "*", color="gold",
                         markersize=15, markeredgecolor="black", zorder=5)

            # robot triangles
            poses = [states[0], states[-1]]
            if triangle_every is not None:
                poses = list(states[::triangle_every])
                if not np.array_equal(poses[-1], states[-1]):
                    poses.append(states[-1])
            for pose in poses:
                tri = self._robot_triangle(pose, size=robot_size)
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

        if show:
            plt.show()
        return ax_traj, (ax_v, ax_omega)