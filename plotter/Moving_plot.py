import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class live_plotter:
    def __init__(
        self,
        xlim=(-1.0, 6.0),
        ylim=(-2.0, 6.0),
        robot_length=0.25,
        obs_pos=None,
        obs_radius=None
    ):
        self.robot_length = robot_length

        plt.close("all")
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

        # Robot current position
        self.robot_point, = self.ax.plot(
            [],
            [],
            "bo",
            markersize=8,
            label="Robot"
        )

        # Robot heading
        self.robot_heading, = self.ax.plot(
            [],
            [],
            "b-",
            linewidth=2
        )

        # Robot trajectory
        self.robot_path, = self.ax.plot(
            [],
            [],
            "b--",
            linewidth=1
        )

        # Goal current position
        self.goal_point, = self.ax.plot(
            [],
            [],
            "r*",
            markersize=12,
            label="Goal"
        )

        # Goal trajectory
        self.goal_path, = self.ax.plot(
            [],
            [],
            "r--",
            linewidth=1
        )

        # ----------------------------------
        # Obstacles
        # ----------------------------------

        self.obs_pos = None
        self.obs_radius = None
        self.obstacle_circles = []

        if obs_pos is not None:

            self.obs_pos = np.asarray(
                obs_pos,
                dtype=float
            ).reshape(-1, 2)

            # Obstacle centers
            self.obstacle_points, = self.ax.plot(
                self.obs_pos[:, 0],
                self.obs_pos[:, 1],
                "kx",
                markersize=10,
                markeredgewidth=2,
                label="Obstacle"
            )

            # ----------------------------------
            # Obstacle boundary circles
            # ----------------------------------

            if obs_radius is not None:

                self.obs_radius = np.asarray(
                    obs_radius,
                    dtype=float
                )

                # Allow one radius for all obstacles
                if self.obs_radius.ndim == 0:
                    self.obs_radius = np.full(
                        self.obs_pos.shape[0],
                        float(self.obs_radius)
                    )

                else:
                    self.obs_radius = self.obs_radius.reshape(-1)

                if self.obs_radius.shape[0] != self.obs_pos.shape[0]:
                    raise ValueError(
                        "obs_radius must either be a scalar or have "
                        "one radius per obstacle.\n"
                        f"Number of obstacles: {self.obs_pos.shape[0]}\n"
                        f"Number of radii:     {self.obs_radius.shape[0]}"
                    )

                for pos, radius in zip(
                    self.obs_pos,
                    self.obs_radius
                ):

                    circle = Circle(
                        (pos[0], pos[1]),
                        radius,
                        fill=False,
                        linestyle="--",
                        linewidth=2
                    )

                    self.ax.add_patch(circle)

                    self.obstacle_circles.append(circle)

        else:
            self.obstacle_points = None

        self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def update(
        self,
        trajectory,
        goal=None,
        obs_traj=None,
        pause=0.05
    ):
        """
        Animate a full robot trajectory.

        Parameters
        ----------
        trajectory : np.ndarray
            Shape (N, 3):
                [px, py, theta]

        goal : np.ndarray or None
            Static goal:
                shape (2,)

            Moving goal trajectory:
                shape (N, 2)

        obs_traj : np.ndarray or None
            Obstacle centre positions over time, shape (N, n_obs, 2).
            None -> obstacles stay where __init__ drew them.

        pause : float
            Delay between each trajectory state, in seconds.
        """

        trajectory = np.asarray(trajectory, dtype=float)

        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError(
                f"trajectory must have shape (N, 3), got {trajectory.shape}"
            )

        N = trajectory.shape[0]

        # ----------------------------------
        # Prepare goal
        # ----------------------------------

        static_goal = False
        moving_goal = False

        if goal is not None:
            goal = np.asarray(goal, dtype=float)

            if goal.ndim == 1:
                if goal.shape != (2,):
                    raise ValueError(
                        f"Static goal must have shape (2,), got {goal.shape}"
                    )
                static_goal = True

            elif goal.ndim == 2:
                if goal.shape != (N, 2):
                    raise ValueError(
                        "Moving goal must have shape (N, 2).\n"
                        f"trajectory: {trajectory.shape}\n"
                        f"goal:       {goal.shape}"
                    )
                moving_goal = True

            else:
                raise ValueError(
                    "goal must have shape (2,) or (N, 2)"
                )

        # ----------------------------------
        # Prepare obstacles
        # ----------------------------------

        moving_obs = False

        if obs_traj is not None:
            if self.obstacle_points is None:
                raise ValueError(
                    "obs_traj given but no obstacles were passed to __init__"
                )

            obs_traj = np.asarray(obs_traj, dtype=float)
            n_obs = self.obs_pos.shape[0]

            if obs_traj.shape != (N, n_obs, 2):
                raise ValueError(
                    "obs_traj must have shape (N, n_obs, 2).\n"
                    f"trajectory: {trajectory.shape}\n"
                    f"obs_traj:   {obs_traj.shape}"
                )
            moving_obs = True

            # Dashed trace of each obstacle centre (created once, reused)
            if not hasattr(self, "obstacle_paths"):
                self.obstacle_paths = [
                    self.ax.plot([], [], "k--", linewidth=1, alpha=0.6)[0]
                    for _ in range(n_obs)
                ]

        # ----------------------------------
        # Animate trajectory
        # ----------------------------------

        for k in range(N):

            px, py, theta = trajectory[k]

            # Robot path up to current state
            self.robot_path.set_data(
                trajectory[:k + 1, 0],
                trajectory[:k + 1, 1]
            )

            # Current robot position
            self.robot_point.set_data(
                [px],
                [py]
            )

            # Robot heading
            hx = px + self.robot_length * np.cos(theta)
            hy = py + self.robot_length * np.sin(theta)

            self.robot_heading.set_data(
                [px, hx],
                [py, hy]
            )

            # ----------------------------------
            # Goal
            # ----------------------------------

            if static_goal:

                gx, gy = goal

                self.goal_point.set_data(
                    [gx],
                    [gy]
                )

                self.goal_path.set_data(
                    [],
                    []
                )

            elif moving_goal:

                gx, gy = goal[k]

                # Current goal
                self.goal_point.set_data(
                    [gx],
                    [gy]
                )

                # Goal history up to current time
                self.goal_path.set_data(
                    goal[:k + 1, 0],
                    goal[:k + 1, 1]
                )

            # ----------------------------------
            # Obstacles
            # ----------------------------------

            if moving_obs:

                # Current obstacle centres
                self.obstacle_points.set_data(
                    obs_traj[k, :, 0],
                    obs_traj[k, :, 1]
                )

                # Move each boundary circle and extend its trace
                for i, circle in enumerate(self.obstacle_circles):
                    circle.center = (obs_traj[k, i, 0], obs_traj[k, i, 1])
                    self.obstacle_paths[i].set_data(
                        obs_traj[:k + 1, i, 0],
                        obs_traj[:k + 1, i, 1]
                    )

            # ----------------------------------
            # Redraw
            # ----------------------------------

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # Pause at EVERY state
            plt.pause(pause)
    
    def close(self):
        plt.ioff()
        plt.show()