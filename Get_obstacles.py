import numpy as np


class ObsDyn:
    def __init__(self, layout: str, static: bool, dt: float, barrier_inflate: float = 0.0,
                 amplitude: float = 1.0, freq: float = 1.0, phi: float = 0.0):
        self.static = static
        self.dt = dt
        self.a, self.w, self.phi = amplitude, freq, phi
        self.t = 0.0                                   # sim clock, only step() moves it

        if layout == "multi":
            self.obs0 = np.array([[2.0, 2.5], [3.0, 3.5], [1.5, 1.8]])
            self.R_O  = np.array([0.3, 0.2, 0.1]) + barrier_inflate
        elif layout == "single":
            self.obs0 = np.array([[2.0, 2.5]])
            self.R_O  = np.array([0.3]) + barrier_inflate
        else:
            raise ValueError(layout)

        self.n_obs = self.obs0.shape[0]

    # ---------- obstacle position at time t (t can be a float or an array of times) ----------
    def pos(self, t):
        t = np.atleast_1d(t)                           # shape (H,)   (H = 1 for a single time)
        H = t.shape[0]
        p = np.zeros((H, self.n_obs, 2))
        p[:, :, 0] = self.obs0[:, 0]                   # x never moves
        if self.static:
            p[:, :, 1] = self.obs0[:, 1]
        else:
            y = self.a * np.sin(self.w * t + self.phi)     # (H,)
            p[:, :, 1] = self.obs0[:, 1] + y[:, None]      # each obstacle oscillates about its own y
        return p                                       # (H, n_obs, 2)

    # ---------- obstacle velocity at time t ----------
    def vel(self, t):
        t = np.atleast_1d(t)
        H = t.shape[0]
        v = np.zeros((H, self.n_obs, 2))
        if not self.static:
            v[:, :, 1] = (self.a * self.w * np.cos(self.w * t + self.phi))[:, None]
        return v                                       # (H, n_obs, 2)

    # ---------- convenience ----------
    def pos_now(self):
        return self.pos(self.t)[0]                     # (n_obs, 2)

    def rollout_obs(self, n_samples, t_shift=0.0):
        """Obstacle positions at the n_samples rollout times starting from the current clock.
        t_shift lets you evaluate the same timeline slightly later (used for dV/dt)."""
        ts = self.t + t_shift + np.arange(n_samples) * self.dt
        return self.pos(ts)                            # (n_samples, n_obs, 2)

    # ---------- simulator ----------
    def step(self):
        self.t += self.dt

    def get_obs(self, horizon=None):
        if horizon is None:                            # current obstacle state
            obs_pos = self.pos(self.t)[0]              # (n_obs, 2)
            obs_vel = self.vel(self.t)[0]              # (n_obs, 2)
        else:                                          # predicted over the rollout horizon
            t_roll = self.t + np.arange(horizon) * self.dt
            obs_pos = self.pos(t_roll)                 # (H, n_obs, 2)
            obs_vel = self.vel(t_roll)                 # (H, n_obs, 2)
        return obs_pos, self.R_O, obs_vel


if __name__ == "__main__":
    obs = ObsDyn("multi", static=False, dt=0.1, amplitude=0.5, freq=2.0)
    obs_pos, R_O, obs_vel = obs.get_obs(horizon=8)
    print("obs_pos", obs_pos.shape, " R_O", R_O.shape, " obs_vel", obs_vel.shape)

    # vel consistency check by finite differences
    t0, eps = 0.37, 1e-6
    fd = (obs.pos(t0 + eps) - obs.pos(t0 - eps)) / (2 * eps)
    assert np.allclose(fd, obs.vel(t0), atol=1e-5)
    print("vel matches d/dt pos")

    obs.step()
    print("t =", obs.t, " current obs:", obs.pos_now())
