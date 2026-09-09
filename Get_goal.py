import numpy as np
from Noise_sampler import noise_test_sampler

class goal_dyn:
    def __init__(self, goal_dyn: str, goal_motion: str,
                 noise_sampler: noise_test_sampler, init_goal, dt,
                 A=0.5, w=0.5, sigma=0.1):
        self.goal_dyn = goal_dyn
        self.goal_motion = goal_motion
        self.noise_sampler = noise_sampler
        self.init_goal = np.array(init_goal, dtype=float)
        assert self.init_goal.shape == (2,), "goal must be [x, y]"
        self.goal_update = self.init_goal.copy()
        self.dt = float(dt)
        self.t = 0.0
        self.A, self.w, self.sigma = A, w, sigma 

    def call_goal(self):
        if self.goal_dyn == "static":
            return self.init_goal

        elif self.goal_dyn == "sin_y":
            y = self.init_goal[1] + self.A * np.sin(self.w * self.t)
            if self.goal_motion == "stoc":
                y += self.noise_sampler.rng.uniform(-self.sigma, self.sigma)
            elif self.goal_motion != "det":
                raise ValueError(f"Unknown goal_motion: {self.goal_motion}")
            self.goal_update[1] = y
            self.t += self.dt
            return self.goal_update

        elif self.goal_dyn == "random":
            step = self.noise_sampler.rng.normal(0.0, self.sigma, size=2)
            return self.init_goal + step

        else:
            raise ValueError(f"Unknown goal_dyn: {self.goal_dyn}")
        