import numpy as np
from Noise_sampler import noise_test_sampler

class goal_dyn():
    def __init__(self, goal_dyn: str, goal_motion: str ,noise_sampler: noise_test_sampler, init_goal):
        self.goal_dyn = goal_dyn
        self.init_goal = np.array(init_goal)
        self.goal_motion = goal_motion
        self.noise_sampler = noise_sampler

    def call_goal(self, x):
        if self.goal_dyn == "static":
            return self.init_goal

        elif self.goal_dyn == "dynamic":
            assert self.init_goal.shape == 2 
            if self.goal_motion == "det":
                self.init_goal[1] = np.sin(x)

            else:
                noise = self.noise_sampler.uniform_test(rng=1234,scale=1.0)
                self.init_goal[1] = (np.sin(x) + noise) / 2

            return  self.init_goal
