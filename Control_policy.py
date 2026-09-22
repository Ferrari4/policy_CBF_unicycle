import numpy as np
from Get_obstacles import ObsDyn

class policy:
    def __init__(self, v_max, om_max, obs_class: ObsDyn, 
                 eps=0.6, process="batch"):
        self.v_max = v_max
        self.om_max = om_max
        self.eps = eps
        self.process = process
        self.kpw = 2.0 # proportional gain for heading error
        self.obs_class = obs_class          # ObsDyn: obstacle positions come from obs_class.pos_now()
        self.rng = np.random.default_rng(12345)

    def random_policy(self, state, goal):
        state = np.asarray(state).reshape(-1, 3)
        B = state.shape[0]
        v = self.rng.uniform(low=0.0,high=self.v_max,size=B)
        om = self.rng.uniform(low=-self.om_max, high=self.om_max,size=B)
        u = np.column_stack((v, om))
        if self.process == "single":
            assert B == 1
            return u[0]
        return u

    def proportional_policy(self, state, goal):
        state = np.asarray(state).reshape(-1, 3)
        px, py, th = state[:, 0], state[:, 1], state[:, 2]
        goal = np.asarray(goal, dtype=float).reshape(-1, 2)   
        dx, dy = goal[:, 0] - px, goal[:, 1] - py
        th_goal = np.arctan2(dy, dx)
        th_err = np.arctan2(
            np.sin(th_goal - th),
            np.cos(th_goal - th)
        )
        v = self.v_max * np.tanh(np.hypot(dx, dy))
        
        # d = np.hypot(dx, dy)        
        # v = np.where(d < 0.5, 0.0, self.v_max)  
        
        om = self.kpw * th_err
        # om = np.where(d < 0.5, 0.0, om)  

        if self.process == "single":
            assert state.shape[0] == 1
            return np.array([v[0], om[0]])
        return np.stack([v, om], axis=1)

    def constant_policy(self, state, goal):
        state = np.asarray(state).reshape(-1, 3)
        u = np.tile(np.array([0.5, 0.0]), (state.shape[0], 1))
        if self.process == "single":
            assert state.shape[0] == 1
            return u[0]
        return u

    def backup_policy(self, state, goal):
        state = np.asarray(state).reshape(-1, 3)
        p = state[:, :2]                      
        th = state[:, 2] 
        # NOTE: uses the obstacle position at the CURRENT sim time for every rollout step
        # (the policy is not told the rollout time). Approximation for moving obstacles.
        obs_pos = self.obs_class.pos_now()                 # (n_obs, 2)
        diff = p[:, None, :] - obs_pos[None, :, :]         # (B, n_obs, 2)
        dists = np.linalg.norm(diff, axis=2)               # (B, n_obs)
        i = np.argmin(dists, axis=1)                       # (B,)
        closest_diff = diff[np.arange(len(state)), i]      # (B, 2)
        closest_dist = dists[np.arange(len(state)), i]     # (B,)
        n = closest_diff / closest_dist[:, None]           # (B, 2)
        r = np.column_stack((-np.sin(th), np.cos(th)))     # (B, 2)
        n_dot_r = np.sum(n * r, axis=1)                    # (B,)
        v_b = np.full(len(state), self.v_max)
        om_b = self.om_max * np.tanh(n_dot_r / self.eps)
        u = np.column_stack((v_b, om_b))
        if self.process == "single":
            assert state.shape[0] == 1
            return u[0]
        return u

    