import numpy as np

class policy:
    def __init__(self, v_max=1.0, om_max=2.0, obs_pos=None, eps=0.6, goal=np.array([4.5, 4.5])):
        self.v_max = v_max
        self.om_max = om_max
        self.eps = eps
        self.goal = np.asarray(goal)
        # (n_obs, 2); default = single obstacle from Control_env.py
        self.obs_pos = np.atleast_2d(obs_pos if obs_pos is not None
                                     else np.array([2.0, 2.5]))

    def proportional_policy(self, state):
        kpw = 2.0 # proportional gain for heading error
        px, py, th = state
        dx, dy = self.goal[0] - px, self.goal[1] - py
        th_goal = np.arctan2(dy, dx)
        th_err = np.arctan2(np.sin(th_goal - th), np.cos(th_goal - th))
        v = self.v_max * np.tanh(np.hypot(dx, dy)) # Automcatically bounds speed to v_max
        om = np.clip(kpw * th_err, -self.om_max, self.om_max)
        return np.array([v, om])

    def constant_policy(self, state):
        return np.array([0.5, 0.0])

    def backup_policy(self, state):
        # turn-away backup (extracted from your Backup_cbf), closest obstacle
        px, py, th = state
        p = np.array([px, py])
        dists = np.linalg.norm(p - self.obs_pos, axis=1)
        i = np.argmin(dists)                       # closest obstacle
        n = (p - self.obs_pos[i]) / dists[i]       # outward normal
        r = np.array([-np.sin(th), np.cos(th)])    # heading-perpendicular
        v_b = self.v_max
        om_b = self.om_max * np.tanh((n @ r) / self.eps)
        return np.array([v_b, om_b])
    
    