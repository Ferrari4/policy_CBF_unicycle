import numpy as np
from Dynamics import sys_dynm_dd

class v_certificate:
    def __init__(self ,dynamic_class: sys_dynm_dd, policy_name):
        self.sys_dynm = dynamic_class
        self.nx = self.sys_dynm.nx
        self.policy_name = policy_name
        self.k = 1.0
        self.nV = 1  

    def get_params(self, x, goal): # Batch ready
        x, goal = np.asarray(x), np.asarray(goal)
        px, py, theta = x[..., 0], x[..., 1], x[..., 2]
        dx, dy = px - goal[..., 0], py - goal[..., 1]
        d = np.hypot(dx, dy)
        theta_goal = np.arctan2(-dy, -dx)
        theta_err = np.arctan2(np.sin(theta_goal - theta), 
                               np.cos(theta_goal - theta))
        return dx, dy, d, theta_err

    def clf_value(self, x, goal): # Should not be Batched since a Valid CLF is not previewd
        x = self.sys_dynm.chk_x(x) 
        _, _, d, theta_err = self.get_params(x, goal)
        V = d**2 * (0.5 + self.k * (1.0 - np.cos(theta_err))) # CLF
        return V

    def clf_value_gradient(self, x, goal): # Should not be Batched since a Valid CLF is not previewd
        x = self.sys_dynm.chk_x(x) 
        dx, dy, d, theta_err = self.get_params(x, goal)
        sin_err = np.sin(theta_err)
        cos_err = np.cos(theta_err)
        A = 0.5 + self.k * (1.0 - cos_err)
        # Analytical gradient
        dV_dpx = 2.0 * dx * A - self.k * dy * sin_err
        dV_dpy = 2.0 * dy * A + self.k * dx * sin_err
        dV_dtheta = -self.k * (d**2) * sin_err
        grad_V = np.array([
            dV_dpx,
            dV_dpy,
            dV_dtheta
        ])
        return grad_V

    def clf_certificate(self, x, goal): # Batch ready
        _, _, d, theta_err = self.get_params(x, goal)
        V = 0.5 * d**2 + self.k * (1.0 - np.cos(theta_err))
        return V[..., None]

    def evaluate_v_trajectory(self, trajectory, goal):
        if trajectory.ndim != 2:
            raise ValueError("trajectory must have shape (horizon + 1, nx)")
        if trajectory.shape[1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {trajectory.shape[1]}")

        return np.stack([self.clf_certificate(state, goal) for state in trajectory], axis=0)

    def aggregate_v(self, t_values, v_values):
        return np.max(v_values, axis=0)

    def compute_v_vmax(self, x0, bH_dstb, goal, include_v0):
        x0 = self.sys_dynm.chk_x(x0)
        bH_dstb = np.asarray(bH_dstb, dtype=float)
        if bH_dstb.ndim != 3:
            raise ValueError("bH_dstb must have shape " "(n_samples, horizon, nd)")
        n_samples, horizon, nd = bH_dstb.shape
        if nd != self.sys_dynm.nd:
            raise ValueError(f"Expected disturbance dimension " f"{self.sys_dynm.nd}, got {nd}")
        sample_vmax = []
        trajectories = []
        value_histories = []
        for H_dstb in bH_dstb:
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, goal=goal,policy_name=self.policy_name)
            if self.sys_dynm.process == "batch":
                assert trajectory.shape[1] == 1
                trajectory = trajectory[:, 0, :]
            v_history = self.evaluate_v_trajectory(trajectory, goal)
            t_history = np.linspace(0.0, self.sys_dynm.dt * (v_history.shape[0] - 1), v_history.shape[0])
            v_values = v_history if include_v0 else v_history[1:]
            t_values = t_history if include_v0 else t_history[1:]
            v_agg = self.aggregate_v(t_values, v_values)
            trajectories.append(trajectory)
            value_histories.append(v_history)
            sample_vmax.append(v_agg)

        bHp1_x = np.stack(trajectories, axis=0)
        bHp1v_v = np.stack(value_histories, axis=0)
        bv_vmax = np.stack(sample_vmax, axis=0)

        assert bHp1_x.shape == (n_samples, horizon + 1, self.nx)
        assert bHp1v_v.shape == (n_samples, horizon + 1, self.nV)
        assert bv_vmax.shape == (n_samples, self.nV)

        v_argmax = np.argmax(bv_vmax, axis=0)
        value_indices = np.arange(self.nV)
        v_vmax = bv_vmax[v_argmax, value_indices]
        vH_dstb = bH_dstb[v_argmax]
        vHp1_x = bHp1_x[v_argmax]
        vHp1v_v = bHp1v_v[v_argmax]

        assert v_vmax.shape == (self.nV,)
        assert vH_dstb.shape == (self.nV, horizon, self.sys_dynm.nd)

        info = {
            "v_argmax": v_argmax,
            "bv_vmax": bv_vmax,
            "vHp1_x": vHp1_x,
            "vHp1v_v": vHp1v_v,
            "bHp1_x": bHp1_x,
            "bHp1v_v": bHp1v_v,
            "v_vmax": v_vmax,
        }

        return v_vmax, vH_dstb, info

    def compute_v_diag(self, x0, vH_dstb, goal ,include_v0):
        v_out = np.zeros(self.nV)
        for j, H_dstb in enumerate(vH_dstb):
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, goal=goal, policy_name=self.policy_name)
            if self.sys_dynm.process == "batch":
                assert trajectory.shape[1] == 1
                trajectory = trajectory[:, 0, :]
            v_history = self.evaluate_v_trajectory(trajectory, goal)
            t_history = np.linspace(0.0, self.sys_dynm.dt * (v_history.shape[0] - 1), v_history.shape[0])
            v_values = v_history if include_v0 else v_history[1:]
            t_values = t_history if include_v0 else t_history[1:]
            v_out[j] = self.aggregate_v(t_values, v_values)[j]
        return v_out

    def get_value_and_grad(self, x0, bH_dstb, goal ,include_v0, eps=1e-5):
        v_vmax, vH_dstb, info = self.compute_v_vmax(x0, bH_dstb, goal, include_v0)
        grad_v_vmax = np.zeros((self.nV, self.nx))
        for i in range(self.nx):
            e = np.zeros(self.nx)
            e[i] = eps
            vp = self.compute_v_diag(x0 + e, vH_dstb, goal, include_v0)
            vm = self.compute_v_diag(x0 - e, vH_dstb, goal, include_v0)
            grad_v_vmax[:, i] = (vp - vm) / (2.0 * eps)

        v0_dstb = vH_dstb[:, 0]
        v_f = np.stack([self.sys_dynm.f(x0, d) for d in v0_dstb], axis=0)
        v_G = np.stack([self.sys_dynm.G(x0, d) for d in v0_dstb], axis=0)

        info["vx_gradvmax"] = grad_v_vmax

        return v_vmax, vH_dstb, grad_v_vmax, v_f, v_G, info

