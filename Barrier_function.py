import numpy as np
from scipy.interpolate import CubicSpline

from Dynamics import sys_dynm_dd
from Get_obstacles import ObsDyn

class h_certificate:
    def __init__(self, dynamic_class: sys_dynm_dd, obs_class: ObsDyn,
                 policy_h="policy_h", policy_name="proportional_policy", delta=0.3):

        self.sys_dynm = dynamic_class
        self.v_max = self.sys_dynm.controller.v_max
        self.om_max = self.sys_dynm.controller.om_max
        self.nx = self.sys_dynm.nx          # state dimension [px, py, th]
        self.obs_class = obs_class          # ObsDyn: positions are a function of time
        self.R_O = np.atleast_1d(np.asarray(obs_class.R_O, dtype=float))
        self.nh = self.R_O.shape[0]         # one barrier per obstacle
        self.delta = delta                  # heading inflation term
        self.policy_h = policy_h            # "policy_h" | "backup_h" | "mixed_h"
        self.policy_name = policy_name
     
    # Convention: h > 0 unsafe, safe set = {h <= 0}
    def h_function(self, state):
        state = self.sys_dynm.chk_x(state)
        px, py, th = state
        p = np.array([px, py])
        q = np.array([np.cos(th), np.sin(th)])
        obs_pos = self.obs_class.pos_now()  # (n_obs, 2) obstacle position NOW
        h = np.zeros(self.nh)
        for i in range(self.nh):
            diff = p - obs_pos[i]
            D = np.linalg.norm(diff)
            n = diff / D
            h[i] = -(D - self.R_O[i] + self.delta * (n @ q))
        return h

    def h_fun_backup(self, state):
        # Terminal certificate: h_b <= 0 iff heading component points away
        # from the obstacle. Only meaningful at the end of a backup rollout.
        state = self.sys_dynm.chk_x(state)
        px, py, th = state
        p = np.array([px, py])
        q = np.array([np.cos(th), np.sin(th)])
        obs_pos = self.obs_class.pos_now()
        h_b = np.zeros(self.nh)
        for i in range(self.nh):
            diff = p - obs_pos[i]
            D = np.linalg.norm(diff)
            n = diff / D
            h_b[i] = -(self.v_max * (n @ q))
        return h_b

    def evaluate_h_trajectory(self, trajectory):
        if trajectory.ndim != 2:
            raise ValueError("trajectory must have shape (horizon + 1, nx)")
        if trajectory.shape[1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {trajectory.shape[1]}")

        if self.policy_h == "policy_h":
            return np.stack([self.h_function(state) for state in trajectory], axis=0)

        elif self.policy_h == "backup_h":
            return np.stack([self.h_fun_backup(state) for state in trajectory], axis=0)

        elif self.policy_h == "mixed_h":
            # h on all states, h_b replacing h at the terminal state
            h_hist = [self.h_function(state) for state in trajectory[:-1]]
            h_hist.append(self.h_fun_backup(trajectory[-1]))
            return np.stack(h_hist, axis=0)

        else:
            raise ValueError(f"Unknown policy_h: {self.policy_h}")

    def compute_h_hmax(self, x0, bH_dstb, goal, include_h0, max_type="cubic_spline"):

        x0 = self.sys_dynm.chk_x(x0)
        bH_dstb = np.asarray(bH_dstb, dtype=float)
        if bH_dstb.ndim != 3:
            raise ValueError("bH_dstb must have shape " "(n_samples, horizon, nd)")
        n_samples, horizon, nd = bH_dstb.shape
        if nd != self.sys_dynm.nd:
            raise ValueError(f"Expected disturbance dimension " f"{self.sys_dynm.nd}, got {nd}")

        sample_hmax = []
        trajectories = []
        barrier_histories = []

        for H_dstb in bH_dstb:
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, goal=goal ,policy_name=self.policy_name)
            h_history = self.evaluate_h_trajectory(trajectory)
            t_history = np.linspace(0.0, self.sys_dynm.dt * (h_history.shape[0] - 1), h_history.shape[0])
            h_values = h_history if include_h0 else h_history[1:]
            t_values = t_history if include_h0 else t_history[1:]

            if max_type == "cubic_spline":
                h_max = np.array([
                    self.max_cubic_spline(t_values, h_values[:, j])
                    for j in range(self.nh)
                ])
            else:
                h_max = np.max(h_values, axis=0)

            trajectories.append(trajectory)
            barrier_histories.append(h_history)
            sample_hmax.append(h_max)

        bHp1_x = np.stack(trajectories, axis=0)
        bHp1h_h = np.stack(barrier_histories, axis=0)
        bh_hmax = np.stack(sample_hmax, axis=0)

        assert bHp1_x.shape == (n_samples, horizon + 1, self.nx)
        assert bHp1h_h.shape == (n_samples, horizon + 1, self.nh)
        assert bh_hmax.shape == (n_samples, self.nh)

        h_argmax = np.argmax(bh_hmax, axis=0)
        barrier_indices = np.arange(self.nh)
        h_hmax = bh_hmax[h_argmax, barrier_indices]
        hH_dstb = bH_dstb[h_argmax]
        hHp1_x = bHp1_x[h_argmax]
        hHp1h_h = bHp1h_h[h_argmax]

        assert h_hmax.shape == (self.nh,)
        assert hH_dstb.shape == (self.nh, horizon, self.sys_dynm.nd)

        info = {
            "h_argmax": h_argmax,
            "bh_hmax": bh_hmax,
            "hHp1_x": hHp1_x,
            "hHp1h_h": hHp1h_h,
            "bHp1_x": bHp1_x,
            "bHp1h_h": bHp1h_h,
            "h_hmax": h_hmax,
        }

        return h_hmax, hH_dstb, info

    def max_cubic_spline(self, t, y, n_prev=5):
        n = y.shape[0]
        n_total = 2 * n_prev + 1

        if n < n_total:
            return float(np.max(y))

        idx = int(np.argmax(y))
        s = idx - n_prev
        s += max(0, -s)
        s -= max(0, s + n_total - n)

        t_win = t[s:s + n_total]
        y_win = y[s:s + n_total]

        spl = CubicSpline(t_win, y_win, bc_type="not-a-knot", extrapolate=False)
        crit = np.atleast_1d(spl.derivative().roots(extrapolate=False))
        crit = crit[np.isfinite(crit)]
        cand = np.concatenate([t_win, crit])
        vals = spl(cand)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float(np.max(y))

        return float(np.max(vals))

    def compute_h_hmax_diag(self, x0, hH_dstb, goal, include_h0, max_type="cubic_spline"):
        h_out = np.zeros(self.nh)
        for j, H_dstb in enumerate(hH_dstb):
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, goal=goal,policy_name=self.policy_name)
            h_history = self.evaluate_h_trajectory(trajectory)
            t_history = np.linspace(0.0, self.sys_dynm.dt * (h_history.shape[0] - 1), h_history.shape[0])
            h_values = h_history if include_h0 else h_history[1:]
            t_values = t_history if include_h0 else t_history[1:]
            if max_type == "cubic_spline":
                h_out[j] = self.max_cubic_spline(t_values, h_values[:, j])
            else:
                h_out[j] = np.max(h_values[:, j])
        return h_out

    def get_value_and_grad(self, x0, bH_dstb, goal ,include_h0, max_type="cubic_spline", eps=1e-5):
        h_hmax, hH_dstb, info = self.compute_h_hmax(x0, bH_dstb, goal, include_h0, max_type)
        grad_h_hmax = np.zeros((self.nh, self.nx))
        for i in range(self.nx):
            e = np.zeros(self.nx)
            e[i] = eps
            vp = self.compute_h_hmax_diag(x0 + e, hH_dstb, goal, include_h0, max_type)
            vm = self.compute_h_hmax_diag(x0 - e, hH_dstb, goal ,include_h0, max_type)
            grad_h_hmax[:, i] = (vp - vm) / (2.0 * eps)

        h0_dstb = hH_dstb[:, 0]
        h_f = np.stack([self.sys_dynm.f(x0, d) for d in h0_dstb], axis=0)
        h_G = np.stack([self.sys_dynm.G(x0, d) for d in h0_dstb], axis=0)
        info["hx_gradhmax"] = grad_h_hmax

        return h_hmax, hH_dstb, grad_h_hmax, h_f, h_G, info

# ---------------- Batching --------------------- #

    def h_function_batch(self, bx, obs_pos):
        q = np.stack([np.cos(bx[..., 2]), np.sin(bx[..., 2])], axis=-1)   # (B, H+1, 2)
        diff = bx[:, :, None, :2] - obs_pos[None]                            # (B, H+1, nh, 2)
        D = np.linalg.norm(diff, axis=-1)                                    # (B, H+1, nh)
        nq = np.einsum("bhij,bhj->bhi", diff / D[..., None], q)             # (B, H+1, nh)
        return -(D - self.R_O + self.delta * nq)

    def h_fun_backup_batch(self, bx, obs_pos):
        q = np.stack([np.cos(bx[..., 2]), np.sin(bx[..., 2])], axis=-1)
        diff = bx[:, :, None, :2] - obs_pos[None]
        D = np.linalg.norm(diff, axis=-1)
        nq = np.einsum("bhij,bhj->bhi", diff / D[..., None], q)
        return -(self.v_max * nq)

    def evaluate_h_traj_batch(self, bTraj, obs_pos):
        """(B, H+1, nx), (H+1, n_obs, 2) -> (B, H+1, nh)."""
        if self.policy_h == "policy_h":
            return self.h_function_batch(bTraj, obs_pos)
        elif self.policy_h == "backup_h":
            return self.h_fun_backup_batch(bTraj, obs_pos)
        elif self.policy_h == "mixed_h":
            bh = self.h_function_batch(bTraj, obs_pos)
            bh[:, -1, :] = self.h_fun_backup_batch(bTraj[:, -1:, :], obs_pos[-1:])[:, 0]
            return bh
        else:
            raise ValueError(f"Unknown policy_h: {self.policy_h}")

    def hmax_batch(self, bh_hist, include_h0, max_type="cubic_spline"):

        B, Hp1, nh = bh_hist.shape
        t_hist = np.linspace(0.0, self.sys_dynm.dt * (Hp1 - 1), Hp1)
        h_values = bh_hist if include_h0 else bh_hist[:, 1:]
        t_values = t_hist if include_h0 else t_hist[1:]

        if max_type != "cubic_spline":
            return np.max(h_values, axis=1)

        out = np.empty((B, nh))
        for b in range(B):
            for j in range(nh):
                out[b, j] = self.max_cubic_spline(t_values, h_values[b, :, j])
        return out