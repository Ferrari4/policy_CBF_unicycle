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
        self.vel_eps = 1e-4                 # step for finite-difference obstacle velocity

    # ---------------- Moving-obstacle timeline helpers ---------------- #
    # Obstacle trajectory is a known function of time (ObsDyn).  Along a rollout of
    # H steps we need p_O(t_k) and dp_O/dt(t_k) for k = 0..H.  Velocity is taken from
    # ObsDyn if it exposes one, otherwise by central difference of the position.

    def obs_pos_timeline(self, Hp1, t_shift=0.0):
        """(H+1, n_obs, 2) obstacle position at rollout times t_now + t_shift + k*dt."""
        return np.asarray(self.obs_class.rollout_obs(Hp1, t_shift=t_shift), dtype=float)

    def obs_vel_timeline(self, Hp1, t_shift=0.0):
        """(H+1, n_obs, 2) obstacle velocity dp_O/dt at the same rollout times."""
        if hasattr(self.obs_class, "rollout_obs_vel"):
            return np.asarray(self.obs_class.rollout_obs_vel(Hp1, t_shift=t_shift), dtype=float)
        e = self.vel_eps
        return (self.obs_pos_timeline(Hp1, t_shift + e)
                - self.obs_pos_timeline(Hp1, t_shift - e)) / (2.0 * e)

    def obs_timeline(self, Hp1, t_shift=0.0):
        """Convenience: (obs_pos, obs_vel), each (H+1, n_obs, 2)."""
        return self.obs_pos_timeline(Hp1, t_shift), self.obs_vel_timeline(Hp1, t_shift)

    def obs_pos_now(self):
        """(n_obs, 2) obstacle position at the current time."""
        return np.asarray(self.obs_class.pos_now(), dtype=float).reshape(self.nh, 2)

    def obs_vel_now(self):
        """(n_obs, 2) obstacle velocity at the current time."""
        t = self.obs_class.t
        if hasattr(self.obs_class, "vel"):
            return np.asarray(self.obs_class.vel(t), dtype=float).reshape(self.nh, 2)
        e = self.vel_eps
        pp = np.asarray(self.obs_class.pos(t + e), dtype=float).reshape(self.nh, 2)
        pm = np.asarray(self.obs_class.pos(t - e), dtype=float).reshape(self.nh, 2)
        return (pp - pm) / (2.0 * e)

    # Convention: h > 0 unsafe, safe set = {h <= 0}
    # Paper (24):  h = D - R_O + delta * n^T q        (sign flipped here)
    def h_function(self, state, obs_pos=None):
        """obs_pos: (n_obs, 2) obstacle position at the time of `state`.  Defaults to NOW."""
        state = self.sys_dynm.chk_x(state)
        px, py, th = state
        p = np.array([px, py])
        q = np.array([np.cos(th), np.sin(th)])
        if obs_pos is None:
            obs_pos = self.obs_pos_now()
        h = np.zeros(self.nh)
        for i in range(self.nh):
            diff = p - obs_pos[i]
            D = np.linalg.norm(diff)
            n = diff / D
            h[i] = -(D - self.R_O[i] + self.delta * (n @ q))
        return h

    # Paper (27):  h_b = n^T (q v_max - dp_O/dt)      (sign flipped here)
    # h_b <= 0 iff the robot, driving at v_max along its heading, separates from the
    # obstacle faster than the obstacle closes in.  Only meaningful at the end of a
    # backup rollout.
    def h_fun_backup(self, state, obs_pos=None, obs_vel=None):
        """obs_pos, obs_vel: (n_obs, 2) obstacle position / velocity at the time of `state`."""
        state = self.sys_dynm.chk_x(state)
        px, py, th = state
        p = np.array([px, py])
        q = np.array([np.cos(th), np.sin(th)])
        if obs_pos is None:
            obs_pos = self.obs_pos_now()
        if obs_vel is None:
            obs_vel = self.obs_vel_now()
        h_b = np.zeros(self.nh)
        for i in range(self.nh):
            diff = p - obs_pos[i]
            D = np.linalg.norm(diff)
            n = diff / D
            h_b[i] = -(n @ (self.v_max * q - obs_vel[i]))
        return h_b

    def evaluate_h_trajectory(self, trajectory, obs_pos=None, obs_vel=None):
        """trajectory: (H+1, nx).  obs_pos / obs_vel: (H+1, n_obs, 2) obstacle timeline
        aligned with the trajectory samples.  Default: obstacle timeline from NOW."""
        if trajectory.ndim != 2:
            raise ValueError("trajectory must have shape (horizon + 1, nx)")
        if trajectory.shape[1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {trajectory.shape[1]}")

        Hp1 = trajectory.shape[0]
        if obs_pos is None:
            obs_pos = self.obs_pos_timeline(Hp1)
        if obs_vel is None and self.policy_h in ("backup_h", "mixed_h"):
            obs_vel = self.obs_vel_timeline(Hp1)

        if self.policy_h == "policy_h":
            return np.stack([self.h_function(x, obs_pos[k])
                             for k, x in enumerate(trajectory)], axis=0)

        elif self.policy_h == "backup_h":
            return np.stack([self.h_fun_backup(x, obs_pos[k], obs_vel[k])
                             for k, x in enumerate(trajectory)], axis=0)

        elif self.policy_h == "mixed_h":
            # h on all states, h_b replacing h at the terminal state
            # h_hist = [self.h_function(x, obs_pos[k]) for k, x in enumerate(trajectory[:-1])]
            # h_hist.append(self.h_fun_backup(trajectory[-1], obs_pos[-1], obs_vel[-1]))
            # return np.stack(h_hist, axis=0)
            h_hist = [self.h_function(x, obs_pos[k]) for k, x in enumerate(trajectory)]
            h_hist.append(self.h_fun_backup(trajectory[-1], obs_pos[-1], obs_vel[-1]))
            return np.stack(h_hist, axis=0) 
        
        else:
            raise ValueError(f"Unknown policy_h: {self.policy_h}")

    def compute_h_hmax(self, x0, bH_dstb, goal, include_h0, max_type="cubic_spline",
                       obs_pos=None, obs_vel=None):

        x0 = self.sys_dynm.chk_x(x0)
        bH_dstb = np.asarray(bH_dstb, dtype=float)
        if bH_dstb.ndim != 3:
            raise ValueError("bH_dstb must have shape " "(n_samples, horizon, nd)")
        n_samples, horizon, nd = bH_dstb.shape
        if nd != self.sys_dynm.nd:
            raise ValueError(f"Expected disturbance dimension " f"{self.sys_dynm.nd}, got {nd}")

        # Obstacle timeline over the rollout: the same for every rollout in this control step.
        if obs_pos is None or obs_vel is None:
            obs_pos, obs_vel = self.obs_timeline(horizon + 1)

        sample_hmax = []
        trajectories = []
        barrier_histories = []

        for H_dstb in bH_dstb:
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, goal=goal ,policy_name=self.policy_name)
            h_history = self.evaluate_h_trajectory(trajectory, obs_pos, obs_vel)
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

    def compute_h_hmax_diag(self, x0, hH_dstb, goal, include_h0, max_type="cubic_spline",
                            obs_pos=None, obs_vel=None):
        if obs_pos is None or obs_vel is None:
            obs_pos, obs_vel = self.obs_timeline(np.asarray(hH_dstb).shape[1] + 1)
        h_out = np.zeros(self.nh)
        for j, H_dstb in enumerate(hH_dstb):
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, goal=goal,policy_name=self.policy_name)
            h_history = self.evaluate_h_trajectory(trajectory, obs_pos, obs_vel)
            t_history = np.linspace(0.0, self.sys_dynm.dt * (h_history.shape[0] - 1), h_history.shape[0])
            h_values = h_history if include_h0 else h_history[1:]
            t_values = t_history if include_h0 else t_history[1:]
            if max_type == "cubic_spline":
                h_out[j] = self.max_cubic_spline(t_values, h_values[:, j])
            else:
                h_out[j] = np.max(h_values[:, j])
        return h_out

    def get_value_and_grad(self, x0, bH_dstb, goal ,include_h0, max_type="cubic_spline", eps=1e-5):
        H = np.asarray(bH_dstb).shape[1]
        obs_pos, obs_vel = self.obs_timeline(H + 1)

        h_hmax, hH_dstb, info = self.compute_h_hmax(x0, bH_dstb, goal, include_h0, max_type,
                                                    obs_pos, obs_vel)
        grad_h_hmax = np.zeros((self.nh, self.nx))
        for i in range(self.nx):
            e = np.zeros(self.nx)
            e[i] = eps
            vp = self.compute_h_hmax_diag(x0 + e, hH_dstb, goal, include_h0, max_type, obs_pos, obs_vel)
            vm = self.compute_h_hmax_diag(x0 - e, hH_dstb, goal ,include_h0, max_type, obs_pos, obs_vel)
            grad_h_hmax[:, i] = (vp - vm) / (2.0 * eps)

        # dV/dt: explicit time dependence through the moving obstacle (paper (25)/(28), dh/dt).
        # The worst-case trajectories do not depend on obstacle time, so re-evaluate h on the
        # SAME trajectories with the obstacle timeline shifted by eps.  Zero for a static obstacle.
        obs_pos_s, obs_vel_s = self.obs_timeline(H + 1, t_shift=eps)
        dV_dt = np.zeros(self.nh)
        for j in range(self.nh):
            h_hist_s = self.evaluate_h_trajectory(info["hHp1_x"][j], obs_pos_s, obs_vel_s)
            t_hist = np.linspace(0.0, self.sys_dynm.dt * (h_hist_s.shape[0] - 1), h_hist_s.shape[0])
            h_vals = h_hist_s if include_h0 else h_hist_s[1:]
            t_vals = t_hist if include_h0 else t_hist[1:]
            if max_type == "cubic_spline":
                h_s = self.max_cubic_spline(t_vals, h_vals[:, j])
            else:
                h_s = np.max(h_vals[:, j])
            dV_dt[j] = (h_s - h_hmax[j]) / eps

        h0_dstb = hH_dstb[:, 0]
        h_f = np.stack([self.sys_dynm.f(x0, d) for d in h0_dstb], axis=0)
        h_G = np.stack([self.sys_dynm.G(x0, d) for d in h0_dstb], axis=0)
        info["hx_gradhmax"] = grad_h_hmax
        info["dV_dt"] = dV_dt

        return h_hmax, hH_dstb, grad_h_hmax, h_f, h_G, info

# ---------------- Batching --------------------- #

    # bx: (B, H+1, nx) states; obs_pos / obs_vel: (H+1, n_obs, 2) obstacle timeline aligned
    # with the H+1 rollout samples (broadcast over the batch B).

    def h_function_batch(self, bx, obs_pos):
        """Paper (24), sign flipped: h = -(D - R_O + delta n^T q)."""
        q = np.stack([np.cos(bx[..., 2]), np.sin(bx[..., 2])], axis=-1)   # (B, H+1, 2)
        diff = bx[:, :, None, :2] - obs_pos[None]                            # (B, H+1, nh, 2)
        D = np.linalg.norm(diff, axis=-1)                                    # (B, H+1, nh)
        nq = np.einsum("bhij,bhj->bhi", diff / D[..., None], q)             # (B, H+1, nh)
        return -(D - self.R_O + self.delta * nq)

    def h_fun_backup_batch(self, bx, obs_pos, obs_vel=None):
        """Paper (27), sign flipped: h_b = -( n^T (q v_max - dp_O/dt) ).
        obs_vel=None (or zeros) recovers the static-obstacle certificate."""
        q = np.stack([np.cos(bx[..., 2]), np.sin(bx[..., 2])], axis=-1)   # (B, H+1, 2)
        diff = bx[:, :, None, :2] - obs_pos[None]                            # (B, H+1, nh, 2)
        D = np.linalg.norm(diff, axis=-1)                                    # (B, H+1, nh)
        n = diff / D[..., None]                                              # (B, H+1, nh, 2)
        nq = np.einsum("bhij,bhj->bhi", n, q)                                # (B, H+1, nh)
        if obs_vel is None:
            return -(self.v_max * nq)
        nv = np.einsum("bhij,hij->bhi", n, obs_vel)                          # (B, H+1, nh) n^T dp_O/dt
        return -(self.v_max * nq - nv)

    def evaluate_h_traj_batch(self, bTraj, obs_pos, obs_vel=None):
        """(B, H+1, nx), (H+1, n_obs, 2), (H+1, n_obs, 2) -> (B, H+1, nh)."""
        if obs_vel is None and self.policy_h in ("backup_h", "mixed_h"):
            obs_vel = self.obs_vel_timeline(bTraj.shape[1])
        if self.policy_h == "policy_h":
            return self.h_function_batch(bTraj, obs_pos)
        elif self.policy_h == "backup_h":
            return self.h_fun_backup_batch(bTraj, obs_pos, obs_vel)
        elif self.policy_h == "mixed_h":
            bh = self.h_function_batch(bTraj, obs_pos)
            bh[:, -1, :] = self.h_fun_backup_batch(bTraj[:, -1:, :], obs_pos[-1:], obs_vel[-1:])[:, 0]
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