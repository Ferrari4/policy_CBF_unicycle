import numpy as np
from scipy.interpolate import CubicSpline

from Dynamics import sys_dynm_dd
from Control_policy import policy
from Noise_sampler import noise_train_sampler

class h_certificate:
    def __init__(self, dynamic_class: sys_dynm_dd, obs_pos, R_O,
                 policy_h="policy_h", policy_name="proportional_policy",
                 ivp_method="manual_RK4", delta=0.3):

        self.sys_dynm = dynamic_class
        self.v_max = self.sys_dynm.controller.v_max
        self.om_max = self.sys_dynm.controller.om_max
        self.nx = self.sys_dynm.nx          # state dimension [px, py, th]

        self.obs_pos = np.asarray(obs_pos, dtype=float).reshape(-1, 2)
        self.R_O = np.atleast_1d(np.asarray(R_O, dtype=float))
        assert self.obs_pos.shape[0] == self.R_O.shape[0]
        self.nh = self.R_O.shape[0]         # one barrier per obstacle
        self.delta = delta                  # heading inflation term

        self.policy_h = policy_h            # "policy_h" | "backup_h" | "mixed_h"
        self.policy_name = policy_name
        self.ivp_method = ivp_method

    # Convention: h > 0 unsafe, safe set = {h <= 0}
    def h_function(self, state):
        state = self.sys_dynm.chk_x(state)
        px, py, th = state
        p = np.array([px, py])
        q = np.array([np.cos(th), np.sin(th)])
        h = np.zeros(self.nh)
        for i in range(self.nh):
            diff = p - self.obs_pos[i]
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
        h_b = np.zeros(self.nh)
        for i in range(self.nh):
            diff = p - self.obs_pos[i]
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
            return None

    def compute_h_hmax(self, x0, bH_dstb, include_h0=True, max_type="cubic_spline"):

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
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, policy_name=self.policy_name, method=self.ivp_method)
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

    def compute_h_hmax_from_dstb(self, x0, hH_dstb, include_h0=False, max_type="cubic_spline"):
        x0 = self.sys_dynm.chk_x(x0)
        hh_hmax = []
        for H_dstb in hH_dstb:
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, policy_name=self.policy_name, method=self.ivp_method)
            h_history = self.evaluate_h_trajectory(trajectory)
            t_history = np.linspace(0.0, self.sys_dynm.dt * (h_history.shape[0] - 1), h_history.shape[0])
            h_values = h_history if include_h0 else h_history[1:]
            t_values = t_history if include_h0 else t_history[1:]
            if max_type == "cubic_spline":
                h_max = np.array([self.max_cubic_spline(t_values, h_values[:, j])
                                  for j in range(self.nh)])
            else:
                h_max = np.max(h_values, axis=0)
            hh_hmax.append(h_max)

        h_out = np.max(np.stack(hh_hmax, axis=0), axis=0)

        # print("compute_h_hmax_from_dstb: h_out", h_out)

        return h_out

    def compute_h_hmax_diag(self, x0, hH_dstb, include_h0=False, max_type="cubic_spline"):
        h_out = np.zeros(self.nh)
        for j, H_dstb in enumerate(hH_dstb):
            trajectory = self.sys_dynm.rollout_ivp(x0, H_dstb, policy_name=self.policy_name, method=self.ivp_method)
            h_history = self.evaluate_h_trajectory(trajectory)
            t_history = np.linspace(0.0, self.sys_dynm.dt * (h_history.shape[0] - 1), h_history.shape[0])
            h_values = h_history if include_h0 else h_history[1:]
            t_values = t_history if include_h0 else t_history[1:]
            if max_type == "cubic_spline":
                h_out[j] = self.max_cubic_spline(t_values, h_values[:, j])
            else:
                h_out[j] = np.max(h_values[:, j])
        return h_out

    def get_value_and_grad(self, x0, bH_dstb, include_h0=False, max_type="cubic_spline", eps=1e-5):
        h_hmax, hH_dstb, info = self.compute_h_hmax(x0, bH_dstb, include_h0, max_type)
        grad_h_hmax = np.zeros((self.nh, self.nx))
        for i in range(self.nx):
            e = np.zeros(self.nx)
            e[i] = eps
            vp = self.compute_h_hmax_diag(x0 + e, hH_dstb, include_h0, max_type)
            vm = self.compute_h_hmax_diag(x0 - e, hH_dstb, include_h0, max_type)
            grad_h_hmax[:, i] = (vp - vm) / (2.0 * eps)

        h0_dstb = hH_dstb[:, 0]
        h_f = np.stack([self.sys_dynm.f(x0, d) for d in h0_dstb], axis=0)
        h_G = np.stack([self.sys_dynm.G(x0, d) for d in h0_dstb], axis=0)

        info["hx_gradhmax"] = grad_h_hmax

        return h_hmax, hH_dstb, grad_h_hmax, h_f, h_G, info

if __name__ == "__main__":
    # Single obstacle config from Control_env.py (margin_bar already added)
    obs_pos = np.array([[2.0, 2.5]])
    R_O = np.array([0.3 + 0.03])

    pol_clas = policy()
    dynm = sys_dynm_dd(policy_class=pol_clas)
    cert = h_certificate(dynamic_class=dynm, obs_pos=obs_pos, R_O=R_O,
                         policy_h="policy_h", policy_name="backup_policy")
    noise = noise_train_sampler(nd=dynm.nd, rng=np.random.default_rng(42))

    # Start near the obstacle, heading toward it: interesting h values
    x0 = np.array([1.0, 1.5, np.arctan2(2.5 - 1.5, 2.0 - 1.0)])
    BH_dstb, _ = noise.bangbang_uniform_train(n_samples=10, n_samples_uniform=1, horizon=10, interval_size=3)

    h_x = cert.h_function(state=x0)
    hb_x = cert.h_fun_backup(state=x0)
    print("---------Test for H function--------------")
    print("h  value for individual state", h_x)
    print("hb value for individual state", hb_x)
    for H_dstb in BH_dstb[:2]:
        trajectory = dynm.rollout_ivp(x0, H_dstb, policy_name="backup_policy", method="manual_RK4")
        h_trajectory = cert.evaluate_h_trajectory(trajectory=trajectory)
        print("Trajectory:\n", trajectory)
        print("h along trajectory:\n", h_trajectory)

    h_hmax, hH_dstb, info = cert.compute_h_hmax(x0, BH_dstb, True, "cubic_spline")
    print("Max value of H function: ", h_hmax)
    print("Max disturbance shape: ", hH_dstb.shape)

    print("-------Test: value from worst-case dstb matches---------------------")
    v_from_dstb = cert.compute_h_hmax_from_dstb(x0, hH_dstb, include_h0=True, max_type="cubic_spline")
    print("value from worst dstb :", v_from_dstb)
    print("h_hmax (from compute) :", h_hmax)
    print("match:", np.allclose(v_from_dstb, h_hmax),
          "  max|diff| =", np.max(np.abs(v_from_dstb - h_hmax)))

    print("\n-------- Test: get_value_and_grad --------")
    h_hmax_g, hH_dstb_g, grad, h_f, h_G, info_g = cert.get_value_and_grad(
        x0, BH_dstb, include_h0=False, max_type="cubic_spline")
    print("h_hmax        :", h_hmax_g,          " shape", h_hmax_g.shape)
    print("grad_h_hmax   :\n", grad,            "\n              shape", grad.shape, "(nh, nx)")
    print("h_f           :\n", h_f,             "\n              shape", h_f.shape, "(nh, nx)")
    print("h_G           :\n", h_G,             "\n              shape", h_G.shape, "(nh, nx, nu)")