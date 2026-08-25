import numpy as np
from scipy.interpolate import CubicSpline

from Value_function import h_certificate
from Dynamics import sys_dynm_dd
from Control_policy import policy
from Noise_sampler import noise_train_sampler
class h_certificate_batch:
    def __init__(self, dynamic_class: sys_dynm_dd,
                 obs_pos, R_O, hcert_class: h_certificate,
                 policy_h="policy_h", policy_name="proportional_policy",
                 ivp_method="manual_RK4", delta=0.3):

        assert ivp_method in ["manual_RK4"], "ivp_method should be 'manual_RK4' for batch processing"
        self.sys_dynm = dynamic_class
        self.hcert_class = hcert_class
        self.v_max = self.sys_dynm.controller.v_max
        self.om_max = self.sys_dynm.controller.om_max
        self.nx = self.sys_dynm.nx          
        self.nu = self.sys_dynm.nu
        self.nd = self.sys_dynm.nd

        self.obs_pos = np.asarray(obs_pos, dtype=float).reshape(-1, 2)
        self.R_O = np.atleast_1d(np.asarray(R_O, dtype=float))
        assert self.obs_pos.shape[0] == self.R_O.shape[0]
        self.nh = self.R_O.shape[0]         # one barrier per obstacle
        self.delta = delta                  # heading inflation term

        self.policy_h = policy_h            # "policy_h" | "backup_h" | "mixed_h"
        self.policy_name = policy_name
        self.ivp_method = ivp_method

        # The batch path reimplements h; drift against the scalar class would be
        # silent, so pin the parameters that define it.
        assert self.delta == self.hcert_class.delta, "delta mismatch vs hcert_class"
        assert self.policy_h == self.hcert_class.policy_h, "policy_h mismatch vs hcert_class"

    # ---------------- batched RK4 rollout ----------------
    def dynamics_batch(self, bx, bu, bd):
        """f(x,d) + G(x,d) @ u, fused. All args (B, ·) -> (B, nx)."""
        th = bx[:, 2]
        v, om = bu[:, 0], bu[:, 1]
        return np.stack([bd[:, 0] + np.cos(th) * v,
                         bd[:, 1] + np.sin(th) * v,
                         bd[:, 2] + om], axis=1)

    def rollout_batch(self, bx0, bH_dstb):
        """bx0 (nx,) or (B, nx); bH_dstb (B, H, nd) -> (B, H+1, nx)."""
        bx0 = np.asarray(bx0, dtype=float)
        bH_dstb = np.asarray(bH_dstb, dtype=float)
        B, H, nd = bH_dstb.shape
        if nd != self.nd:
            raise ValueError(f"Expected nd={self.nd}, got {nd}")
        if bx0.ndim == 1:
            bx0 = np.broadcast_to(bx0, (B, self.nx))
        if bx0.shape != (B, self.nx):
            raise ValueError(f"bx0 must be ({B}, {self.nx}), got {bx0.shape}")

        dt = self.sys_dynm.dt
        traj = np.empty((B, H + 1, self.nx))
        state = np.array(bx0, dtype=float)
        traj[:, 0] = state
        for k in range(H):
            bd = bH_dstb[:, k]
            rhs = lambda s: self.dynamics_batch(s, self.policy_batch(s), bd)
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt * k1)
            k3 = rhs(state + 0.5 * dt * k2)
            k4 = rhs(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj[:, k + 1] = state
        return traj

    # ---------------- batched barrier ----------------
    def h_function_batch(self, bx):
        """(..., nx) -> (..., nh)."""
        q = np.stack([np.cos(bx[..., 2]), np.sin(bx[..., 2])], axis=-1)
        diff = bx[..., None, :2] - self.obs_pos                   # (..., nh, 2)
        D = np.linalg.norm(diff, axis=-1)                         # (..., nh)
        nq = np.einsum("...ij,...j->...i", diff / D[..., None], q)
        return -(D - self.R_O + self.delta * nq)

    def h_fun_backup_batch(self, bx):
        q = np.stack([np.cos(bx[..., 2]), np.sin(bx[..., 2])], axis=-1)
        diff = bx[..., None, :2] - self.obs_pos
        D = np.linalg.norm(diff, axis=-1)
        nq = np.einsum("...ij,...j->...i", diff / D[..., None], q)
        return -(self.v_max * nq)

    def evaluate_h_traj_batch(self, bTraj):
        """(B, H+1, nx) -> (B, H+1, nh)."""
        if self.policy_h == "policy_h":
            return self.h_function_batch(bTraj)
        elif self.policy_h == "backup_h":
            return self.h_fun_backup_batch(bTraj)
        elif self.policy_h == "mixed_h":
            bh = self.h_function_batch(bTraj)
            bh[..., -1, :] = self.h_fun_backup_batch(bTraj[..., -1, :])
            return bh
        else:
            raise ValueError(f"Unknown policy_h: {self.policy_h}")

    def hmax_batch(self, bh_hist, include_h0=False, max_type="cubic_spline"):
        """(B, H+1, nh) -> (B, nh)."""
        B, Hp1, nh = bh_hist.shape
        t_hist = np.linspace(0.0, self.sys_dynm.dt * (Hp1 - 1), Hp1)
        h_values = bh_hist if include_h0 else bh_hist[:, 1:]
        t_values = t_hist if include_h0 else t_hist[1:]

        if max_type != "cubic_spline":
            return np.max(h_values, axis=1)

        out = np.empty((B, nh))
        for b in range(B):
            for j in range(nh):
                out[b, j] = self.hcert_class.max_cubic_spline(t_values, h_values[b, :, j])
        return out

    # ---------------- public API ----------------
    def compute_h_hmax_diag(self, x0, hH_dstb, include_h0=False, max_type="cubic_spline"):
        """Barrier j evaluated ONLY under hH_dstb[j] -> diagonal, not column-max."""
        bh_hmax = self.hmax_batch(
            self.evaluate_h_traj_batch(self.rollout_batch(x0, hH_dstb)),
            include_h0, max_type)                                 # (nh, nh)
        return np.diagonal(bh_hmax).copy()

    def compute_h_hmax(self, x0, bH_dstb, include_h0=True, max_type="cubic_spline"):
        x0 = self.sys_dynm.chk_x(x0)
        bH_dstb = np.asarray(bH_dstb, dtype=float)
        if bH_dstb.ndim != 3:
            raise ValueError("bH_dstb must have shape (n_samples, horizon, nd)")
        n_samples, horizon, nd = bH_dstb.shape
        if nd != self.nd:
            raise ValueError(f"Expected disturbance dimension {self.nd}, got {nd}")

        bHp1_x = self.rollout_batch(x0, bH_dstb)
        bHp1h_h = self.evaluate_h_traj_batch(bHp1_x)
        bh_hmax = self.hmax_batch(bHp1h_h, include_h0, max_type)

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
        assert hH_dstb.shape == (self.nh, horizon, self.nd)

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

    def get_value_and_grad(self, x0, bH_dstb, include_h0=False,
                           max_type="cubic_spline", eps=1e-5):
        h_hmax, hH_dstb, info = self.compute_h_hmax(x0, bH_dstb, include_h0, max_type)
        nx, nh = self.nx, self.nh

        # All 2*nx perturbations x nh worst-case disturbances in ONE rollout.
        # repeat on states (outer) + tile on dstb (inner) => reshape axes are
        # (perturbation, dstb_index, barrier_index); the diagonal over the last
        # two reproduces compute_h_hmax_diag. Swapping repeat/tile silently
        # gives back the column-max bug.
        E = np.eye(nx) * eps
        pert = np.concatenate([x0 + E, x0 - E], axis=0)           # (2nx, nx)
        gx0 = np.repeat(pert, nh, axis=0)                         # (2nx*nh, nx)
        gd = np.tile(hH_dstb, (2 * nx, 1, 1))                     # (2nx*nh, H, nd)

        gh = self.hmax_batch(
            self.evaluate_h_traj_batch(self.rollout_batch(gx0, gd)),
            include_h0, max_type).reshape(2 * nx, nh, nh)
        gdiag = np.diagonal(gh, axis1=1, axis2=2)                 # (2nx, nh)
        grad_h_hmax = ((gdiag[:nx] - gdiag[nx:]) / (2.0 * eps)).T  # (nh, nx)

        h0_dstb = hH_dstb[:, 0]
        h_f = np.stack([self.sys_dynm.f(x0, d) for d in h0_dstb], axis=0)
        h_G = np.stack([self.sys_dynm.G(x0, d) for d in h0_dstb], axis=0)

        info["hx_gradhmax"] = grad_h_hmax
        return h_hmax, hH_dstb, grad_h_hmax, h_f, h_G, info


if __name__ == "__main__":
    import time

    obs_pos = np.array([[2.0, 2.5], [3.0, 3.5], [1.5, 1.8]])
    R_O = np.array([0.3, 0.2, 0.1]) + 0.03
    POL, DELTA = "backup_policy", 0.0

    pol_clas = policy(obs_pos=obs_pos)
    dynm = sys_dynm_dd(policy_class=pol_clas, dt=0.05)
    cert_s = h_certificate(dynamic_class=dynm, obs_pos=obs_pos, R_O=R_O,
                           policy_h="policy_h", policy_name=POL,
                           ivp_method="manual_RK4", delta=DELTA)
    cert_b = h_certificate_batch(dynamic_class=dynm, obs_pos=obs_pos, R_O=R_O,
                                 hcert_class=cert_s, policy_h="policy_h",
                                 policy_name=POL, ivp_method="manual_RK4", delta=DELTA)
    noise = noise_train_sampler(nd=dynm.nd, rng=np.random.default_rng(42))
    BH_dstb, _ = noise.bangbang_uniform_train(n_samples=50, n_samples_uniform=25,
                                              horizon=30, interval_size=6, scale=0.1)

    X_TEST = [np.array([1.0, 1.5, np.arctan2(1.0, 1.0)]),
              np.array([2.25, 3.0, 0.0]),        # equidistant-ish: argmin can flip
              np.array([0.5, 2.5, 0.0])]

    print("-------- scalar vs batch (exactness) --------")
    for x0 in X_TEST:
        r_s = cert_s.get_value_and_grad(x0, BH_dstb, include_h0=False)
        r_b = cert_b.get_value_and_grad(x0, BH_dstb, include_h0=False)
        diffs = [np.max(np.abs(a - b)) for a, b in zip(r_s[:5], r_b[:5])]
        print(f"x0={x0}  " + "  ".join(
            f"{n}:{d:.2e}" for n, d in zip(["h", "dstb", "grad", "f", "G"], diffs)))

    print("\n-------- timing: where does the cost live --------")
    x0 = np.array([1.0, 1.5, 0.785])
    timings = {}
    for name, c in [("scalar", cert_s), ("batch", cert_b)]:
        for mt in ["cubic_spline", "raw"]:
            c.get_value_and_grad(x0, BH_dstb, include_h0=False, max_type=mt)  # warm-up
            t = time.perf_counter()
            for _ in range(10):
                c.get_value_and_grad(x0, BH_dstb, include_h0=False, max_type=mt)
            ms = 1000 * (time.perf_counter() - t) / 10
            timings[(name, mt)] = ms
            print(f"  {name:7s} {mt:13s} {ms:8.2f} ms")
    print(f"  spline overhead (batch): "
          f"{timings[('batch','cubic_spline')] - timings[('batch','raw')]:.2f} ms"
          f"   speedup batch+raw vs scalar+spline: "
          f"{timings[('scalar','cubic_spline')] / timings[('batch','raw')]:.1f}x")

    print("\n-------- does the spline change the answer? --------")
    for x0 in X_TEST:
        h_sp, _, g_sp, _, _, _ = cert_b.get_value_and_grad(x0, BH_dstb, False, "cubic_spline")
        h_rw, _, g_rw, _, _, _ = cert_b.get_value_and_grad(x0, BH_dstb, False, "raw")
        dh = np.max(np.abs(h_sp - h_rw))
        dg = np.max(np.abs(g_sp - g_rw))
        rel = np.max(np.abs(g_sp - g_rw) / (np.abs(g_sp) + 1e-9))
        print(f"x0={x0}\n"
              f"    |dh|={dh:.2e}  (barrier_inflate=0.03 for scale)\n"
              f"    |dg|={dg:.2e}  rel={rel:.2%}\n"
              f"    g_spline={np.array2string(g_sp, precision=4)}\n"
              f"    g_raw   ={np.array2string(g_rw, precision=4)}")

    print("\n-------- gradient stability under eps (raw only) --------")
    x0 = X_TEST[0]
    for eps in [1e-6, 1e-5, 1e-4, 1e-3]:
        _, _, g, _, _, _ = cert_b.get_value_and_grad(x0, BH_dstb, False, "raw", eps=eps)
        print(f"  eps={eps:.0e}  grad={np.array2string(g, precision=4)}")

    print("\n-------- config sweep --------")
    for pol in ["backup_policy", "proportional_policy", "constant_policy"]:
        for ph in ["policy_h", "backup_h", "mixed_h"]:
            for dl in [0.0, 0.3]:
                for mt in ["cubic_spline", "raw"]:
                    cs = h_certificate(dynamic_class=dynm, obs_pos=obs_pos, R_O=R_O,
                                       policy_h=ph, policy_name=pol,
                                       ivp_method="manual_RK4", delta=dl)
                    cb = h_certificate_batch(dynamic_class=dynm, obs_pos=obs_pos, R_O=R_O,
                                             hcert_class=cs, policy_h=ph, policy_name=pol,
                                             ivp_method="manual_RK4", delta=dl)
                    worst = 0.0
                    for x0 in X_TEST:
                        rs = cs.get_value_and_grad(x0, BH_dstb, False, mt)
                        rb = cb.get_value_and_grad(x0, BH_dstb, False, mt)
                        worst = max(worst, max(np.max(np.abs(a - b)) for a, b in zip(rs[:5], rb[:5])))
                        for k in ["bHp1_x", "bHp1h_h", "bh_hmax", "h_argmax", "hHp1_x", "hHp1h_h"]:
                            worst = max(worst, np.max(np.abs(np.asarray(rs[5][k], float)
                                                             - np.asarray(rb[5][k], float))))
                    flag = "OK " if worst == 0.0 else "FAIL"
                    print(f"  {flag} {pol:20s} {ph:10s} delta={dl}  {mt:13s}  {worst:.2e}")