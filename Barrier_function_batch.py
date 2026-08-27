import numpy as np
from scipy.interpolate import CubicSpline

from Barrier_function import h_certificate
from Dynamics import sys_dynm_dd

class h_certificate_batch:
    def __init__(self, dynamic_class: sys_dynm_dd, hcert_class: h_certificate, obs_pos, R_O):

        self.sys_dynm = dynamic_class
        self.hcert_class = hcert_class
        assert self.sys_dynm.process == "batch", \
        "dynamic_class must use process='batch'"

        self.v_max = self.sys_dynm.controller.v_max
        self.om_max = self.sys_dynm.controller.om_max
        self.nx = self.sys_dynm.nx          
        self.nu = self.sys_dynm.nu
        self.nd = self.sys_dynm.nd
        self.obs_pos = np.asarray(obs_pos, dtype=float).reshape(-1, 2)
        self.R_O = np.atleast_1d(np.asarray(R_O, dtype=float))
        assert self.obs_pos.shape[0] == self.R_O.shape[0]
        self.nh = self.R_O.shape[0]             # one barrier per obstacle
        self.delta = self.hcert_class.delta     # heading inflation term
        self.policy_h = self.hcert_class.policy_h    
        self.policy_name = self.hcert_class.policy_name
  

    def compute_h_hmax_diag(self, x0, hH_dstb, include_h0=False, max_type="cubic_spline"):
        bx0 = np.tile(x0, (self.nh, 1))
        bh_hmax = self.hcert_class.hmax_batch(
            self.hcert_class.evaluate_h_traj_batch(
                self.sys_dynm.rollout_ivp(
                    bx0,
                    np.transpose(hH_dstb, (1, 0, 2)),
                    policy_name=self.policy_name
                ).transpose(1, 0, 2)
            ),
            include_h0,
            max_type
        )

        return np.diagonal(bh_hmax).copy()

    def compute_h_hmax(self, x0, bH_dstb, include_h0=True, max_type="cubic_spline"):
        x0 = self.sys_dynm.chk_x(x0)
        bH_dstb = np.asarray(bH_dstb, dtype=float)
        if bH_dstb.ndim != 3:
            raise ValueError("bH_dstb must have shape (n_samples, horizon, nd)")
        n_samples, horizon, nd = bH_dstb.shape
        if nd != self.nd:
            raise ValueError(f"Expected disturbance dimension {self.nd}, got {nd}")

        bx0 = np.tile(x0, (n_samples, 1))
        bHp1_x = self.sys_dynm.rollout_ivp(
            bx0,
            np.transpose(bH_dstb, (1, 0, 2)),
            policy_name=self.policy_name
        ).transpose(1, 0, 2)

        bHp1h_h = self.hcert_class.evaluate_h_traj_batch(bHp1_x)
        bh_hmax = self.hcert_class.hmax_batch(
            bHp1h_h,
            include_h0,
            max_type
        )

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
        E = np.eye(nx) * eps
        pert = np.concatenate([x0 + E, x0 - E], axis=0)           # (2nx, nx)
        gx0 = np.repeat(pert, nh, axis=0)                         # (2nx*nh, nx)
        gd = np.tile(hH_dstb, (2 * nx, 1, 1))                     # (2nx*nh, H, nd)

        gh = self.hcert_class.hmax_batch(
            self.hcert_class.evaluate_h_traj_batch(
                self.sys_dynm.rollout_ivp(
                    gx0,
                    np.transpose(gd, (1, 0, 2)),
                    policy_name=self.policy_name
                ).transpose(1, 0, 2)
            ),
            include_h0,
            max_type
        ).reshape(2 * nx, nh, nh)
        gdiag = np.diagonal(gh, axis1=1, axis2=2)                 # (2nx, nh)
        grad_h_hmax = ((gdiag[:nx] - gdiag[nx:]) / (2.0 * eps)).T  # (nh, nx)

        h0_dstb = hH_dstb[:, 0]
        h_f = np.stack([self.sys_dynm.f(x0, d) for d in h0_dstb], axis=0)
        h_G = np.stack([self.sys_dynm.G(x0, d) for d in h0_dstb], axis=0)

        info["hx_gradhmax"] = grad_h_hmax
        return h_hmax, hH_dstb, grad_h_hmax, h_f, h_G, info

if __name__ == "__main__":

    from Control_policy import policy
    from Noise_sampler import noise_train_sampler
    import time

    # ---------------------------------------------------------
    # Problem setup
    # ---------------------------------------------------------
    obs_pos = np.array([
        [2.0, 2.5],
        [3.0, 3.5],
        [1.5, 1.8]
    ])

    R_O = np.array([0.3, 0.2, 0.1]) + 0.03

    x0 = np.array([
        1.0,
        1.5,
        np.arctan2(1.0, 1.0)
    ])

    policy_name = "proportional_policy"
    policy_h = "backup_h"

    dt = 0.05

    # ---------------------------------------------------------
    # SINGLE setup
    # ---------------------------------------------------------
    policy_single = policy(
        obs_pos=obs_pos,
        process="single"
    )

    dynm_single = sys_dynm_dd(
        policy_class=policy_single,
        dt=dt,
        ivp_method="manual_RK4",
        process="single"
    )

    cert_single = h_certificate(
        dynamic_class=dynm_single,
        obs_pos=obs_pos,
        R_O=R_O,
        policy_h=policy_h,
        policy_name=policy_name,
        delta=0.0
    )

    # ---------------------------------------------------------
    # BATCH setup
    # ---------------------------------------------------------
    policy_batch = policy(
        obs_pos=obs_pos,
        process="batch"
    )

    dynm_batch = sys_dynm_dd(
        policy_class=policy_batch,
        dt=dt,
        ivp_method="manual_RK4",
        process="batch"
    )

    cert_batch = h_certificate_batch(
        dynamic_class=dynm_batch,
        hcert_class=cert_single,
        obs_pos=obs_pos,
        R_O=R_O
    )

    # ---------------------------------------------------------
    # Generate identical disturbance samples
    # ---------------------------------------------------------
    noise = noise_train_sampler(
        nd=dynm_single.nd,
        rng=np.random.default_rng(42)
    )

    bH_dstb, _ = noise.bangbang_uniform_train(
        n_samples=50,
        n_samples_uniform=25,
        horizon=30,
        interval_size=6,
        scale=0.1
    )

    print("bH_dstb shape:", bH_dstb.shape)

    # ---------------------------------------------------------
    # SINGLE
    # ---------------------------------------------------------
    t0 = time.perf_counter()

    result_single = cert_single.get_value_and_grad(
        x0,
        bH_dstb,
        include_h0=True,
        max_type="raw"
    )

    time_single = time.perf_counter() - t0

    # ---------------------------------------------------------
    # BATCH
    # ---------------------------------------------------------
    t0 = time.perf_counter()

    result_batch = cert_batch.get_value_and_grad(
        x0,
        bH_dstb,
        include_h0=True,
        max_type="raw"
    )

    time_batch = time.perf_counter() - t0

    # ---------------------------------------------------------
    # Unpack
    # ---------------------------------------------------------
    h_s, d_s, grad_s, hf_s, hG_s, info_s = result_single
    h_b, d_b, grad_b, hf_b, hG_b, info_b = result_batch

    # ---------------------------------------------------------
    # Compare outputs
    # ---------------------------------------------------------
    print("\n========== SINGLE vs BATCH ==========")

    print("\nh_hmax")
    print("single:", h_s)
    print("batch :", h_b)
    print("max error:", np.max(np.abs(h_s - h_b)))

    print("\nWorst-case disturbance")
    print("max error:", np.max(np.abs(d_s - d_b)))

    print("\nGradient")
    print("single:\n", grad_s)
    print("batch:\n", grad_b)
    print("max error:", np.max(np.abs(grad_s - grad_b)))

    print("\nh_f")
    print("max error:", np.max(np.abs(hf_s - hf_b)))

    print("\nh_G")
    print("max error:", np.max(np.abs(hG_s - hG_b)))

    # ---------------------------------------------------------
    # Compare internal data too
    # ---------------------------------------------------------
    print("\n========== INTERNAL DATA ==========")

    keys = [
        "bh_hmax",
        "bHp1_x",
        "bHp1h_h",
        "hHp1_x",
        "hHp1h_h",
        "h_argmax"
    ]

    for key in keys:

        a = np.asarray(info_s[key])
        b = np.asarray(info_b[key])

        if np.issubdtype(a.dtype, np.number):
            error = np.max(np.abs(a - b))
        else:
            error = np.array_equal(a, b)

        print(f"{key:12s}: {error}")

    # ---------------------------------------------------------
    # Overall accuracy test
    # ---------------------------------------------------------
    atol = 1e-10

    passed = (
        np.allclose(h_s, h_b, atol=atol, rtol=0)
        and np.allclose(d_s, d_b, atol=atol, rtol=0)
        and np.allclose(grad_s, grad_b, atol=atol, rtol=0)
        and np.allclose(hf_s, hf_b, atol=atol, rtol=0)
        and np.allclose(hG_s, hG_b, atol=atol, rtol=0)
    )

    print("\n========== RESULT ==========")

    print("Accuracy test:", "PASS" if passed else "FAIL")

    print(f"single time : {time_single * 1000:.3f} ms")
    print(f"batch time  : {time_batch * 1000:.3f} ms")

    if time_batch > 0:
        print(f"speedup     : {time_single / time_batch:.2f}x")