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
  

    def compute_h_hmax_diag(self, x0, hH_dstb, goal, include_h0, max_type="cubic_spline"):
        bx0 = np.tile(x0, (self.nh, 1))
        bh_hmax = self.hcert_class.hmax_batch(
            self.hcert_class.evaluate_h_traj_batch(
                self.sys_dynm.rollout_ivp(
                    bx0,
                    np.transpose(hH_dstb, (1, 0, 2)),
                    goal=goal,
                    policy_name=self.policy_name
                ).transpose(1, 0, 2)
            ),
            include_h0,
            max_type
        )

        return np.diagonal(bh_hmax).copy()

    def compute_h_hmax(self, x0, bH_dstb, goal, include_h0, max_type="cubic_spline"):
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
            goal=goal,
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

    def get_value_and_grad(self, x0, bH_dstb, goal, include_h0,
                           max_type="cubic_spline", eps=1e-5):
        h_hmax, hH_dstb, info = self.compute_h_hmax(x0, bH_dstb, goal, include_h0, max_type)
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
                    goal=goal,
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
