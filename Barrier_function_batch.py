import numpy as np
from scipy.interpolate import CubicSpline

from Barrier_function import h_certificate
from Get_obstacles import ObsDyn
from Dynamics import sys_dynm_dd

class h_certificate_batch:
    def __init__(self, dynamic_class: sys_dynm_dd, hcert_class: h_certificate, 
                 obs_class: ObsDyn, terminate_on_hb=False):

        self.sys_dynm = dynamic_class
        self.hcert_class = hcert_class
        assert self.sys_dynm.process == "batch", \
        "dynamic_class must use process='batch'"

        self.v_max = self.sys_dynm.controller.v_max
        self.om_max = self.sys_dynm.controller.om_max
        self.nx = self.sys_dynm.nx          
        self.nu = self.sys_dynm.nu
        self.nd = self.sys_dynm.nd
        self.obs_class = obs_class
        self.R_O = np.atleast_1d(np.asarray(obs_class.R_O, dtype=float))
        self.nh = self.R_O.shape[0]             # one barrier per obstacle
        self.delta = self.hcert_class.delta     # heading inflation term
        self.policy_h = self.hcert_class.policy_h    
        self.policy_name = self.hcert_class.policy_name
        self.terminate_on_hb = terminate_on_hb          # new ctor arg, default False
        self.stop_fn = (
            (lambda s, d: self.hcert_class.h_fun_backup_batch(
                np.atleast_2d(s)[:, None, :],            # (B, 1, nx)
                self.obs_class.pos(self.obs_class.t)     # (1, n_obs, 2) obstacle NOW
            )[:, 0].max(axis=1) <= 0.0)
            if terminate_on_hb else None
        )


    def _eval_h_frozen(self, bTraj, b_stop, obs_pos):
        """h along the rollout, but once a row has been frozen by stop_fn (state no longer
        evolves) its h is held at the stop sample instead of being re-evaluated against an
        obstacle that keeps moving.  No-op for a static obstacle."""
        bh = self.hcert_class.evaluate_h_traj_batch(bTraj, obs_pos)          # (B, H+1, nh)
        if b_stop is not None:
            B, Hp1, _ = bh.shape
            k = np.arange(Hp1)[None, :]                                       # (1, H+1)
            stop = np.where(b_stop < 0, Hp1 - 1, b_stop)[:, None]            # (B, 1); -1 = never stopped
            frozen = k > stop                                                 # (B, H+1)
            held = bh[np.arange(B), stop[:, 0]][:, None, :]                   # (B, 1, nh) value at stop
            bh = np.where(frozen[..., None], held, bh)
        return bh

    def compute_h_hmax_diag(self, x0, hH_dstb, goal, include_h0, obs_pos, max_type="cubic_spline"):
        bx0 = np.tile(x0, (self.nh, 1))
        bTraj, b_stop = self.sys_dynm.rollout_ivp(
            bx0, np.transpose(hH_dstb, (1, 0, 2)), goal=goal,
            policy_name=self.policy_name, stop_fn=self.stop_fn, return_stop=True)
        bh_hmax = self.hcert_class.hmax_batch(
            self._eval_h_frozen(bTraj.transpose(1, 0, 2), b_stop, obs_pos),
            include_h0,
            max_type
        )

        return np.diagonal(bh_hmax).copy()

    def compute_h_hmax(self, x0, bH_dstb, goal, include_h0, obs_pos, max_type="cubic_spline"):
        x0 = self.sys_dynm.chk_x(x0)
        bH_dstb = np.asarray(bH_dstb, dtype=float)
        if bH_dstb.ndim != 3:
            raise ValueError("bH_dstb must have shape (n_samples, horizon, nd)")
        n_samples, horizon, nd = bH_dstb.shape
        if nd != self.nd:
            raise ValueError(f"Expected disturbance dimension {self.nd}, got {nd}")

        bx0 = np.tile(x0, (n_samples, 1))
        bHp1_x, b_stop = self.sys_dynm.rollout_ivp(
        bx0, np.transpose(bH_dstb, (1, 0, 2)), goal=goal,
        policy_name=self.policy_name, stop_fn=self.stop_fn, return_stop=True)
        bHp1_x = bHp1_x.transpose(1, 0, 2)
        bHp1h_h = self._eval_h_frozen(bHp1_x, b_stop, obs_pos)
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

        info["b_stop_step"] = b_stop                     
        info["h_stop_step"] = b_stop[h_argmax] if b_stop is not None else None   

        return h_hmax, hH_dstb, info

    def get_value_and_grad(self, x0, bH_dstb, goal, include_h0,
                           max_type="cubic_spline", eps=1e-5):
        # Obstacle timeline over the rollout: same for every rollout in this control step.
        H = np.asarray(bH_dstb).shape[1]
        obs_pos = self.obs_class.rollout_obs(H + 1)                          # (H+1, n_obs, 2)

        h_hmax, hH_dstb, info = self.compute_h_hmax(x0, bH_dstb, goal, include_h0, obs_pos, max_type)
        nx, nh = self.nx, self.nh
        E = np.eye(nx) * eps
        pert = np.concatenate([x0 + E, x0 - E], axis=0)           # (2nx, nx)
        gx0 = np.repeat(pert, nh, axis=0)                         # (2nx*nh, nx)
        gd = np.tile(hH_dstb, (2 * nx, 1, 1))                     # (2nx*nh, H, nd)
        gTraj, g_stop = self.sys_dynm.rollout_ivp(
            gx0, np.transpose(gd, (1, 0, 2)), goal=goal,
            policy_name=self.policy_name, stop_fn=self.stop_fn, return_stop=True)
        gh = self.hcert_class.hmax_batch(
            self._eval_h_frozen(gTraj.transpose(1, 0, 2), g_stop, obs_pos),
            include_h0,
            max_type
        ).reshape(2 * nx, nh, nh)
        gdiag = np.diagonal(gh, axis1=1, axis2=2)                 # (2nx, nh)
        grad_h_hmax = ((gdiag[:nx] - gdiag[nx:]) / (2.0 * eps)).T  # (nh, nx)

        # dV/dt: explicit time dependence through the moving obstacle.
        # The trajectories themselves do not depend on obstacle time (backup policy reads
        # the current position), so re-evaluate h on the SAME worst-case trajectories with
        # the obstacle timeline shifted by eps.  Identically zero for a static obstacle.
        h_dt = self.hcert_class.hmax_batch(
            self._eval_h_frozen(
                info["hHp1_x"],                                             # (nh, H+1, nx)
                info["h_stop_step"],
                self.obs_class.rollout_obs(H + 1, t_shift=eps)
            ),
            include_h0,
            max_type
        )                                                                   # (nh, nh)
        dV_dt = (np.diagonal(h_dt) - h_hmax) / eps                          # (nh,)

        h0_dstb = hH_dstb[:, 0]
        h_f = np.stack([self.sys_dynm.f(x0, d) for d in h0_dstb], axis=0)
        h_G = np.stack([self.sys_dynm.G(x0, d) for d in h0_dstb], axis=0)

        info["hx_gradhmax"] = grad_h_hmax
        info["dV_dt"] = dV_dt
        return h_hmax, hH_dstb, grad_h_hmax, h_f, h_G, info
