import numpy as np

from Dynamics import sys_dynm_dd
from Lyapunov_function import v_certificate

class v_certificate_batch:
    def __init__(self ,dynamic_class: sys_dynm_dd, 
                 vfun_class: v_certificate):
        self.sys_dynm = dynamic_class
        self.vfun_class = vfun_class
        self.nx = self.sys_dynm.nx
        self.policy_name = self.vfun_class.policy_name
        self.k = self.vfun_class.k
        self.nV = self.vfun_class.nV

    def vmax_batch(self, bh_hist, include_v0):
        B, Vp1, nh = bh_hist.shape
        t_hist = np.linspace(0.0, self.sys_dynm.dt * (Vp1 - 1), Vp1)
        v_values = bh_hist if include_v0 else bh_hist[:, 1:]
        t_values = t_hist if include_v0 else t_hist[1:]
        return np.max(v_values, axis=1)

    def compute_v_vmax(self, x0, bH_dstb, goal, include_v0):
        x0 = self.sys_dynm.chk_x(x0)
        bH_dstb = np.asarray(bH_dstb, dtype=float)
        if bH_dstb.ndim != 3:
            raise ValueError("bH_dstb must have shape (n_samples, horizon, nd)")
        n_samples, horizon, nd = bH_dstb.shape
        if nd != self.sys_dynm.nd:
            raise ValueError(f"Expected disturbance dimension " f"{self.sys_dynm.nd}, got {nd}")

        bx0 = np.tile(x0, (n_samples, 1))
        bHp1_x = self.sys_dynm.rollout_ivp(
            bx0,
            np.transpose(bH_dstb, (1, 0, 2)),
            goal=goal,
            policy_name=self.policy_name
        ).transpose(1, 0, 2)

        bHp1h_h = self.vfun_class.clf_certificate(bHp1_x, goal)
        bh_hmax = self.vmax_batch(bHp1h_h,include_v0)
  
        assert bHp1_x.shape == (n_samples, horizon + 1, self.nx)
        assert bHp1h_h.shape == (n_samples, horizon + 1, self.nV)
        assert bh_hmax.shape == (n_samples, self.nV)

        v_argmax = np.argmax(bh_hmax, axis=0)
        barrier_indices = np.arange(self.nV)
        v_hmax = bh_hmax[v_argmax, barrier_indices]
        vH_dstb = bH_dstb[v_argmax]
        vHp1_x = bHp1_x[v_argmax]
        vHp1h_h = bHp1h_h[v_argmax]

        assert v_hmax.shape == (self.nV,)
        assert vH_dstb.shape == (self.nV, horizon, self.sys_dynm.nd)
        
        info = {
            "v_argmax": v_argmax,
            "bv_vmax": bh_hmax,
            "vHp1_x": vHp1_x,
            "vHp1v_v": vHp1h_h,
            "bHp1_x": bHp1_x,
            "bHp1v_v": bHp1h_h,
            "v_vmax": v_hmax,
        }

        return v_hmax, vH_dstb, info

    def get_value_and_grad(self, x0, bH_dstb, goal, include_v0, eps=1e-5):
        v_vmax, vH_dstb, info = self.compute_v_vmax(x0, bH_dstb, goal, include_v0)
        nx, nV = self.nx, self.nV
        E    = np.eye(nx) * eps
        pert = np.concatenate([x0 + E, x0 - E], axis=0)          # (2nx, nx)
        gx0  = np.repeat(pert, nV, axis=0)                        # (2nx*nV, nx)
        gd   = np.tile(vH_dstb, (2 * nx, 1, 1))                   # (2nx*nV, H, nd)
        gTraj = self.sys_dynm.rollout_ivp(                        # <- the missing rollout
            gx0,
            np.transpose(gd, (1, 0, 2)),
            goal=goal,
            policy_name=self.policy_name
        ).transpose(1, 0, 2)                                                # (2nx*nV, H+1, nx)
        gv = self.vmax_batch(
            self.vfun_class.clf_certificate(gTraj, goal),                   # (2nx*nV, H+1, nV)
            include_v0
        ).reshape(2 * nx, nV, nV)                                           # [perturbation][sequence][value]
        gdiag       = np.diagonal(gv, axis1=1, axis2=2)                     # (2nx, nV)
        grad_v_vmax = ((gdiag[:nx] - gdiag[nx:]) / (2.0 * eps)).T           # (nV, nx)
        v0_dstb = vH_dstb[:, 0]                                             # (nV, nd)  disturbance at t=0
        v_f = np.stack([self.sys_dynm.f(x0, d) for d in v0_dstb], axis=0)   # (nV, nx)
        v_G = np.stack([self.sys_dynm.G(x0, d) for d in v0_dstb], axis=0)   # (nV, nx, nu)
        info["vx_gradvmax"] = grad_v_vmax

        return v_vmax, vH_dstb, grad_v_vmax, v_f, v_G, info

    # ------------------------------------------------------------------
    # Goal-sampled policy CLF (approach A): mean over goal samples of the
    # per-goal rollout maximum.  Zero disturbance in this path.
    # ------------------------------------------------------------------
    def _wk_goals(self, bx0, bgoals, horizon, include_v0, k_cert):
        """One rollout per row: state bx0[k] toward goal bgoals[k].
        Returns per-rollout max of V, shape (B,), plus trajectories."""
        B = bx0.shape[0]
        H_dstb = np.zeros((horizon, B, self.sys_dynm.nd))               # time-first
        bHp1_x = self.sys_dynm.rollout_ivp(bx0, H_dstb, goal=bgoals,
                                           policy_name=self.policy_name
                                           ).transpose(1, 0, 2)          # (B, H+1, nx)
        bHp1v_v = self.vfun_class.clf_certificate(bHp1_x, bgoals[:, None, :], k=k_cert)
        assert bHp1v_v.shape == (B, horizon + 1, self.nV)
        bv = self.vmax_batch(bHp1v_v, include_v0)                        # (B, nV)
        return bv[:, 0], bHp1_x, bHp1v_v

    # def get_value_and_grad_goals(self, x0, goals, horizon, include_v0,
    #                              k_cert=0.0, eps=1e-5):
    #     x0    = self.sys_dynm.chk_x(x0)
    #     goals = np.asarray(goals, dtype=float).reshape(-1, 2)          # (K, 2)
    #     K, nx = goals.shape[0], self.nx
    #     # value  W_A = mean_k max_t V(x_k(t), g_k)
    #     W_k, bHp1_x, bHp1v_v = self._wk_goals(np.tile(x0, (K, 1)), goals,
    #                                           horizon, include_v0, k_cert)

    #     W_A = W_k.mean()

    #     # floor  mean_k 0.5|p-g_k|^2 = 0.5|p-ghat|^2 + c
    #     g_hat = goals.mean(axis=0)
    #     c = 0.5 * np.mean(np.sum((goals - g_hat) ** 2, axis=1))

    #     # gradient: central differences of W_A  (envelope thm per rollout)
    #     E    = np.eye(nx) * eps
    #     pert = np.concatenate([x0 + E, x0 - E], axis=0)               # (2nx, nx)
    #     gx0  = np.repeat(pert, K, axis=0)                              # (2nx*K, nx)  [pert][goal]
    #     gg   = np.tile(goals, (2 * nx, 1))                             # (2nx*K, 2)
    #     gW, _, _ = self._wk_goals(gx0, gg, horizon, include_v0, k_cert)
    #     gW   = gW.reshape(2 * nx, K).mean(axis=1)                      # (2nx,)
    #     grad_W = (gW[:nx] - gW[nx:]) / (2.0 * eps)                     # (nx,)

    #     d0  = np.zeros(self.sys_dynm.nd)
    #     v_f = self.sys_dynm.f(x0, d0)[None]                            # (1, nx)
    #     v_G = self.sys_dynm.G(x0, d0)[None]                            # (1, nx, nu)
    #     vH_dstb = np.zeros((1, horizon, self.sys_dynm.nd))

    #     info = {"W_k": W_k, "W_A": W_A, "c_floor": c, "g_hat": g_hat,
    #             "bHp1_x": bHp1_x, "bHp1v_v": bHp1v_v}
    #     return np.array([W_A]), vH_dstb, grad_W[None], v_f, v_G, info

    def get_value_and_grad_goals(self, x0, goals, horizon, include_v0, k_cert=0.0, eps=1e-5):
        x0 = self.sys_dynm.chk_x(x0)
        goals = np.asarray(goals, dtype=float).reshape(-1, 2)
        K, nx = goals.shape[0], self.nx

        # Rollout each candidate goal
        W_k, bHp1_x, bHp1v_v = self._wk_goals(np.tile(x0, (K, 1)), goals, horizon, include_v0, k_cert)

        # Outer max over goal hypotheses
        k_star = np.argmax(W_k)
        W_A = W_k[k_star]
        winner_goal = goals[k_star]

        # Mean goal only for diagnostics
        g_hat = goals.mean(axis=0)
        c = 0.5 * np.mean(np.sum((goals - g_hat) ** 2, axis=1))

        # Gradient of winning goal branch
        E = np.eye(nx) * eps
        pert = np.concatenate([x0 + E, x0 - E], axis=0)                 # (2nx, nx)
        gg = np.tile(winner_goal[None, :], (2 * nx, 1))                  # (2nx, 2)
        gW, _, _ = self._wk_goals(pert, gg, horizon, include_v0, k_cert)
        grad_W = (gW[:nx] - gW[nx:]) / (2.0 * eps)

        # Dynamics
        d0 = np.zeros(self.sys_dynm.nd)
        v_f = self.sys_dynm.f(x0, d0)[None]
        v_G = self.sys_dynm.G(x0, d0)[None]
        vH_dstb = np.zeros((1, horizon, self.sys_dynm.nd))

        info = {
            "W_k": W_k, "W_A": W_A, "k_star": k_star, "winner_goal": winner_goal,
            "c_floor": c, "g_hat": g_hat, "bHp1_x": bHp1_x, "bHp1v_v": bHp1v_v,
        }

        return np.array([W_A]), vH_dstb, grad_W[None], v_f, v_G, info

