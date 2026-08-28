import numpy as np
from scipy.integrate import solve_ivp

from Control_policy import policy
from Noise_sampler import noise_train_sampler

class sys_dynm_dd:
    def __init__(self, policy_class: policy, dt=0.1, ivp_method="manual_RK4", process="batch"):
        self.dt = dt
        self.controller = policy_class
        self.ivp_method = ivp_method
        self.process=process
        if getattr(policy_class, "process", None) != process:
            policy_class.process = process
        self.nx = 3  # state dimension [px, py, theta]
        self.nu = 2  # control dimension [v, omega]
        self.nd = 3  # disturbance dimension [d1, d2, d3], intriduced as additive to dot products of x,y,theetha

    def dynamics(self, state, control, disturbance):
        state = self.chk_x(state)
        control = self.chk_u(control)
        disturbance = self.chk_d(disturbance)
        xdot = self.f(state, disturbance) + self.G(state, disturbance) @ control
        return xdot

    def chk_x(self, state):
        state = np.asarray(state)
        assert state.shape == (self.nx,)
        return state

    def chk_u(self, control):
        control = np.asarray(control)
        assert control.shape == (self.nu,)
        return control

    def chk_d(self, disturbance):
        disturbance = np.asarray(disturbance)
        assert disturbance.shape == (self.nd,)
        return disturbance

    def f(self, state, disturbance):
        state = self.chk_x(state)
        disturbance = self.chk_d(disturbance)
        return np.array([disturbance[0], disturbance[1], disturbance[2]])

    def G(self, state, disturbance):
        state = self.chk_x(state)
        disturbance = self.chk_d(disturbance)
        _, _, th = state
        return np.array([
            [np.cos(th), 0.0],
            [np.sin(th), 0.0],
            [0.0,        1.0],
        ])

    def dynamics_batch(self, bx, bu, bd):
        th = bx[:, 2]
        v, om = bu[:, 0], bu[:, 1]

        return np.stack([
            bd[:, 0] + np.cos(th) * v,
            bd[:, 1] + np.sin(th) * v,
            bd[:, 2] + om
        ], axis=1)

    def solve_ivp_fun(self, x0, d, control_fn=None, u=None):

        if self.process == "single":
            x0 = self.chk_x(x0)
            d = self.chk_d(d)

            if control_fn is None:
                if u is None:
                    raise ValueError("u must be provided when control_fn is None.")
                u = self.chk_u(u)

                def rhs_fixed(state):
                    return self.dynamics(state, u, d)
                rhs = rhs_fixed

            else:
                def rhs_control(state):
                    control = self.chk_u(control_fn(state))
                    return self.dynamics(state, control, d)
                rhs = rhs_control
                
        elif self.process == "batch":
            x0 = np.asarray(x0).reshape(-1, self.nx)
            d = np.asarray(d).reshape(-1, self.nd)
            if x0.shape[0] != d.shape[0]:
                raise ValueError(
                    f"Batch sizes must match: x0={x0.shape}, d={d.shape}"
                )

            B = x0.shape[0]
            if self.ivp_method == "scipy_IVP":
                print("Batch process detected: switching scipy_IVP -> manual_RK4")
                self.ivp_method = "manual_RK4"

            if control_fn is None:
                if u is None:
                    raise ValueError("u must be provided when control_fn is None.")
                u = np.asarray(u).reshape(-1, self.nu)
                # Allow one fixed u to be applied to entire batch
                if u.shape[0] == 1 and B > 1:
                    u = np.tile(u, (B, 1))
                if u.shape[0] != B:
                    raise ValueError(
                        f"Batch sizes must match: x0={x0.shape}, u={u.shape}"
                    )

                def rhs(state):
                    return self.dynamics_batch(state, u, d)

            else:

                def rhs(state):
                    control = np.asarray(control_fn(state)).reshape(-1, self.nu)

                    if control.shape[0] != state.shape[0]:
                        raise ValueError(
                            f"Controller returned {control.shape}, "
                            f"expected ({state.shape[0]}, 2)"
                        )

                    return self.dynamics_batch(state, control, d)

        else:
            raise ValueError(
                f"Unknown process: {self.process}. "
                "Use 'single' or 'batch'."
            )

        if self.ivp_method == "scipy_IVP":
            sol = solve_ivp(
                fun=lambda t, state: rhs(state),
                t_span=(0.0, self.dt),
                y0=x0,
                t_eval=[self.dt],
                method="RK45"
            )
            if not sol.success:
                raise RuntimeError(
                    f"Integration failed: {sol.message}"
                )
            return sol.y[:, -1]

        elif self.ivp_method == "manual_RK4":
            k1 = rhs(x0)
            k2 = rhs(x0 + 0.5 * self.dt * k1)
            k3 = rhs(x0 + 0.5 * self.dt * k2)
            k4 = rhs(x0 + self.dt * k3)
            return x0 + (self.dt / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )
        else:
            raise ValueError(f"Unknown IVP method: {self.ivp_method}")

    def rollout_ivp(self, x0, H_dstb, policy_name: str, u=None):

        if self.process == "single":
            state = self.chk_x(x0)
        else:
            state = np.asarray(x0).reshape(-1, self.nx)

        H_dstb = np.asarray(H_dstb)
        trajectory = [state.copy()]

        if policy_name == "proportional_policy":
            control_method = self.controller.proportional_policy

        elif policy_name == "constant_policy":
            control_method = self.controller.constant_policy

        elif policy_name == "backup_policy":
            control_method = self.controller.backup_policy

        elif policy_name == "random_policy":
            control_method = self.controller.random_policy

        elif policy_name == "input_U":
            control_method = None

        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        for k in range(H_dstb.shape[0]):
            d_k = H_dstb[k]
            state = self.solve_ivp_fun(
                x0=state,
                d=d_k,
                control_fn=control_method,
                u=u
            )
            trajectory.append(state.copy())

        return np.asarray(trajectory)

if __name__ == "__main__":
    pol_clas = policy()
    dynm = sys_dynm_dd(policy_class=pol_clas, process="single")
    noise = noise_train_sampler(nd=dynm.nd, rng=np.random.default_rng(42))

    x0 = np.array([-0.5, 0.0, 0.0])
    u_test = np.array([0.5, 0.1])

    # --- integrator-only check: fixed u, no feedback, single step --------
    d_one = np.zeros((1, dynm.nd))          # horizon-1 disturbance sequence

    dynm.ivp_method = "scipy_IVP"
    x_s1 = dynm.rollout_ivp(x0, d_one, policy_name="input_U", u=u_test)[-1]

    dynm.ivp_method = "manual_RK4"
    x_s2 = dynm.rollout_ivp(x0, d_one, policy_name="input_U", u=u_test)[-1]

    print("one-step scipy :", x_s1)
    print("one-step RK4   :", x_s2)
    print("max|scipy - manual| =", np.max(np.abs(x_s1 - x_s2)))

    # --- closed-loop rollout check --------------------------------------
    BH_dstb, _ = noise.bangbang_uniform_train(
        n_samples=2, n_samples_uniform=1, horizon=10, interval_size=3
    )
    print("BH_dstb shape:", BH_dstb.shape)

    for i, H_dstb in enumerate(BH_dstb):
        print(f"\n--- sample {i} ---")
        print("Disturbance sequence:\n", H_dstb)

        dynm.ivp_method = "scipy_IVP"
        traj_scipy = dynm.rollout_ivp(x0, H_dstb, policy_name="proportional_policy")

        dynm.ivp_method = "manual_RK4"
        traj_rk4 = dynm.rollout_ivp(x0, H_dstb, policy_name="proportional_policy")

        print("Trajectory (manual RK4):\n", traj_rk4)
        print("Trajectory (scipy):\n", traj_scipy)
        print(traj_scipy.shape, "shape scipy")
        print(traj_rk4.shape, "shape manual")
        print(np.allclose(traj_scipy, traj_rk4), ": Are the trajectories close?")
        print("max|scipy - manual| =", np.max(np.abs(traj_scipy - traj_rk4)))

    # still process="single" here
    singles = [dynm.rollout_ivp(x0, H, policy_name="proportional_policy")
               for H in BH_dstb]
    traj_single = np.stack(singles, axis=1)         # (11, B, 3)

    # --- batch: all samples in one rollout ------------------------------
    dynm.process = "batch"
    pol_clas.process = "batch"          # both flags must agree
    dynm.ivp_method = "manual_RK4"

    B = BH_dstb.shape[0]
    x0_batch = np.tile(x0, (B, 1))                  # (B, 3)
    H_batch = np.transpose(BH_dstb, (1, 0, 2))      # (2,10,3) -> (10,B,3)

    traj_batch = dynm.rollout_ivp(
        x0_batch, H_batch, policy_name="proportional_policy"
    )
    print("batch trajectory shape:", traj_batch.shape)   # (11, B, 3)

    # per-sample max deviation from the single-mode RK4 runs, vectorised
    traj_single = np.stack(singles, axis=1)              # (11, B, 3)
    err = np.max(np.abs(traj_batch - traj_single), axis=(0, 2))   # (B,)
    print("per-sample max|batch - single| =", err)
    print("batch matches single:", np.allclose(traj_batch, traj_single, rtol=0, atol=1e-10))