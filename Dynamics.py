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

    def solve_ivp_fun(self, x0, d, control_fn=None, u=None):

        method = self.ivp_method

        # ==========================================================
        # SINGLE PROCESS
        # ==========================================================
        if self.process == "single":

            x0 = self.chk_x(x0)
            d = self.chk_d(d)

            # ------------------------------------------------------
            # Fixed input u
            # ------------------------------------------------------
            if control_fn is None:

                if u is None:
                    raise ValueError("u must be provided when control_fn is None.")

                u = self.chk_u(u)

                def rhs(state):
                    return self.dynamics(state, u, d)

            # ------------------------------------------------------
            # Feedback controller
            # ------------------------------------------------------
            else:

                def rhs(state):
                    control = self.chk_u(control_fn(state))
                    return self.dynamics(state, control, d)

        # ==========================================================
        # BATCH PROCESS
        # ==========================================================
        elif self.process == "batch":

            x0 = np.asarray(x0).reshape(-1, 3)
            d = np.asarray(d).reshape(-1, 3)

            if x0.shape[0] != d.shape[0]:
                raise ValueError(
                    f"Batch sizes must match: x0={x0.shape}, d={d.shape}"
                )

            B = x0.shape[0]

            # scipy solve_ivp is not being used for batched states
            if method == "scipy_IVP":
                print("Batch process detected: switching scipy_IVP -> manual_RK4")
                method = "manual_RK4"

            # ------------------------------------------------------
            # Fixed input u
            # ------------------------------------------------------
            if control_fn is None:

                if u is None:
                    raise ValueError("u must be provided when control_fn is None.")

                u = np.asarray(u).reshape(-1, 2)

                # Allow one fixed u to be applied to entire batch
                if u.shape[0] == 1 and B > 1:
                    u = np.tile(u, (B, 1))

                if u.shape[0] != B:
                    raise ValueError(
                        f"Batch sizes must match: x0={x0.shape}, u={u.shape}"
                    )

                def rhs(state):
                    return self.dynamics_batch(state, u, d)

            # ------------------------------------------------------
            # Feedback controller
            # ------------------------------------------------------
            else:

                def rhs(state):
                    control = np.asarray(control_fn(state)).reshape(-1, 2)

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

        # ==========================================================
        # SCIPY IVP
        # ==========================================================
        if method == "scipy_IVP":

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

        # ==========================================================
        # MANUAL RK4
        # ==========================================================
        elif method == "manual_RK4":

            k1 = rhs(x0)
            k2 = rhs(x0 + 0.5 * self.dt * k1)
            k3 = rhs(x0 + 0.5 * self.dt * k2)
            k4 = rhs(x0 + self.dt * k3)

            return x0 + (self.dt / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )

        else:
            raise ValueError(f"Unknown IVP method: {method}")

    def rollout_ivp(self, x0, H_dstb, policy_name: str, u=None):

        if self.process == "single":
            state = self.chk_x(x0)
        else:
            state = np.asarray(x0).reshape(-1, 3)

        H_dstb = np.asarray(H_dstb)

        trajectory = [state.copy()]

        if policy_name == "proportional_policy":
            control_method = self.controller.proportional_policy

        elif policy_name == "constant_policy":
            control_method = self.controller.constant_policy

        elif policy_name == "backup_policy":
            control_method = self.controller.backup_policy

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
    dynm = sys_dynm_dd(policy_class=pol_clas)
    noise = noise_train_sampler(nd=dynm.nd, rng=np.random.default_rng(42))
    x0 = np.array([-0.5, 0.0, 0.0])
    x_s1 = dynm.one_step_u(x0, [0.5, 0.1], [0.0, 0.0, 0.0], method="scipy_IVP")
    x_s2 = dynm.one_step_u(x0, [0.5, 0.1], [0.0, 0.0, 0.0], method="manual_RK4")
    BH_dstb, _ = noise.bangbang_uniform_train(n_samples=2, n_samples_uniform=1, horizon=10, interval_size=3)
    print(BH_dstb, " BH_dstb")
    for H_dstb in BH_dstb:
        print("Disturbance sequence:\n", H_dstb)
        trajectory_scippy = dynm.rollout_ivp(x0, H_dstb, policy_name="proportional_policy", method="scipy_IVP")
        trajectory_mannual = dynm.rollout_ivp(x0, H_dstb, policy_name="proportional_policy", method="manual_RK4")
        print("Trajectory (manual RK4):\n", trajectory_mannual)
        print("Trajectory (scipy):\n", trajectory_scippy)
        print(trajectory_scippy.shape, "Trajectory shape scippy")
        print(trajectory_mannual.shape, "Trajectory shape manual")
        print(np.allclose(trajectory_scippy, trajectory_mannual), ": Are the trajectories close?")
        print("max|scipy - manual| =", np.max(np.abs(trajectory_scippy - trajectory_mannual)))
        print(x_s1)
        print(x_s2)

        