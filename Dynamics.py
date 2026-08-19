import numpy as np
from scipy.integrate import solve_ivp

from Control_policy import policy
from Noise_sampler import noise_train_sampler

class sys_dynm_dd:
    def __init__(self, policy_class: policy, dt=0.1):
        self.dt = dt
        self.controller = policy_class
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

    def solve_one_step_scippy(self, x0, d, control_fn):
        x0 = self.chk_x(x0)
        d = self.chk_d(d)

        def closed_loop_dynamics(t, state):
            control = self.chk_u(control_fn(state))
            return self.dynamics(state, control, d)
        sol = solve_ivp(
            fun=closed_loop_dynamics,
            t_span=(0.0, self.dt),
            y0=x0,
            t_eval=[self.dt],
            method="RK45"
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        return sol.y[:, -1]

    def solve_one_step_mannual(self, x0, d, control_fn):
        x0 = self.chk_x(x0)
        d = self.chk_d(d)

        def rhs(state):
            return self.dynamics(state, control_fn(state), d)

        k1 = rhs(x0)
        k2 = rhs(x0 + 0.5 * self.dt * k1)
        k3 = rhs(x0 + 0.5 * self.dt * k2)
        k4 = rhs(x0 + self.dt * k3)
        return x0 + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def one_step_u(self, x0, u, d, method="scipy_IVP"):
        x0 = self.chk_x(x0)
        d = self.chk_d(d)
        u = self.chk_u(u)

        def rhs(state, t=None):
            return self.dynamics(state, u, d)

        if method == "scipy_IVP":
            sol = solve_ivp(
                fun=lambda t, state: rhs(state, t),
                t_span=(0.0, self.dt),
                y0=x0,
                t_eval=[self.dt],
                method="RK45"
            )
            if not sol.success:
                raise RuntimeError(f"Integration failed: {sol.message}")
            return sol.y[:, -1]

        elif method == "manual_RK4":
            k1 = rhs(x0)
            k2 = rhs(x0 + 0.5 * self.dt * k1)
            k3 = rhs(x0 + 0.5 * self.dt * k2)
            k4 = rhs(x0 + self.dt * k3)
            return x0 + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        else:
            raise ValueError(f"Unknown method: {method}")

    def rollout_ivp(self, x0, H_dstb, policy_name: str, method="scipy_IVP"):
        if policy_name == "proportional_policy":
            control_method = self.controller.proportional_policy
        elif policy_name == "constant_policy":
            control_method = self.controller.constant_policy
        elif policy_name == "backup_policy":
            control_method = self.controller.backup_policy
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        state = self.chk_x(x0)
        H_dstb = np.asarray(H_dstb)
        trajectory = [state.copy()]
        for k in range(H_dstb.shape[0]):
            d_k = self.chk_d(H_dstb[k])
            if method == "scipy_IVP":
                state = self.solve_one_step_scippy(state, d_k, control_method)
            else:
                state = self.solve_one_step_mannual(state, d_k, control_method)
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
        