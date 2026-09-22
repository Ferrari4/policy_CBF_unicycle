import matplotlib.pyplot as plt
import numpy as np
import os

from Main import policy_filter
from Backup_pure import backup_filter
from Get_goal import goal_dyn
from Moving_plot import live_plotter
from plotter.Data_generator import save_results_to_excel
from Plot_results import (plot_trajectories, plot_h_history, plot_v_history, plot_vdot_history)

def print_step_summary(kk, x, u_nom, u_act, solve_dt, intervening,
                       goal=None, h_values=None, V=None, delta=None, value_name="CLF",
                       stop_step=None, dt=None):
    
    status = (
        "INTERVENE" if intervening is True
        else intervening if isinstance(intervening, str)
        else "NOMINAL"
    )
    lines = [
        f"\n[{kk:2d}] Step Summary",
        f"  State          : x = [{x[0]:7.4f}, {x[1]:7.4f}, {x[2]:7.4f}]",
        f"  Nominal control: u_nom = [{u_nom[0]:6.3f}, {u_nom[1]:6.3f}]",
        f"  Actual control : u_act = [{u_act[0]:6.3f}, {u_act[1]:6.3f}]",
        f"  Goal           : goal = [{goal[0]:7.4f}, {goal[1]:7.4f}]"
    ]
    if h_values is not None:
        h_str = ", ".join(f"{h:7.4f}" for h in np.atleast_1d(h_values))
        lines.append(f"  Barrier values : h = [{h_str}]")

    if V is not None:
        if delta is not None:
            lines.append(f"  {value_name:<15}: V = {V:8.4f}   slack = {delta:8.4f}")
        else:
            lines.append(f"  {value_name:<15}: V = {V:8.4f}")
    lines.append(f"  Solve time     : {solve_dt * 1000:.2f} ms")
    lines.append(f"  Status         : {status}")
    if goal is not None:
        distance = np.linalg.norm(goal - x[:2])
        lines.append(f"  Distance goal  : {distance:.4f}")
    if stop_step is not None:
        s = ", ".join("full" if k < 0 else f"{k*dt:.2f}s" for k in np.atleast_1d(stop_step))
        lines.append(f"  Rollout stop   : [{s}]")
    print("\n".join(lines))

def run_simulation(method, x_s, controller, h_controller ,v_controller, var_slack, rollout_noise, 
                   env_noise, no_obs, obs_static,init_goal, goal_dyn_op, goal_motion, include_h0, include_v0):

    print_summary = True
    live_plot = False
    early_stop = 5000

    safety = policy_filter(controller=controller,
                           h_controller=h_controller, 
                           v_controller=v_controller, 
                           obstacles=no_obs, 
                           static_obs=obs_static,
                           noise_choice=rollout_noise,
                           include_h0=include_h0,
                           include_v0=include_v0)

    backup_safety = backup_filter(policy_class=safety)

    goal_class = goal_dyn(goal_dyn=goal_dyn_op, 
                          goal_motion=goal_motion, 
                          noise_sampler=safety.test_noise, 
                          init_goal=init_goal,
                          dt=safety.dt)
    
    valid_methods = {"rpcbf", "clf", "clf_cbf", "pclf_goals",
                     "pclf", "pure_backup", "None", "pclf_rpcbf_qp", "two_step_pclf_pcbf"}
    
    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}")
    
    x_s = np.array(x_s)
    g_hat = None
    if method == "pclf_goals":
        K_goals, A_goal = 3, 0.5
        r  = A_goal * np.sqrt(safety.rng.uniform(size=K_goals))
        ph = safety.rng.uniform(0.0, 2 * np.pi, size=K_goals)
        goal_samples = np.asarray(init_goal, dtype=float) + np.stack([r * np.cos(ph), r * np.sin(ph)], axis=1)
        g_hat = goal_samples.mean(axis=0)

    trajectory_actual = [x_s.copy()]
    obs_log = [safety.obs_class.pos_now().copy()]     # (n_obs, 2) per step
    applied_u = []
    h_now_log = []
    h_hmax_log = []
    V_log = []
    delta_log = []
    v_dot_log = []
    alV_log = []
    
    for kk in range(safety.n_steps_sim):
        x_control = x_s.copy()
        d_env = safety.noise_single(env_noise, kk)
        goal = goal_class.call_goal()

        goals = np.stack(
                [np.asarray(goal_class.call_goal(), dtype=float).copy() for _ in range(15)],
                axis=0)
        goal_mean = np.mean(goals, axis=0)

        # Bypass goal
        goal = goal_mean

        if controller != "clf_nom":
            u_nom = np.asarray(safety.u_nominal(x_control, controller, goal), dtype=float).copy()
            u_nom[0] = np.clip(u_nom[0], safety.v_min, safety.v_max)
            u_nom[1] = np.clip(u_nom[1], -safety.om_max, safety.om_max)
        else:
            u_nom = np.zeros(safety.dyn.nu)

        h_stop = None
        h_hmax = None
        V = None
        delta = None

        if method == "rpcbf":
            u_act, intervening, solve_dt, h_hmax, h_stop = safety.rpcbf_qp(x=x_control, 
                                                                   u_nom=u_nom, 
                                                                   goal=goal,
                                                                   use_slack=var_slack)
            h_now = safety.cert.h_function(x_control)
            h_now_log.append(h_now)
            h_hmax_log.append(h_hmax)

        elif method == "clf":
            u_act, intervening, solve_dt, V, delta, Vdot, alV = safety.clf_qp(u_nom=u_nom, 
                                                                   x=x_control, 
                                                                   d_nom=d_env, 
                                                                   goal=goal,
                                                                   use_slack=var_slack)
            V_log.append(V)
            v_dot_log.append(Vdot)
            alV_log.append(alV)
            delta_log.append(delta if intervening != "infeasible" else np.nan)

        elif method == "clf_cbf":
            u_act, intervening, solve_dt, V, delta, h_hmax, Vdot, alV = safety.clf_cbf_qp(x=x_control, 
                                                                               u_nom=u_nom, 
                                                                               d_nom=d_env, 
                                                                               goal=goal,
                                                                               use_slack=var_slack)
            h_now_log.append(safety.cert.h_function(x_control))
            h_hmax_log.append(h_hmax)
            V_log.append(V)
            v_dot_log.append(Vdot)
            alV_log.append(alV)
            delta_log.append(delta if intervening != "infeasible" else np.nan)

        elif method == "pclf":
            u_act, intervening, solve_dt, V, delta, Vdot, alV = safety.pclf_qp(x=x_control, 
                                                                                u_nom=u_nom,
                                                                                goal=goal,
                                                                                use_slack=var_slack)
            V_log.append(V)
            v_dot_log.append(Vdot)
            alV_log.append(alV)
            delta_log.append(delta if intervening != "infeasible" else np.nan)

        elif method == "pclf_rpcbf_qp":
            u_act, intervening, solve_dt, V, delta, h_hmax, Vdot, alV = safety.pclf_rpcbf_qp(x=x_control, 
                                                                                            u_nom=u_nom,
                                                                                            d_nom=d_env,
                                                                                            goal=goal,
                                                                                            use_slack=var_slack)
            h_now_log.append(safety.cert.h_function(x_control))
            h_hmax_log.append(h_hmax)
            V_log.append(V)
            v_dot_log.append(Vdot)
            alV_log.append(alV)
            delta_log.append(delta if intervening != "infeasible" else np.nan)

        elif method == "two_step_pclf_pcbf":
            u_act, intervening, solve_dt, V, delta, h_hmax, Vdot, alV = safety.two_step_pclf_pcbf(x=x_control,
                                                                                                 u_nom=u_nom,
                                                                                                 d_nom=d_env,
                                                                                                 goal=goal,
                                                                                                 use_slack=var_slack)
            h_now_log.append(safety.cert.h_function(x_control))
            h_hmax_log.append(h_hmax)
            V_log.append(V)
            v_dot_log.append(Vdot)
            alV_log.append(alV)
            delta_log.append(delta if intervening != "infeasible" else np.nan)

        elif method == "pure_backup":
            u_act, intervening, solve_dt = backup_safety.safety_Bcbf(x=x_control,
                                                                     u_nom=u_nom, 
                                                                     d_nom=d_env)
            h_now = safety.cert.h_function(x_control)
            h_now_log.append(h_now)
            h_hmax = np.zeros_like(h_now)
            V, delta = 0.0, 0.0
            delta_log.append(delta if intervening != "infeasible" else np.nan)
            h_hmax_log.append(h_hmax)
            V_log.append(V)

        elif method == "pclf_goals":
            u_act, intervening, solve_dt, V, delta, Vdot, alV = safety.pclf_goals_qp(x=x_control,
                                                                                     u_nom=u_nom,
                                                                                     goals=goal_samples,
                                                                                     use_slack=var_slack)
            V_log.append(V)                      # this is W_A - c
            delta_log.append(delta if intervening != "infeasible" else np.nan)
            v_dot_log.append(Vdot)
            alV_log.append(alV)

        elif method == "None":
            u_act=u_nom
            solve_dt = 0.0
            intervening = "None"
            h_hmax, V, delta = 0.0, 0.0, 0.0

        x_s = safety.propagate(x=x_control, u=u_act, d=d_env, goal=goal)
        safety.obs_class.step()                       # obstacle moves with the same dt as the robot
        obs_log.append(safety.obs_class.pos_now().copy())
        trajectory_actual.append(x_s.copy())
        applied_u.append(np.asarray(u_act).copy())

        if print_summary:
            print_step_summary(kk=kk, x=x_s, u_nom=u_nom, u_act=u_act, solve_dt=solve_dt, 
                    intervening=intervening, goal=goal, h_values=h_hmax, V=V, delta=delta, 
                    value_name={"clf": "CLF value", "pclf": "P-CLF value", "clf_cbf": "CLF value","two_step_pclf_pcbf": "P-CLF value"}.get(method, "CLF"),
                    stop_step=h_stop, dt=safety.dt)

        if np.linalg.norm(goal - x_s[:2]) < 0.05:
            print("Goal reached!")
            print(f"final: {np.linalg.norm(init_goal - x_s[:2])}")
            print(f"compute mean: {np.linalg.norm(goal_mean - x_s[:2])}")
            if g_hat is not None:
                print(f"compute mean: {np.linalg.norm(g_hat - x_s[:2])}")
            break

        if kk >= early_stop:
            print(f"Early stop at step:{kk}")
            print(f"final: {np.linalg.norm(init_goal - x_s[:2])}")
            if g_hat is not None:
                print(f"compute mean: {np.linalg.norm(g_hat - x_s[:2])}")
            break

        if intervening == "infeasible":
            print(f"QP stopped due to infeasibility at step {kk}")
            break

    # Plotting
    trajectory_actual = np.asarray(trajectory_actual)
    applied_u = np.asarray(applied_u)
    h_now_log = np.asarray(h_now_log)
    h_hmax_log = np.asarray(h_hmax_log)
    V_log = np.asarray(V_log)
    delta_log = np.asarray(delta_log)
    v_dot_log = np.array(v_dot_log)
    alV_log = np.array(alV_log)
    obs_log = np.asarray(obs_log)                                   # (N+1, n_obs, 2)
    obs_now = safety.obs_class.pos_now()
    obstacles = [(obs_now[i, 0], obs_now[i, 1], safety.obs_class.R_O[i]) for i in range(len(safety.obs_class.R_O))]

    plot_trajectories(states_list=trajectory_actual, inputs_list=applied_u, goal=init_goal,
                      obstacles=obstacles, dt=safety.dt, title=method.upper(), results_dir="Results")

    if len(h_now_log) > 0:
        plot_h_history(h_now=h_now_log, h_hmax=h_hmax_log, 
                       dt=safety.dt, path=os.path.join("Results", f"{method}_h_history.png"))

    if len(V_log) > 0:
        label = "V_pclf(x)" if method == "pclf" else "V(x)"
        title = "P-CLF value" if method == "pclf" else "CLF value"

        plot_v_history(V_log=V_log, delta_log=delta_log, dt=safety.dt, 
                       path=os.path.join("Results", f"{method}_V_history.png"), 
                       value_label=label, title=title)

        Vdot_actual = np.append(np.diff(V_log) / safety.dt, np.nan)
        if len(v_dot_log) > 0:
            plot_vdot_history(Vdot_log=v_dot_log, V_log=V_log, delta_log=delta_log, dt=safety.dt, gamma=safety.gamma, 
                            path=os.path.join("Results", f"{method}_V_dot_history.png"),
                            Vdot_actual=Vdot_actual,  value_label="W" if method == "pclf" else "V", 
                            title="P-CLF decrease condition" if method == "pclf" else "CLF decrease condition")
        if len(safety.tail_log) > 0:
            tl = np.asarray(safety.tail_log)
            print(f"[tail l(x_T)/l(x_0)] median {np.median(tl):.3f}  90% {np.percentile(tl,90):.3f}  max {tl.max():.3f}")

    if live_plot:    
        plt.close("all")
        live_plot = live_plotter(obs_pos=obs_log[0], obs_radius=safety.obs_class.R_O)
        live_plot.update(trajectory=trajectory_actual, goal=init_goal, obs_traj=obs_log, pause=1e-5)
        
    return {"states": trajectory_actual, "inputs": applied_u, "h_now": h_now_log, "obs": obs_log,
            "h_hmax": h_hmax_log, "V": V_log, "delta": delta_log,
            "Vdot": v_dot_log, "alV": alV_log,
            "ell0": np.asarray(safety.ell0_log, dtype=float),
            "ellT": np.asarray(safety.ellT_log, dtype=float),
            "cert_valid": np.asarray(safety.cert_valid_log, dtype=float),   # 1.0 / 0.0
            "safety": safety}

if __name__ == "__main__":

    # *1 "proportional_policy" or "random_policy" or "constant_policy" or "backup_policy"
    settings = {
        "method": "pclf",   # "rpcbf", "clf", "clf_cbf", "pclf", "pclf_goals", "pure_backup", "None", "pclf_rpcbf_qp", "two_step_pclf_pcbf"
        "x_s": [1.0, 2.8, 0.0],           # initial position [x, y, yaw]
        "controller": "clf_nom",  # "clf_nom" or *1
        "h_controller": "backup_policy",      # *1
        "v_controller": "constant_policy",# *1
        "var_slack": True,
        "rollout_noise": "Zero",              # Uniform or Zero or BangBang
        "env_noise": "Zero",                  # Uniform or Zero or BangBang
        "no_obs": "single",                    # multi or single
        "obs_static": True,
        "init_goal": [4.0, 1.0],              # mean goal position
        "goal_dyn_op": "static",              # static or sin_y or random
        "goal_motion": "stoc",                # stoc or det (only for sin_y)
        "include_h0": True,
        "include_v0": True,
    }

    results  = run_simulation(**settings)  
    save_results_to_excel({settings["controller"]: results}, settings)

