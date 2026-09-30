import os, glob
import numpy as np
import matplotlib.pyplot as plt

from Main import policy_filter
from Backup_pure import backup_filter
from Get_goal import goal_dyn
from plotter.Heatmap import heatmap_V
from plotter.Moving_plot import live_plotter
from Get_obstacles import check_collision
from plotter.Save_results import save_results_to_excel
from plotter.Plot_results import (plot_trajectories, plot_h_history, 
plot_v_history, plot_vdot_history, plot_hdot_history)

def clear_results(results_dir="plotter/Results", pattern="*.png"):
    os.makedirs(results_dir, exist_ok=True)
    for f in glob.glob(os.path.join(results_dir, pattern)):
        os.remove(f)

def print_step_summary(kk, x, u_nom, u_act, solve_dt, intervening,
                       goal=None, h_values=None, V=None, delta=None, value_name="CLF",
                       stop_step=None, dt=None, hdot=None, alh=None, V_max=None):
    
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

    if hdot is not None and alh is not None:
        hd = np.atleast_1d(hdot); al = np.atleast_1d(alh)
        hd_str = ", ".join(f"{v:7.4f}" for v in hd)
        mg_str = ", ".join(f"{v:7.4f}" for v in (al - hd))
        lines.append(f"  CBF rate       : hdot = [{hd_str}]   margin (-alpha*h - hdot) = [{mg_str}]")

    if V is not None:
        v_str = f"V = {V:8.4f}"
        if V_max is not None and V_max is not V:
            v_str += f"   V_max = {V_max:8.4f}"
        if delta is not None:
            v_str += f"   slack = {delta:8.4f}"
        lines.append(f"  {value_name:<15}: {v_str}")
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
                   env_noise, no_obs, obs_static,init_goal, goal_dyn_op, goal_motion, include_h0, 
                   include_v0, det_collison, make_plots, early_stop):

    print_summary = False
    live_plot = False
    if make_plots:
        clear_results()    

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
    
    # methods whose CLF is the worst-case cumulative cost V_max (P-CLF family)
    pclf_methods = {"pclf", "pclf_rpcbf_qp", "two_step_pclf_pcbf"}
    is_pclf = method in pclf_methods

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
    collision = None
    applied_u = []
    h_now_log = []
    h_hmax_log = []
    V_log = []          # instantaneous certificate V(x_t)  (P-CLF: clf_certificate at the current state)
    V_max_log = []      # the CLF the QP acts on: worst-case cumulative cost for P-CLF, plain V otherwise
    delta_log = []
    v_dot_log = []
    alV_log = []
    d_log = [] 

    if make_plots:
        X, Y, V = heatmap_V(safety, goal=init_goal, theta=x_s[2])

    for kk in range(safety.n_steps_sim):
        x_control = x_s.copy()
        d_env = safety.noise_single(env_noise, kk)
        # if kk % safety.interval_size == 0 and kk < 5 * safety.interval_size:
        #     print(f"env d at t={kk*safety.dt:.2f}s:", np.round(d_env, 3))
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
        V_max = None
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
            V_max = V                            # single CLF: same value in both logs
            V_log.append(V)
            V_max_log.append(V_max)
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
            V_max = V                            # single CLF: same value in both logs
            V_log.append(V)
            V_max_log.append(V_max)
            v_dot_log.append(Vdot)
            alV_log.append(alV)
            delta_log.append(delta if intervening != "infeasible" else np.nan)

        elif method == "pclf":
            # V_max: worst-case cumulative cost (what the QP / grad_V act on)
            # V    : instantaneous certificate V(x_t), reported for plotting
            u_act, intervening, solve_dt, V_max, delta, Vdot, alV, V = safety.pclf_qp(x=x_control,
                                                                                       u_nom=u_nom,
                                                                                       goal=goal,
                                                                                       use_slack=var_slack)
            V_log.append(V)
            V_max_log.append(V_max)
            v_dot_log.append(Vdot)
            alV_log.append(alV)
            delta_log.append(delta if intervening != "infeasible" else np.nan)

        elif method == "pclf_rpcbf_qp":
            u_act, intervening, solve_dt, V_max, delta, h_hmax, Vdot, alV, V = safety.pclf_rpcbf_qp(x=x_control,
                                                                                                   u_nom=u_nom,
                                                                                                   goal=goal,
                                                                                                   use_slack=var_slack)
            h_now_log.append(safety.cert.h_function(x_control))
            h_hmax_log.append(h_hmax)
            V_log.append(V)
            V_max_log.append(V_max)
            v_dot_log.append(Vdot)
            alV_log.append(alV)
            delta_log.append(delta if intervening != "infeasible" else np.nan)

        elif method == "two_step_pclf_pcbf":
            u_act, intervening, solve_dt, V_max, delta, h_hmax, Vdot, alV, V = safety.two_step_pclf_pcbf(x=x_control,
                                                                                                        u_nom=u_nom,
                                                                                                        goal=goal,
                                                                                                        use_slack=var_slack)
            h_now_log.append(safety.cert.h_function(x_control))
            h_hmax_log.append(h_hmax)
            V_log.append(V)
            V_max_log.append(V_max)
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
            V, V_max, delta = 0.0, 0.0, 0.0
            delta_log.append(delta if intervening != "infeasible" else np.nan)
            h_hmax_log.append(h_hmax)
            V_log.append(V)
            V_max_log.append(V_max)

        elif method == "pclf_goals":
            u_act, intervening, solve_dt, V, delta, Vdot, alV = safety.pclf_goals_qp(x=x_control,
                                                                                     u_nom=u_nom,
                                                                                     goals=goal_samples,
                                                                                     use_slack=var_slack)
            V_max = V                            # this is W_A - c ; single CLF, same value in both logs
            V_log.append(V)
            V_max_log.append(V_max)
            delta_log.append(delta if intervening != "infeasible" else np.nan)
            v_dot_log.append(Vdot)
            alV_log.append(alV)

        elif method == "None":
            u_act=u_nom
            solve_dt = 0.0
            intervening = "None"
            h_hmax, V, V_max, delta = 0.0, 0.0, 0.0, 0.0

        hit, clearance, j = check_collision(x_s, safety.obs_class, robot_radius=0.0)
        if hit and det_collison:
            collision = dict(step=kk, t=kk * safety.dt, obstacle=j, clearance=clearance, state=x_s.copy())
            print(f"COLLISION at t = {kk*safety.dt:.2f}s with obstacle {j} "
                f"(penetration {-clearance:.4f} m). Stopping simulation.")
            break

        x_s = safety.propagate(x=x_control, u=u_act, d=d_env, goal=goal)
        safety.obs_class.step()                       # obstacle moves with the same dt as the robot
        
        obs_log.append(safety.obs_class.pos_now().copy())
        trajectory_actual.append(x_s.copy())
        applied_u.append(np.asarray(u_act).copy())
        d_log.append(np.asarray(d_env).copy()) 

        if print_summary:
            # last logged CBF row for this step, if this method has a CBF
            hdot_k = safety.hdot_log[-1] if len(safety.hdot_log) == kk + 1 else None
            alh_k  = safety.alh_log[-1]  if len(safety.alh_log)  == kk + 1 else None
            print_step_summary(kk=kk, x=x_s, u_nom=u_nom, u_act=u_act, solve_dt=solve_dt,
                    intervening=intervening, goal=goal, h_values=h_hmax, V=V, delta=delta,
                    V_max=V_max if is_pclf else None,
                    value_name={"clf": "CLF value", "pclf": "P-CLF value", "clf_cbf": "CLF value",
                                "pclf_rpcbf_qp": "P-CLF value",
                                "two_step_pclf_pcbf": "P-CLF value"}.get(method, "CLF"),
                    stop_step=h_stop, dt=safety.dt, hdot=hdot_k, alh=alh_k)

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
    V_max_log = np.asarray(V_max_log)
    delta_log = np.asarray(delta_log)
    v_dot_log = np.array(v_dot_log)
    alV_log = np.array(alV_log)
    hdot_log = np.asarray(safety.hdot_log, dtype=float)             # (N, nh)  grad_h (f + G u_act) + dV_dt
    alh_log  = np.asarray(safety.alh_log,  dtype=float)             # (N, nh)  -alpha * h_hmax
    cbf_margin_log = (alh_log - hdot_log) if hdot_log.size else hdot_log   # >= 0 when satisfied
    obs_log = np.asarray(obs_log)                                   # (N+1, n_obs, 2)
    obs_now = safety.obs_class.pos_now()
    obstacles = [(obs_now[i, 0], obs_now[i, 1], safety.obs_class.R_O[i]) for i in range(len(safety.obs_class.R_O))]

    if make_plots:
        plot_trajectories(states_list=trajectory_actual, inputs_list=applied_u, goal=init_goal,
                        obstacles=obstacles, dt=safety.dt, title=method.upper(), results_dir="plotter/Results")

        if len(h_now_log) > 0:
            plot_h_history(h_now=h_now_log, h_hmax=h_hmax_log,
                        dt=safety.dt, path=os.path.join("plotter/Results", f"{method}_h_history.png"))

        if hdot_log.size > 0 and len(h_hmax_log) > 0:
            # finite-difference hdot of the same certificate (h_hmax) the QP gradient belongs to
            hdot_actual = np.vstack([np.diff(h_hmax_log, axis=0) / safety.dt,
                                    np.full((1, h_hmax_log.shape[1]), np.nan)])
            plot_hdot_history(hdot_log=hdot_log, h_hmax_log=h_hmax_log, dt=safety.dt, alpha=safety.alpha,
                            path=os.path.join("plotter/Results", f"{method}_h_dot_history.png"),
                            hdot_actual=hdot_actual)

        if len(V_log) > 0:
            if is_pclf:
                label, title = "V(x_t)", "instantaneous P-CLF value"
                max_label    = "V_max"
                max_title    = r"cumulative cost, worst case  $\max_d \int_0^T V(x_t)\,dt$  (gradient source)"
            else:
                label, title = "V(x)", "CLF value"
                max_label    = "V"
                max_title    = "CLF value (same certificate; no rollout for this method)"

            # top panel: instantaneous V ;  bottom panel: V_max (the CLF the QP acts on)
            plot_v_history(V_log=V_log, V_max_log=V_max_log, dt=safety.dt,
                        path=os.path.join("plotter/Results", f"{method}_V_history.png"),
                        value_label=label, title=title,
                        max_label=max_label, max_title=max_title)

            # Vdot is d/dt of the CLF the QP acts on, so the finite difference must use V_max
            Vdot_actual = np.append(np.diff(V_max_log) / safety.dt, np.nan)
            if len(v_dot_log) > 0:
                plot_vdot_history(Vdot_log=v_dot_log, V_log=V_max_log, delta_log=delta_log, dt=safety.dt, gamma=safety.gamma,
                                path=os.path.join("plotter/Results", f"{method}_V_dot_history.png"),
                                Vdot_actual=Vdot_actual, value_label="V_{max}" if is_pclf else "V",
                                title="P-CLF decrease condition" if is_pclf else "CLF decrease condition",
                                req_log=alV_log,
                                ell0_log=safety.ell0_log or None,
                                ellT_log=safety.ellT_log or None)
            if len(safety.tail_log) > 0:
                tl = np.asarray(safety.tail_log)
                print(f"[tail l(x_T)/l(x_0)] median {np.median(tl):.3f}  90% {np.percentile(tl,90):.3f}  max {tl.max():.3f}")
    
        if live_plot:    
            plt.close("all")
            live_plot = live_plotter(obs_pos=obs_log[0], obs_radius=safety.obs_class.R_O)
            live_plot.update(trajectory=trajectory_actual, goal=init_goal, obs_traj=obs_log, pause=1e-5)
        
    return {"states": trajectory_actual, "inputs": applied_u, "h_now": h_now_log, "obs": obs_log,
            "h_hmax": h_hmax_log, "V": V_log, "V_max": V_max_log, "delta": delta_log,
            "Vdot": v_dot_log, "alV": alV_log,
            "hdot": hdot_log, "alH": alh_log, "cbf_margin": cbf_margin_log,
            "ell0": np.asarray(safety.ell0_log, dtype=float),
            "ellT": np.asarray(safety.ellT_log, dtype=float),
            "cert_valid": np.asarray(safety.cert_valid_log, dtype=float),   # 1.0 / 0.0
            "collision": collision,
            "d_env": np.asarray(d_log), 
            "safety": safety}

if __name__ == "__main__":

    # *1 "proportional_policy" or "random_policy" or "constant_policy" or "backup_policy"
    # *2 "rpcbf", "clf", "clf_cbf", "pclf", "pclf_goals", "pure_backup", "None", "pclf_rpcbf_qp", "two_step_pclf_pcbf"
   
    settings = {
        "method": "pclf",        # *2
        "x_s": [1.0, 2.8, 0.0],           # initial position [x, y, yaw]
        "controller": "backup_policy",  # "clf_nom" or *1
        "h_controller": "backup_policy",       # *1
        "v_controller": "proportional_policy", # *1
        "var_slack": True,
        "rollout_noise": "BangBang",              # Uniform or Zero or BangBang
        "env_noise": "BangBang",                  # Uniform or Zero or BangBang
        "no_obs": "single",                    # multi or single
        "obs_static": True,
        "init_goal": [4.0, 1.0],              # mean goal position
        "goal_dyn_op": "static",              # static or sin_y or random
        "goal_motion": "stoc",                # stoc or det (only for sin_y)
        "include_h0": True,
        "include_v0": True,
        "det_collison" : False,
        "make_plots"   : False,
        "early_stop"   : 4000
    }

    results  = run_simulation(**settings)  
    save_results_to_excel({settings["controller"]: results}, settings)
    