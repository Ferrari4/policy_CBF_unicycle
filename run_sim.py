import os, glob
import numpy as np
import pandas as pd
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

def append_params_sheet(xlsx_path, params, sheet="params"):
    """Add/replace a key-value sheet of run parameters in an existing workbook."""
    flat = {k: (",".join(f"{v:.4f}" for v in val) if isinstance(val, (list, tuple, np.ndarray)) else val)
            for k, val in params.items()}
    df = pd.Series(flat, name="value").rename_axis("parameter").reset_index()
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        df.to_excel(xw, sheet_name=sheet, index=False)

def clear_results(results_dir="plotter/Results", pattern="*.png"):
    os.makedirs(results_dir, exist_ok=True)
    for f in glob.glob(os.path.join(results_dir, pattern)):
        os.remove(f)

def print_step_summary(kk, x, u_nom, u_act, solve_dt, intervening,
                       goal=None, h_values=None, V=None, delta=None, value_name="CLF",
                       stop_step=None, dt=None, hdot=None, alh=None, V_max=None,
                       J_u=None, J_int=None, cbf_on=None):

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
    if J_u is not None:
        e_str = f"J_u = {J_u:7.4f} s"
        if J_int is not None:
            e_str += f"   J_int = {J_int:7.4f} s   CBF {'ACTIVE' if cbf_on else 'idle'}"
        lines.append(f"  Effort (cum.)  : {e_str}")
    lines.append(f"  Solve time     : {solve_dt * 1000:.2f} ms")
    lines.append(f"  Status         : {status}")
    if goal is not None:
        distance = np.linalg.norm(goal - x[:2])
        lines.append(f"  Distance goal  : {distance:.4f}")
    if stop_step is not None:
        s = ", ".join("full" if k < 0 else f"{k*dt:.2f}s" for k in np.atleast_1d(stop_step))
        lines.append(f"  Rollout stop   : [{s}]")
    print("\n".join(lines))

def run_simulation(method, x_s, controller, h_controller, v_controller, var_slack,
                   rollout_noise_cbf, rollout_noise_clf, env_noise, no_obs, obs_static,
                   init_goal, goal_dyn_op, goal_motion, include_h0, include_v0, det_collison, make_plots, early_stop,
                   print_summary=True, T_rollout=None, T_rollout_clf=None, seed=None):

    live_plot = False
    if make_plots:
        clear_results()

    safety = policy_filter(controller=controller,
                           h_controller=h_controller, 
                           v_controller=v_controller, 
                           obstacles=no_obs, 
                           static_obs=obs_static,
                           noise_choice_cbf=rollout_noise_cbf,
                           noise_choice_clf=rollout_noise_clf,
                           include_h0=include_h0,
                           include_v0=include_v0,
                           T_rollout=T_rollout,
                           T_rollout_clf=T_rollout_clf)

    backup_safety = backup_filter(policy_class=safety)

    goal_class = goal_dyn(goal_dyn=goal_dyn_op, 
                          goal_motion=goal_motion, 
                          noise_sampler=safety.test_noise, 
                          init_goal=init_goal,
                          dt=safety.dt)

    valid_methods = {"rpcbf", "pclf", "clf_qp", "clf_rpcbf_qp", "pure_backup", "None", "pclf_rpcbf_qp", "two_step_pclf_pcbf"}

    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}")

    pclf_methods = {"pclf", "pclf_rpcbf_qp", "two_step_pclf_pcbf"}
    cbf_methods  = {"rpcbf", "pclf_rpcbf_qp", "two_step_pclf_pcbf", "pure_backup", "clf_rpcbf_qp"}
    is_pclf = method in pclf_methods
    is_cbf  = method in cbf_methods

    x_s_init = np.array(x_s, dtype=float)
    x_s = np.array(x_s, dtype=float)
    u_scale = np.array([safety.v_max, safety.om_max])      # normalization for effort: (v/v_max)^2 + (om/om_max)^2
    dt = safety.dt

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
    u_nom_log = []      # reference the filter was measured against (CLF-filtered u for two-step)
    interv_log = []     # ||u_act - u_ref|| >= inter_input
    cbf_on_log = []     # CBF multiplier > 0 (backup: deviation flag)
    effort_log = []     # per-step ||u_act/u_max||^2
    dev_sq_log = []     # per-step ||(u_act - u_ref)/u_max||^2
    J_u_log = []        # running integral of effort                     -> J_u(t)
    J_int_log = []      # running integral of dev_sq over CBF-active steps -> J_int(t)
    J_u = 0.0
    J_int = 0.0

    if make_plots:
        X, Y, V = heatmap_V(safety, goal=init_goal, theta=x_s[2])

    for kk in range(safety.n_steps_sim):
        x_control = x_s.copy()
        d_env = safety.noise_single(env_noise, kk)
        goal = goal_class.call_goal()

        if controller != "clf_nom":
            u_nom = np.asarray(safety.u_nominal(x_control, controller, goal), dtype=float).copy()
            u_nom[0] = np.clip(u_nom[0], safety.v_min, safety.v_max)
            u_nom[1] = np.clip(u_nom[1], -safety.om_max, safety.om_max)
        else:
            u_nom = np.zeros(safety.dyn.nu)

        u_ref = u_nom
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

        elif method == "clf_qp":
            u_act, intervening, solve_dt, V, delta, Vdot, alV = safety.clf_qp(x=x_control, 
                                                                              u_nom=u_nom, 
                                                                              goal=goal, 
                                                                              use_slack=var_slack)
            V_max = V                                   # no preview: the QP acts on V directly
            V_log.append(V)
            V_max_log.append(V_max)
            v_dot_log.append(Vdot)
            alV_log.append(alV)
            delta_log.append(delta if intervening != "infeasible" else np.nan)

        elif method == "clf_rpcbf_qp":
            u_act, intervening, solve_dt, V, delta, h_hmax, Vdot, alV  = safety.clf_rpcbf_qp(x=x_control,
                                                                                            u_nom=u_nom,
                                                                                            goal=goal,
                                                                                            use_slack=var_slack)
            V_max = V                                   # no preview: the QP acts on V directly
            h_now_log.append(safety.cert.h_function(x_control))
            h_hmax_log.append(h_hmax)
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
            u_act, intervening, solve_dt, V_max, delta, h_hmax, Vdot, alV, V, u_ref = safety.two_step_pclf_pcbf(x=x_control, 
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
            d_new = safety.noise_single(rollout_noise_cbf, kk, rng=safety.rng)
            u_act, intervening, solve_dt = backup_safety.safety_Bcbf(x=x_control,
                                                                     u_nom=u_nom, 
                                                                     d_nom=d_new)
            h_now = safety.cert.h_function(x_control)
            h_now_log.append(h_now)
            h_hmax = np.zeros_like(h_now)
            V, V_max, delta = 0.0, 0.0, 0.0
            delta_log.append(delta if intervening != "infeasible" else np.nan)
            h_hmax_log.append(h_hmax)
            V_log.append(V)
            V_max_log.append(V_max)

        elif method == "None":
            u_act = u_nom
            solve_dt = 0.0
            intervening = "None"
            h_hmax, V, V_max, delta = 0.0, 0.0, 0.0, 0.0

        hit, clearance, j = check_collision(x_s, safety.obs_class, robot_radius=0.0)
        if hit and det_collison:
            collision = dict(step=kk, t=kk * safety.dt, obstacle=j, clearance=clearance, state=x_s.copy())
            if print_summary:
                print(f"COLLISION at t = {kk*safety.dt:.2f}s with obstacle {j} "
                    f"(penetration {-clearance:.4f} m). Stopping simulation.")
            break

        # ---- per-step effort / intervention bookkeeping -------------------------------------
        u_act = np.asarray(u_act, dtype=float)
        u_ref = np.asarray(u_ref, dtype=float)
        effort_k = float(np.sum((u_act / u_scale)**2))                 # dimensionless, <= 2
        dev_sq_k = float(np.sum(((u_act - u_ref) / u_scale)**2))       # dimensionless, <= 2
        interv_k = (intervening is True)
        if method == "pure_backup":
            cbf_on_k = interv_k                                        # no QP multiplier available
        elif is_cbf and len(safety.cbf_active_log) == kk + 1:
            cbf_on_k = bool(safety.cbf_active_log[-1])                 # this step's CBF multiplier > 0
        else:
            cbf_on_k = False

        J_u   += effort_k * dt                                          # exact for zero-order-hold inputs
        J_int += dev_sq_k * dt if cbf_on_k else 0.0

        effort_log.append(effort_k)
        dev_sq_log.append(dev_sq_k)
        interv_log.append(interv_k)
        cbf_on_log.append(cbf_on_k)
        J_u_log.append(J_u)
        J_int_log.append(J_int)
        # --------------------------------------------------------------------------------------

        x_s = safety.propagate(x=x_control, u=u_act, d=d_env, goal=goal)
        safety.obs_class.step()                       # obstacle moves with the same dt as the robot

        u_nom_log.append(u_ref.copy())
        obs_log.append(safety.obs_class.pos_now().copy())
        trajectory_actual.append(x_s.copy())
        applied_u.append(u_act.copy())
        d_log.append(np.asarray(d_env).copy())

        if print_summary:
            # last logged CBF row for this step, if this method has a CBF
            hdot_k = safety.hdot_log[-1] if len(safety.hdot_log) == kk + 1 else None
            alh_k  = safety.alh_log[-1]  if len(safety.alh_log)  == kk + 1 else None
            print_step_summary(kk=kk, x=x_s, u_nom=u_nom, u_act=u_act, solve_dt=solve_dt,
                    intervening=intervening, goal=goal, h_values=h_hmax, V=V, delta=delta,
                    V_max=V_max if is_pclf else None,
                    value_name={"rpcbf": "RP-CBF", 
                                "pclf": "P-CLF", 
                                "pure_backup": "Backup",
                                "None": "None",
                                "pclf_rpcbf_qp": "P-CBF-CLF value",
                                "two_step_pclf_pcbf": "two_step P-CBF-CLF value"}.get(method, "CLF"),
                    stop_step=h_stop, dt=safety.dt, hdot=hdot_k, alh=alh_k,
                    J_u=J_u, J_int=J_int if is_cbf else None, cbf_on=cbf_on_k)

        if np.linalg.norm(goal - x_s[:2]) < 0.05:
            if print_summary:
                print("Goal reached!")
            break

        if kk >= early_stop:
            if print_summary:
                print(f"Early stop at step:{kk}")
            break

        if intervening == "infeasible":
            if print_summary:
                print(f"QP stopped due to infeasibility at step {kk}")
            break

    # ---- arrays ---------------------------------------------------------------------------
    trajectory_actual = np.asarray(trajectory_actual)
    applied_u = np.asarray(applied_u).reshape(-1, safety.dyn.nu)
    u_nom_log = np.asarray(u_nom_log).reshape(-1, safety.dyn.nu)
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

    effort_log = np.asarray(effort_log, dtype=float)
    dev_sq_log = np.asarray(dev_sq_log, dtype=float)
    interv_log = np.asarray(interv_log, dtype=bool)
    cbf_on_log = np.asarray(cbf_on_log, dtype=bool)
    J_u_log    = np.asarray(J_u_log, dtype=float)
    J_int_log  = np.asarray(J_int_log, dtype=float)
    N = len(applied_u)
    T_run = N * dt

    # ---- scalar metrics ------------------------------------------------------------------
    J_nom = dt * float(np.sum(np.sum((u_nom_log / u_scale)**2, axis=1))) if N else np.nan
    metrics = dict(
        J_u=J_u,                                   # integrated normalized control effort [s]
        J_nom=J_nom,                               # effort the reference (nominal / CLF-filtered) alone would spend [s]
        effort_increase=(J_u - J_nom) / J_nom if J_nom > 0 else np.nan,   # "filter raised effort by X%"
        J_int=np.nan, T_int=np.nan, frac=np.nan, peak=np.nan,
        J_u_active=np.nan, share_active=np.nan,
        T_run=T_run, n_steps=N,
        reached=bool(N and collision is None and np.linalg.norm(np.asarray(init_goal) - trajectory_actual[-1, :2]) < 0.05),
        collision=collision is not None,
        h_min=float(np.nanmin(h_now_log)) if h_now_log.size else np.nan,   # closest approach (h <= 0 safe)
        n_infeasible=int(np.sum(np.isnan(hdot_log[:, 0]))) if hdot_log.size else 0,
    )
    if print_summary:
        print(f"Controller effort J_u = {J_u:.4f} s   (reference alone J_nom = {J_nom:.4f} s, "
          f"increase {metrics['effort_increase']:+.1%})")

    if is_cbf and N > 0:
        T_int        = dt * float(np.sum(cbf_on_log))                        # filter-active time [s]
        frac         = T_int / T_run
        peak         = float(np.sqrt(dev_sq_log[cbf_on_log].max())) if cbf_on_log.any() else 0.0
        J_u_active   = dt * float(np.sum(effort_log[cbf_on_log]))            # effort spent while filter active [s]
        share_active = J_u_active / J_u if J_u > 0 else np.nan               # true fraction of J_u, in [0, 1]
        metrics.update(J_int=J_int, T_int=T_int, frac=frac, peak=peak,
                       J_u_active=J_u_active, share_active=share_active)
        if print_summary:
            print(f"Filter intervention: J_int={J_int:.4f} s   active {T_int:.2f} s ({frac:.1%} of run)   "
              f"effort during active {share_active:.1%} of J_u   peak_dev={peak:.3f}")

    # ---- parameters this run used (for the Excel 'params' sheet) --------------------------
    params = {**safety.params(),
              "method": method, "var_slack": var_slack, "env_noise": env_noise,
              "x0": list(map(float, x_s_init)), "goal": list(map(float, init_goal)),
              "goal_dyn_op": goal_dyn_op, "goal_motion": goal_motion,
              "det_collison": det_collison, "early_stop": early_stop}

    # ---- plotting --------------------------------------------------------------------------
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
                if print_summary:
                    print(f"[tail l(x_T)/l(x_0)] median {np.median(tl):.3f}  90% {np.percentile(tl,90):.3f}  max {tl.max():.3f}")

        if live_plot:
            plt.close("all")
            live_plot = live_plotter(obs_pos=obs_log[0], obs_radius=safety.obs_class.R_O)
            live_plot.update(trajectory=trajectory_actual, goal=init_goal, obs_traj=obs_log, pause=1e-5)

    return {"states": trajectory_actual, "inputs": applied_u, "u_nom": u_nom_log,
            "t": np.arange(N) * dt,
            # per-step effort / intervention series (all length N, aligned with inputs)
            "effort": effort_log, "dev_sq": dev_sq_log,
            "interv": interv_log, "cbf_on": cbf_on_log,
            "J_u_run": J_u_log, "J_int_run": J_int_log,
            "metrics": metrics,
            "params": params,
            "lam_cbf": np.asarray(safety.lam_cbf_log, dtype=float),
            "cbf_active": np.asarray(safety.cbf_active_log, dtype=bool),
            "h_now": h_now_log, "obs": obs_log,
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
    # *2 "rpcbf", "pclf", "pure_backup", "None", "pclf_rpcbf_qp", "two_step_pclf_pcbf", "clf_rpcbf_qp"

    settings = {
        "method": "clf_rpcbf_qp",        # *2
        "x_s": [1.0, 2.8, 0.0],           # initial position [x, y, yaw]
        "controller": "proportional_policy",  # "clf_nom" or *1
        "h_controller": "backup_policy",       # *1
        "v_controller": "proportional_policy", # *1
        "var_slack": True,
        "rollout_noise_cbf": "Zero",      # Uniform or Zero or BangBang
        "rollout_noise_clf": "Zero",          # Uniform or Zero or BangBang
        "env_noise": "Zero",              # Uniform or Zero or BangBang
        "no_obs": "single",                   # multi or single
        "obs_static": True,
        "init_goal": [4.0, 1.0],              # goal position
        "goal_dyn_op": "static",              # static or sin_y or random
        "goal_motion": "stoc",                # stoc or det (only for sin_y)
        "include_h0": True,
        "include_v0": True,
        "det_collison" : False,
        "make_plots"   : True,
        "early_stop"   : 4000,
        "print_summary": True,
        "T_rollout": 1.5,
        "T_rollout_clf": 3.0,
    }

    results  = run_simulation(**settings)
    save_results_to_excel({settings["controller"]: results}, settings)