"""
Disturbance sweep for the P-CLF unicycle.

Batches (each = len(RADII) circles x N_PER_CIRCLE start points):
    method="pclf" x controller in {proportional, backup, constant} x env_noise in ENV_NOISES
    method="None" x controller=proportional (benchmark)         x env_noise in ENV_NOISES
Rollout noise is fixed in BASE_SETTINGS (CLF preview on nominal rollouts, rollout_noise_clf="Zero";
CBF preview rollout_noise_cbf="Uniform"); the environment disturbance is env_noise.

Start points: equidistant circles of radius R around the goal, restricted to x >= 0, y >= 0,
heading facing the goal.  Each batch gets its own folder, e.g.  sweep_results_pclf/pclf_bangbang_backup_250/
containing manifest.csv (start points), params.csv (filter parameters), summary.csv (one row per run:
settings + metrics), and the per-run files written by save_sweep_result.

Failures are isolated per run: a simulation error writes FAILED_run_XXX.txt, a per-run save error writes
FAILED_save_run_XXX.txt (the run's summary row is still kept).  Neither stops the sweep.
"""
import os
# one BLAS thread per worker process; must be set before numpy is imported anywhere
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from run_sim import run_simulation
from plotter.Save_results_sweep import save_sweep_result

# ----------------------------------------------------------------------------- sweep definition
GOAL = np.array([4.0, 1.0])
RADII = [0.5, 1.0, 2.0, 3.0, 4.0]
N_PER_CIRCLE = 50
ENV_NOISES = ["BangBang"]
CONTROLLER_SETS = [                      # (method, controller)
    ("two_step_pclf_pcbf", "proportional_policy"),
    ("two_step_pclf_pcbf", "backup_policy"),
    ("two_step_pclf_pcbf", "constant_policy"),
    ("pure_backup", "proportional_policy"),     # benchmark
    # CBF-intervention comparison (Table 1 of the plan) - uncomment to include:
    # ("rpcbf",              "proportional_policy"),
    # ("clf_rpcbf_qp",       "proportional_policy"),
    # ("pclf_rpcbf_qp",      "proportional_policy"),
    # ("two_step_pclf_pcbf", "proportional_policy"),
    # ("pure_backup",        "proportional_policy"),
]
ROOT_DIR = "sweep_results_pclf_rpcbf_two"
WORKERS = 22

BASE_SETTINGS = {
    "method": "pclf",
    "x_s": [0.0, 1.0, 0.0],
    "controller": "proportional_policy",
    "h_controller": "backup_policy",
    "v_controller": "proportional_policy",
    "var_slack": True,
    "rollout_noise_cbf": "Uniform",
    "rollout_noise_clf": "Zero",
    "env_noise": "Zero",
    "no_obs": "single",
    "obs_static": True,
    "init_goal": GOAL.tolist(),
    "goal_dyn_op": "static",
    "goal_motion": "stoc",
    "include_h0": True,
    "include_v0": True,
    "det_collison": True,      # stop + record the collision step (needed to score benchmark vs P-CLF)
    "make_plots": False,
    "early_stop": 4000,        # 40 s at dt = 0.01
    "print_summary": False,    # no per-step console output inside the pool
    "T_rollout": 1.5,          # None -> policy_filter defaults (1.5 s CBF, 3.0 s CLF)
    "T_rollout_clf": 3.0,
}

# columns of summary.csv: job info + the settings that vary + every metric from run_simulation
SUMMARY_SETTINGS = ["method", "controller", "h_controller", "v_controller", "var_slack",
                    "rollout_noise_cbf", "rollout_noise_clf", "env_noise", "T_rollout", "T_rollout_clf"]


# ----------------------------------------------------------------------------- start points
def circle_start_points(goal, r, n):
    """n start states on the circle of radius r about goal, inside x>=0, y>=0, facing the goal."""
    gx, gy = goal
    phi = np.linspace(0.0, 2 * np.pi, 3600, endpoint=False)
    x, y = gx + r * np.cos(phi), gy + r * np.sin(phi)
    ok = (x >= 0.0) & (y >= 0.0)

    if ok.all():                                            # whole circle admissible
        phis = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    else:
        # admissible set is one contiguous arc; rotate so it starts at the first False->True edge
        edge = np.flatnonzero(~ok[:-1] & ok[1:])
        start = edge[0] + 1 if len(edge) else np.flatnonzero(ok)[0]
        rolled = np.roll(ok, -start)
        arc_len = np.flatnonzero(~rolled)[0]                # number of admissible samples
        phi_lo = phi[start]
        phi_hi = phi[start] + arc_len * (2 * np.pi / len(phi))
        phis = np.linspace(phi_lo, phi_hi, n + 2)[1:-1]     # n interior points, none on the axes

    pts = []
    for p in phis:
        px, py = gx + r * np.cos(p), gy + r * np.sin(p)
        th = np.arctan2(gy - py, gx - px)                   # face the goal
        pts.append([float(px), float(py), float(th)])
    return pts


def batch_name(method, env_noise, controller, n_runs):
    ctrl = controller.replace("_policy", "")
    return f"{method.lower()}_{env_noise.lower()}_{ctrl}_{n_runs}"


def build_jobs():
    jobs = []
    for method, controller in CONTROLLER_SETS:
        for env_noise in ENV_NOISES:
            name = batch_name(method, env_noise, controller, len(RADII) * N_PER_CIRCLE)
            run_id = 0
            for ci, r in enumerate(RADII):
                for pi, x_s in enumerate(circle_start_points(GOAL, r, N_PER_CIRCLE)):
                    settings = {
                        **BASE_SETTINGS,
                        "method": method,
                        "controller": controller,
                        "env_noise": env_noise,
                        "x_s": x_s,
                    }
                    jobs.append({"batch": name, "run_id": run_id, "circle": ci, "radius": r,
                                 "point": pi, "settings": settings})
                    run_id += 1
    return jobs


# ----------------------------------------------------------------------------- worker
def _scalar(v):
    """Excel/csv-safe scalar: sequences -> 'a,b,c' string, numpy scalars -> python, dicts -> str."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return ",".join(f"{float(x):.6g}" for x in np.asarray(v).ravel())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return str(v)
    return v


def run_one(job):
    try:
        results = run_simulation(**job["settings"])
        # drop the policy_filter object: results["params"] already holds every scalar setting
        # (pickling the object back through the pool hundreds of times is slow and unnecessary)
        results.pop("safety", None)
        # make every parameter a plain scalar/string so no writer downstream can choke on a list
        results["params"] = {k: _scalar(v) for k, v in results.get("params", {}).items()}
        return job, results, None
    except Exception:
        return job, None, traceback.format_exc()


def summary_row(job, results):
    s = job["settings"]
    row = {"run_id": job["run_id"], "circle": job["circle"], "radius": job["radius"], "point": job["point"],
           "x0": s["x_s"][0], "y0": s["x_s"][1], "theta0": s["x_s"][2]}
    row.update({k: _scalar(s.get(k)) for k in SUMMARY_SETTINGS})
    row.update({k: _scalar(v) for k, v in results.get("metrics", {}).items()})
    return row


def write_summary(out_dir, rows):
    """summary.csv for one batch, sorted by run_id, union of all metric columns."""
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["run_id"])
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_params(out_dir, params):
    with open(os.path.join(out_dir, "params.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in params.items():
            w.writerow([k, _scalar(v)])


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    jobs = build_jobs()
    batches = sorted({j["batch"] for j in jobs})
    print(f"{len(jobs)} runs in {len(batches)} batches, {WORKERS} workers")

    # folders + manifest of start points per batch
    for b in batches:
        os.makedirs(os.path.join(ROOT_DIR, b), exist_ok=True)
        with open(os.path.join(ROOT_DIR, b, "manifest.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "circle", "radius", "point", "x0", "y0", "theta0",
                        "method", "controller", "env_noise", "rollout_noise_cbf", "rollout_noise_clf"])
            for j in jobs:
                if j["batch"] != b:
                    continue
                s = j["settings"]
                w.writerow([j["run_id"], j["circle"], j["radius"], j["point"], *s["x_s"],
                            s["method"], s["controller"], s["env_noise"],
                            s["rollout_noise_cbf"], s["rollout_noise_clf"]])

    sim_failures = []          # (batch, run_id): run_simulation raised -> no data for this run
    save_failures = []         # (batch, run_id): save_sweep_result raised -> summary row kept, per-run file missing
    summaries = {b: [] for b in batches}
    params_written = set()

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        pending = {executor.submit(run_one, j): j for j in jobs}

        with tqdm(total=len(jobs), desc="Simulations", unit="run", smoothing=0,
                  bar_format="{desc}: {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Remaining: {remaining}") as bar:
            for fut in as_completed(pending):
                job, results, err = fut.result()
                out_dir = os.path.join(ROOT_DIR, job["batch"])
                rid = job["run_id"]

                if err is not None:
                    sim_failures.append((job["batch"], rid))
                    with open(os.path.join(out_dir, f"FAILED_run_{rid:03d}.txt"), "w") as f:
                        f.write(err)
                    bar.update(1)
                    continue

                # 1) summary row first: it only needs results["metrics"], so it survives a save failure
                try:
                    summaries[job["batch"]].append(summary_row(job, results))
                except Exception:
                    with open(os.path.join(out_dir, f"FAILED_summary_run_{rid:03d}.txt"), "w") as f:
                        f.write(traceback.format_exc())

                # 2) per-run file; isolated so one bad value cannot kill the sweep
                filter_params = results.get("params", {})
                try:
                    save_sweep_result(results, job["settings"], out_dir, rid,
                                      job_meta={"circle": job["circle"], "radius": job["radius"],
                                                "point": job["point"]},
                                      filter_params=filter_params)
                except Exception:
                    save_failures.append((job["batch"], rid))
                    with open(os.path.join(out_dir, f"FAILED_save_run_{rid:03d}.txt"), "w") as f:
                        f.write(traceback.format_exc())

                # 3) one params.csv per batch (settings shared by every run in the batch)
                if job["batch"] not in params_written and filter_params:
                    try:
                        write_params(out_dir, filter_params)
                        params_written.add(job["batch"])
                    except Exception:
                        with open(os.path.join(out_dir, "FAILED_params.txt"), "w") as f:
                            f.write(traceback.format_exc())

                del results
                bar.update(1)

    for b, rows in summaries.items():
        write_summary(os.path.join(ROOT_DIR, b), rows)

    n_ok = len(jobs) - len(sim_failures)
    print(f"done. {n_ok} simulated, {len(sim_failures)} simulation failures, {len(save_failures)} save failures")
    for b, rid in sim_failures:
        print(f"  FAILED sim  {b} run {rid}")
    for b, rid in save_failures:
        print(f"  FAILED save {b} run {rid}  (summary row kept)")

    # quick batch-level readout
    for b, rows in summaries.items():
        if not rows:
            continue
        n = len(rows)
        reached = sum(bool(r.get("reached")) for r in rows)
        coll = sum(bool(r.get("collision")) for r in rows)
        J_u = np.nanmean([float(r["J_u"]) for r in rows if r.get("J_u") is not None])
        J_int = [float(r["J_int"]) for r in rows if r.get("J_int") not in (None, "nan")]
        J_int_s = f"  J_int {np.nanmean(J_int):.3f}" if J_int and not np.all(np.isnan(J_int)) else ""
        print(f"  {b:40s} reached {reached:3d}/{n}  collided {coll:3d}/{n}  J_u {J_u:.3f}{J_int_s}")
