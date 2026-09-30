"""
Disturbance sweep for the P-CLF unicycle.

Batches (each = 5 circles x 18 start points = 90 runs):
    method="pclf" x controller in {proportional, backup, constant} x env_noise in {Uniform, BangBang}
    method="None" x controller=proportional (benchmark)         x env_noise in {Uniform, BangBang}
-> 8 batches, 720 runs.  Rollout noise always matches env noise.

Start points: equidistant circles of radius R around the goal, restricted to x >= 0, y >= 0,
heading facing the goal.  Each batch gets its own folder, e.g.  sweep_results/pclf_uniform_backup_90/
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

from Main import run_simulation
from plotter.Save_results_sweep import save_sweep_result, extract_filter_params

# ----------------------------------------------------------------------------- sweep definition
GOAL = np.array([4.0, 1.0])
RADII = [0.5, 1.0, 2.0, 3.0, 4.0]
N_PER_CIRCLE = 18
ENV_NOISES = ["Uniform", "BangBang"]
CONTROLLER_SETS = [                      # (method, controller)
    ("pclf", "proportional_policy"),
    ("pclf", "backup_policy"),
    ("pclf", "constant_policy"),
    ("None", "proportional_policy"),     # benchmark
]
ROOT_DIR = "sweep_results"
WORKERS = 40

BASE_SETTINGS = {
    "method": "pclf",
    "x_s": [0.0, 1.0, 0.0],
    "controller": "proportional_policy",
    "h_controller": "backup_policy",
    "v_controller": "proportional_policy",
    "var_slack": False,
    "rollout_noise": "Zero",
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
}


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
                        "rollout_noise": env_noise,        # match the environment
                        "env_noise": env_noise,
                        "x_s": x_s,
                    }
                    jobs.append({"batch": name, "run_id": run_id, "circle": ci, "radius": r,
                                 "point": pi, "settings": settings})
                    run_id += 1
    return jobs


# ----------------------------------------------------------------------------- worker
def run_one(job):
    try:
        results = run_simulation(**job["settings"])
        # keep the scalar filter parameters, drop the policy_filter object itself
        # (pickling it back through the pool 720 times is slow and unnecessary)
        results["filter_params"] = extract_filter_params(results.pop("safety", None))
        return job, results, None
    except Exception:
        return job, None, traceback.format_exc()


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
                        "method", "controller", "env_noise", "rollout_noise"])
            for j in jobs:
                if j["batch"] != b:
                    continue
                s = j["settings"]
                w.writerow([j["run_id"], j["circle"], j["radius"], j["point"], *s["x_s"],
                            s["method"], s["controller"], s["env_noise"], s["rollout_noise"]])

    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        pending = {executor.submit(run_one, j): j for j in jobs}

        with tqdm(total=len(jobs), desc="Simulations", unit="run", smoothing=0,
                  bar_format="{desc}: {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Remaining: {remaining}") as bar:
            for fut in as_completed(pending):
                job, results, err = fut.result()
                out_dir = os.path.join(ROOT_DIR, job["batch"])
                if err is not None:
                    failures.append((job["batch"], job["run_id"], err))
                    with open(os.path.join(out_dir, f"FAILED_run_{job['run_id']:03d}.txt"), "w") as f:
                        f.write(err)
                else:
                    save_sweep_result(results, job["settings"], out_dir, job["run_id"],
                                      job_meta={"circle": job["circle"], "radius": job["radius"],
                                                "point": job["point"]},
                                      filter_params=results.pop("filter_params", {}))
                    del results
                bar.update(1)

    print(f"done. {len(jobs) - len(failures)} ok, {len(failures)} failed")
    for b, rid, _ in failures:
        print(f"  FAILED {b} run {rid}")
