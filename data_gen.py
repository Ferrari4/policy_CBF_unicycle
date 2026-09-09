import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from Main import run_simulation
from plotter.Data_generator import save_results_to_excel 


def run_one(settings):
    results = run_simulation(**settings)
    return settings, results


if __name__ == "__main__":
    settings = {
        "method": "pclf_rpcbf_qp",
        "x_s": [0.0, 1.0, 0.0],
        "controller": "proportional_policy",
        "h_controller": "backup_policy",
        "v_controller": "proportional_policy",
        "var_slack": True,
        "rollout_noise": "Zero",
        "env_noise": "Zero",
        "no_obs": "single",
        "init_goal": [4.0, 1.0],
        "goal_dyn_op": "static",
        "goal_motion": "stoc",
        "include_h0": True,
        "include_v0": True,
    }

    all_settings = [
        {
            **settings,
            "x_s": [0.0, 1.0 + 0.1 * i, 0.0],
        }
        for i in range(20)
    ]

    workers = 20

    # Everything below is STILL inside the main guard.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(run_one, s): i
            for i, s in enumerate(all_settings)
        }

        with tqdm(
            total=len(all_settings),
            desc="Simulations",
            unit="run",
            smoothing=0,
            bar_format=(
                "{desc}: {n_fmt}/{total_fmt} | "
                "Elapsed: {elapsed} | Remaining: {remaining}"
            ),
        ) as progress:
            for future in as_completed(pending):
                run_id = pending.pop(future)
                run_settings, results = future.result()

                save_results_to_excel(
                    {run_settings["controller"]: results},
                    run_settings,
                    run_id=run_id,
                )

                del results
                progress.update(1)