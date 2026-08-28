from Main import run_simulation

import numpy as np
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font

def to_2d(arr):
    arr = np.asarray(arr)

    if arr.size == 0:
        return np.empty((0, 1))

    return arr.reshape(len(arr), -1)

def save_results_to_excel(all_results, settings, filename=None):

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_{timestamp}.xlsx"

    wb = Workbook()

    # Remove the automatically created default sheet
    default_sheet = wb.active
    wb.remove(default_sheet)

    for policy_name, results in all_results.items():

        sheet_name = policy_name[:31]
        ws = wb.create_sheet(title=sheet_name)

        # -------------------------------------------------
        # Simulation settings
        # -------------------------------------------------
        metadata = [
            ["Simulation Settings", ""],
            ["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Method", settings["method"]],
            ["Initial state", str(settings["x_s"])],
            ["Controller", policy_name],
            ["h_controller", policy_name],
            ["v_controller", policy_name],
            ["Rollout noise", settings["rollout_noise"]],
            ["Environment noise", settings["env_noise"]],
            ["Obstacle configuration", settings["no_obs"]],
        ]

        for row in metadata:
            ws.append(row)

        ws["A1"].font = Font(bold=True)

        # Leave one blank row
        ws.append([])

        # -------------------------------------------------
        # Get data
        # -------------------------------------------------
        states = np.asarray(results["states"])
        inputs = np.asarray(results["inputs"])
        h_now = np.asarray(results["h_now"])
        h_hmax = np.asarray(results["h_hmax"])
        V = np.asarray(results["V"])
        delta = np.asarray(results["delta"])

        # Make arrays consistently 2-D
        states = to_2d(results["states"])
        inputs = to_2d(results["inputs"])
        h_now = to_2d(results["h_now"])
        h_hmax = to_2d(results["h_hmax"])
        V = to_2d(results["V"])
        delta = to_2d(results["delta"])
    
        # -------------------------------------------------
        # Some arrays may have different lengths
        # states often has N+1 while inputs has N
        # -------------------------------------------------
        N = min(len(states), len(inputs))

        states = states[:N]
        inputs = inputs[:N]
        h_now = h_now[:N]
        h_hmax = h_hmax[:N]
        V = V[:N]
        delta = delta[:N]

        # -------------------------------------------------
        # Headers
        # -------------------------------------------------
        headers = ["step"]

        headers += [
            f"state_{i}"
            for i in range(states.shape[1])
        ]

        headers += [
            f"input_{i}"
            for i in range(inputs.shape[1])
        ]

        headers += [
            f"h_now_{i}"
            for i in range(h_now.shape[1])
        ]

        headers += [
            f"h_hmax_{i}"
            for i in range(h_hmax.shape[1])
        ]

        headers += [
            f"V_{i}"
            for i in range(V.shape[1])
        ]

        headers += [
            f"delta_{i}"
            for i in range(delta.shape[1])
        ]

        ws.append(headers)

        header_row = ws.max_row

        for cell in ws[header_row]:
            cell.font = Font(bold=True)

        # -------------------------------------------------
        # Save numerical data
        # -------------------------------------------------
        for k in range(N):

            row = [k]

            row += states[k].tolist()
            row += inputs[k].tolist()

            row += h_now[k].tolist() if k < len(h_now) else [None]
            row += h_hmax[k].tolist() if k < len(h_hmax) else [None]
            row += V[k].tolist() if k < len(V) else [None]
            row += delta[k].tolist() if k < len(delta) else [None]

            ws.append(row)

        # Freeze rows above numerical data
        ws.freeze_panes = f"A{header_row + 1}"

        # Basic column sizing
        ws.column_dimensions["A"].width = 12
        ws.column_dimensions["B"].width = 22

    wb.save(filename)

    print(f"Saved results to: {filename}")

if __name__ == "__main__":

    policy_lib = [
        "constant_policy",
        "backup_policy",
        "proportional_policy",
        "random_policy"
    ]

    settings = {
        "method": "rpcbf",
        "x_s": [0.5, 2.0, 0.0],
        "rollout_noise": "Zero",
        "env_noise": "Zero",
        "no_obs": "single"
    }

    all_results = {}

    for policy in policy_lib:

        results = run_simulation(
            method=settings["method"],
            x_s=settings["x_s"],
            controller=policy,
            h_controller="backup_policy",
            v_controller="proportional_policy",
            rollout_noise=settings["rollout_noise"],
            env_noise=settings["env_noise"],
            no_obs=settings["no_obs"]
        )

        all_results[policy] = results

    save_results_to_excel(
        all_results,
        settings
    )

