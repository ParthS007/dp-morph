#!/usr/bin/env python3
"""
Parse U-Net, U-Net++, or LF-UNet DP flat epsilon table: non-private baseline and flat clipping at ε=8 and ε=200.
Duke: Dice over 7 layers L1--L7. UMN: Dice over L1 (fluid) only.

U-Net: BS=16, unet-duke-base, unet-umn-base, unet-*-dp_flat (bs16_eps8, bs16_eps200).
U-Net++: BS=8, nestedunet-duke-base, nestedunet-umn-base, nestedunet-*-dp_flat (bs8_eps8, bs8_eps200).
LF-UNet: BS=8, lfunet-duke-base, lfunet-umn-base, lfunet-*-dp_flat (bs8_eps8, bs8_eps200).

Usage: python parse_dp_flat_epsilon_table.py [--model unet|nestedunet|lfunet]
"""
import argparse
import ast
import csv
import os

RESULTS_ROOT = "results"
LR = 0.001
MODEL_CONFIG = {
    "unet": {
        "bs": 16,
        "base_duke": "unet-duke-base",
        "base_umn": "unet-umn-base",
        "dp_duke": "unet-duke-dp_flat",
        "dp_umn": "unet-umn-dp_flat",
        "batchsize_ref": "tab:unet-oct-batchsize",
    },
    "nestedunet": {
        "bs": 8,
        "base_duke": "nestedunet-duke-base",
        "base_umn": "nestedunet-umn-base",
        "dp_duke": "nestedunet-duke-dp_flat",
        "dp_umn": "nestedunet-umn-dp_flat",
        "batchsize_ref": "tab:nestedunet-oct-batchsize",
    },
    "lfunet": {
        "bs": 8,
        "base_duke": "lfunet-duke-base",
        "base_umn": "lfunet-umn-base",
        "dp_duke": "lfunet-duke-dp_flat",
        "dp_umn": "lfunet-umn-dp_flat",
        "batchsize_ref": "tab:lfunet-oct-batchsize",
    },
}


def dice_duke_7(dice_all_str):
    arr = ast.literal_eval(dice_all_str)
    return sum(arr[1:8]) / 7.0


def dice_umn_l1(dice_all_str):
    arr = ast.literal_eval(dice_all_str)
    return arr[1]


def load_first_row(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return next(r, None)


def load_first_row_lr(path, lr=LR, batch_size=16):
    if not os.path.isfile(path):
        return None
    first_row = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if first_row is None:
                first_row = row
            try:
                lr_val = float(row.get("Learning_Rate", 0))
                bs_val = int(float(row.get("Batch_Size", 0)))
                if lr_val == lr and bs_val == batch_size:
                    return row
            except (ValueError, TypeError):
                continue
    return first_row


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return None, None
    m = sum(vals) / n
    s = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, s


def main(model="unet"):
    cfg = MODEL_CONFIG[model]
    bs = cfg["bs"]
    base_d, base_u = cfg["base_duke"], cfg["base_umn"]
    dp_d, dp_u = cfg["dp_duke"], cfg["dp_umn"]
    batchsize_ref = cfg["batchsize_ref"]

    duke_np_dice, duke_np_mae = [], []
    umn_np_dice, umn_np_mae = [], []
    for run in [1, 2]:
        for base, dice_list, mae_list in [
            (base_d, duke_np_dice, duke_np_mae),
            (base_u, umn_np_dice, umn_np_mae),
        ]:
            path = os.path.join(
                RESULTS_ROOT, base, f"bs{bs}_run{run}", "test", "results.csv"
            )
            row = load_first_row_lr(path, batch_size=bs)
            if not row and os.path.isfile(path):
                with open(path, newline="") as f:
                    row = next(csv.DictReader(f), None)
            if row:
                da = row.get("Dice_All", "")
                if da:
                    if "duke" in base:
                        dice_list.append(dice_duke_7(da))
                    else:
                        dice_list.append(dice_umn_l1(da))
                    mae_list.append(float(row.get("MAE", 0)))
    np_duke_dice = mean_std(duke_np_dice)
    np_duke_mae = mean_std(duke_np_mae)
    np_umn_dice = mean_std(umn_np_dice)
    np_umn_mae = mean_std(umn_np_mae)

    duke_e8_dice, duke_e8_mae = [], []
    umn_e8_dice, umn_e8_mae = [], []
    for run in [1, 2]:
        path_d = os.path.join(
            RESULTS_ROOT, dp_d, f"bs{bs}_eps8_run{run}", "test", "results.csv"
        )
        path_u = os.path.join(
            RESULTS_ROOT, dp_u, f"bs{bs}_eps8_run{run}", "test", "results.csv"
        )
        row_d = load_first_row(path_d)
        row_u = load_first_row(path_u)
        if row_d and row_d.get("Dice_All"):
            duke_e8_dice.append(dice_duke_7(row_d["Dice_All"]))
            duke_e8_mae.append(float(row_d.get("MAE", 0)))
        if row_u and row_u.get("Dice_All"):
            umn_e8_dice.append(dice_umn_l1(row_u["Dice_All"]))
            umn_e8_mae.append(float(row_u.get("MAE", 0)))
    e8_duke_dice = mean_std(duke_e8_dice)
    e8_duke_mae = mean_std(duke_e8_mae)
    e8_umn_dice = mean_std(umn_e8_dice)
    e8_umn_mae = mean_std(umn_e8_mae)

    duke_e200_dice, duke_e200_mae = [], []
    umn_e200_dice, umn_e200_mae = [], []
    for run in [1, 2]:
        path_d = os.path.join(
            RESULTS_ROOT, dp_d, f"bs{bs}_eps200_run{run}", "test", "results.csv"
        )
        path_u = os.path.join(
            RESULTS_ROOT, dp_u, f"bs{bs}_eps200_run{run}", "test", "results.csv"
        )
        row_d = load_first_row(path_d)
        row_u = load_first_row(path_u)
        if row_d and row_d.get("Dice_All"):
            duke_e200_dice.append(dice_duke_7(row_d["Dice_All"]))
            duke_e200_mae.append(float(row_d.get("MAE", 0)))
        if row_u and row_u.get("Dice_All"):
            umn_e200_dice.append(dice_umn_l1(row_u["Dice_All"]))
            umn_e200_mae.append(float(row_u.get("MAE", 0)))
    e200_duke_dice = mean_std(duke_e200_dice)
    e200_duke_mae = mean_std(duke_e200_mae)
    e200_umn_dice = mean_std(umn_e200_dice)
    e200_umn_mae = mean_std(umn_e200_mae)

    def fmt(m, s, decimals=2):
        if m is None:
            return "—"
        if s is not None and s < 0.005:
            s = 0.00
        return f"{m:.2f} $\\pm$ {s:.2f}" if s is not None else f"{m:.2f}"

    # Ranks: 3 rows, 5=best for Dice and for MAE (lower better)
    all_duke_dice = [np_duke_dice[0], e8_duke_dice[0], e200_duke_dice[0]]
    all_duke_mae = [np_duke_mae[0], e8_duke_mae[0], e200_duke_mae[0]]
    all_umn_dice = [np_umn_dice[0], e8_umn_dice[0], e200_umn_dice[0]]
    all_umn_mae = [np_umn_mae[0], e8_umn_mae[0], e200_umn_mae[0]]

    def rank_three(vals, higher_better=True):
        # Treat None as worst for ranking
        key = lambda i: (
            vals[i]
            if vals[i] is not None
            else (-float("inf") if higher_better else float("inf"))
        )
        idx = sorted(range(3), key=key, reverse=higher_better)
        r = [0, 0, 0]
        r[idx[0]], r[idx[1]], r[idx[2]] = 5, 3, 1
        return r

    rd = rank_three(all_duke_dice, True)
    rm_d = rank_three(all_duke_mae, False)
    ru = rank_three(all_umn_dice, True)
    rm_u = rank_three(all_umn_mae, False)

    np_d_d = fmt(np_duke_dice[0], np_duke_dice[1])
    np_d_m = fmt(np_duke_mae[0], np_duke_mae[1])
    np_u_d = fmt(np_umn_dice[0], np_umn_dice[1])
    np_u_m = fmt(np_umn_mae[0], np_umn_mae[1])
    e8_d_d = fmt(e8_duke_dice[0], e8_duke_dice[1])
    e8_d_m = fmt(e8_duke_mae[0], e8_duke_mae[1])
    e8_u_d = fmt(e8_umn_dice[0], e8_umn_dice[1])
    e8_u_m = fmt(e8_umn_mae[0], e8_umn_mae[1])
    e200_d_d = fmt(e200_duke_dice[0], e200_duke_dice[1])
    e200_d_m = fmt(e200_duke_mae[0], e200_duke_mae[1])
    e200_u_d = fmt(e200_umn_dice[0], e200_umn_dice[1])
    e200_u_m = fmt(e200_umn_mae[0], e200_umn_mae[1])

    print(f"% LaTeX for Table (BS={bs}, Duke 7-layer Dice, UMN L1 Dice)")
    print(
        f"Non-private (Table~\\ref{{{batchsize_ref}}}) & \\heatcell{{{rd[0]}}}{{{np_d_d}}} & \\heatcell{{{rm_d[0]}}}{{{np_d_m}}} & \\heatcell{{{ru[0]}}}{{{np_u_d}}} & \\heatcell{{{rm_u[0]}}}{{{np_u_m}}} \\\\"
    )
    print("\\midrule")
    print(
        f"Flat clip. $\\varepsilon = 8$  & \\heatcell{{{rd[1]}}}{{{e8_d_d}}} & \\heatcell{{{rm_d[1]}}}{{{e8_d_m}}} & \\heatcell{{{ru[1]}}}{{{e8_u_d}}} & \\heatcell{{{rm_u[1]}}}{{{e8_u_m}}} \\\\"
    )
    print(
        f"Flat clip. $\\varepsilon = 200$ & \\heatcell{{{rd[2]}}}{{{e200_d_d}}} & \\heatcell{{{rm_d[2]}}}{{{e200_d_m}}} & \\heatcell{{{ru[2]}}}{{{e200_u_d}}} & \\heatcell{{{rm_u[2]}}}{{{e200_u_m}}} \\\\"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse DP flat epsilon table.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIG),
        default="unet",
        help="unet, nestedunet, or lfunet",
    )
    args = parser.parse_args()
    main(model=args.model)
