#!/usr/bin/env python3
"""
Parse unet or nestedunet ablation test results to build the LR/WD ablation table.
- Duke: Dice over 7 layers L1--L7 (indices 1:8 of Dice_All). Mean = sum(arr[1:8]) / 7.
- UMN: Only two layers are actually present (L0 background, L1 fluid). The Dice_All array
  may have 9 elements like Duke, but we only use indices 0 and 1. When calculating any
  mean for UMN, divide only by the number of layers that are actually present (2), never
  by 7 or 9. Zero or placeholder values in other indices are ignored.
  - dice_umn_l1: fluid only (index 1), no mean.
  - dice_umn_two_layers: mean over the two present layers = (arr[0] + arr[1]) / 2.

CSV has no Weight_Decay column; row order per (BS, run) is assumed:
  row 0: LR=1e-3, WD=1e-4
  row 1: LR=1e-3, WD=1e-9
  row 2: LR=5e-4, WD=1e-4
  row 3: LR=5e-4, WD=1e-9

Usage: python parse_ablation_table.py [--model unet|nestedunet|lfunet]
"""
import argparse
import ast
import csv
import os
from collections import defaultdict

RESULTS_ROOT = "results"
MODEL_DATASETS = {
    "unet": ["unet-duke-base", "unet-umn-base"],
    "nestedunet": ["nestedunet-duke-base", "nestedunet-umn-base"],
    "lfunet": ["lfunet-duke-base", "lfunet-umn-base"],
}
# Row index in LR_WD_ORDER used for batch-size table (0 = 1e-3/1e-4, 1 = 1e-3/1e-9, 2 = 5e-4/1e-4, 3 = 5e-4/1e-9)
MODEL_CHOSEN_ROW = {"unet": 0, "nestedunet": 0, "lfunet": 1}
# (LR, WD) order matching ablation script
LR_WD_ORDER = [
    (0.001, 0.0001),  # 1e-3, 1e-4
    (0.001, 1e-9),  # 1e-3, 1e-9
    (0.0005, 0.0001),  # 5e-4, 1e-4
    (0.0005, 1e-9),  # 5e-4, 1e-9
]


def dice_duke_7(dice_all_str):
    """Duke: mean of layers 1--7 (0-based indices 1:8)."""
    arr = ast.literal_eval(dice_all_str)
    return sum(arr[1:8]) / 7.0


def _umn_dice_arr(dice_all_str):
    """UMN: only indices 0 and 1 are the two layers actually present (L0, L1)."""
    arr = ast.literal_eval(dice_all_str)
    return arr


def dice_umn_l1(dice_all_str):
    """UMN: fluid only (0-based index 1). Single value, no mean."""
    arr = _umn_dice_arr(dice_all_str)
    return arr[1]


def dice_umn_two_layers(dice_all_str):
    """UMN: mean over the two layers that are actually present (L0, L1). Divide only by
    the count of non-zero values so we don't count absent/zero layers in the denominator.
    """
    arr = _umn_dice_arr(dice_all_str)
    values_present = [v for v in [arr[0], arr[1]] if v is not None and v != 0]
    if not values_present:
        return 0.0
    return sum(values_present) / len(values_present)


def load_test_dice(dataset, batch_size, run_number):
    """Load test results; return list of 4 Dice values in LR_WD_ORDER (by row order)."""
    path = os.path.join(
        RESULTS_ROOT, dataset, f"bs{batch_size}_run{run_number}", "test", "results.csv"
    )
    if not os.path.isfile(path):
        return None
    use_l1_only = "umn" in dataset.lower()
    values = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            dice_all = row.get("Dice_All", "")
            if not dice_all:
                continue
            if use_l1_only:
                values.append(dice_umn_l1(dice_all))
            else:
                values.append(dice_duke_7(dice_all))
    return values if len(values) == 4 else None


def load_test_rows(dataset, batch_size, run_number):
    """Load test results; return list of 4 dicts {dice, mae, loss} in LR_WD_ORDER."""
    path = os.path.join(
        RESULTS_ROOT, dataset, f"bs{batch_size}_run{run_number}", "test", "results.csv"
    )
    if not os.path.isfile(path):
        return None
    use_l1_only = "umn" in dataset.lower()
    rows_out = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            dice_all = row.get("Dice_All", "")
            if not dice_all:
                continue
            dice = dice_umn_l1(dice_all) if use_l1_only else dice_duke_7(dice_all)
            try:
                mae = float(row.get("MAE", 0))
                loss = float(row.get("Test_Loss", 0))
            except (TypeError, ValueError):
                mae = loss = 0.0
            rows_out.append({"dice": dice, "mae": mae, "loss": loss})
    return rows_out if len(rows_out) == 4 else None


def main(datasets, chosen_lr_wd_row=0):
    """Run ablation table parsing for the given dataset pair [duke_name, umn_name]. chosen_lr_wd_row = row index in LR_WD_ORDER for batch-size block."""
    duke_name, umn_name = datasets[0], datasets[1]
    # Collect per (dataset, lr_wd_index): {bs: list of run Dice values}
    data = defaultdict(lambda: defaultdict(list))
    for dataset in datasets:
        for bs in [8, 16]:
            for run in [1, 2]:
                row = load_test_dice(dataset, bs, run)
                if row is None:
                    print(f"Missing or incomplete: {dataset} bs{bs} run{run}")
                    continue
                for lr_wd_idx in range(4):
                    data[(dataset, lr_wd_idx)][bs].append(row[lr_wd_idx])

    def mean_std(vals):
        n = len(vals)
        if n == 0:
            return None, None
        m = sum(vals) / n
        if n < 2:
            return m, 0.0
        var = sum((x - m) ** 2 for x in vals) / (n - 1)
        return m, var**0.5

    print(f"Duke (7 layers L1--L7) [{duke_name}]:")
    for lr_wd_idx, (lr, wd) in enumerate(LR_WD_ORDER):
        bs8 = data[(duke_name, lr_wd_idx)].get(8, [])
        bs16 = data[(duke_name, lr_wd_idx)].get(16, [])
        m8, s8 = mean_std(bs8)
        m16, s16 = mean_std(bs16)
        print(
            f"  LR={lr}, WD={wd}: BS=8  {m8:.4f} ± {s8:.4f}   BS=16  {m16:.4f} ± {s16:.4f}"
        )

    print(f"\nUMN (L1 fluid only) [{umn_name}]:")
    for lr_wd_idx, (lr, wd) in enumerate(LR_WD_ORDER):
        bs8 = data[(umn_name, lr_wd_idx)].get(8, [])
        bs16 = data[(umn_name, lr_wd_idx)].get(16, [])
        m8, s8 = mean_std(bs8)
        m16, s16 = mean_std(bs16)
        print(
            f"  LR={lr}, WD={wd}: BS=8  {m8:.4f} ± {s8:.4f}   BS=16  {m16:.4f} ± {s16:.4f}"
        )

    # LaTeX-style rounded for table (0.xx ± 0.xx)
    print("\n--- LaTeX table values (Duke 7-layer, UMN L1) ---")

    def r(m, s):
        if m is None:
            return "—"
        return f"{m:.2f} $\\pm$ {s:.2f}"

    for lr_wd_idx, (lr, wd) in enumerate(LR_WD_ORDER):
        d_bs8 = mean_std(data[(duke_name, lr_wd_idx)].get(8, []))
        d_bs16 = mean_std(data[(duke_name, lr_wd_idx)].get(16, []))
        u_bs8 = mean_std(data[(umn_name, lr_wd_idx)].get(8, []))
        u_bs16 = mean_std(data[(umn_name, lr_wd_idx)].get(16, []))
        lr_tex = "1 \\times 10^{-3}" if lr == 0.001 else "5 \\times 10^{-4}"
        wd_tex = "1 \\times 10^{-4}" if wd == 0.0001 else "1 \\times 10^{-9}"
        print(
            f"  {lr_tex} & {wd_tex} & {r(*d_bs8)} & {r(*d_bs16)} & {r(*u_bs8)} & {r(*u_bs16)}"
        )

    # Batch-size table (chosen LR/WD row)
    lr_chosen, wd_chosen = LR_WD_ORDER[chosen_lr_wd_row]
    batch_data = defaultdict(
        lambda: defaultdict(list)
    )  # (dataset, bs) -> {dice, mae, loss} lists
    for dataset in datasets:
        for bs in [8, 16]:
            for run in [1, 2]:
                rows = load_test_rows(dataset, bs, run)
                if rows is None:
                    continue
                batch_data[(dataset, bs)]["dice"].append(rows[chosen_lr_wd_row]["dice"])
                batch_data[(dataset, bs)]["mae"].append(rows[chosen_lr_wd_row]["mae"])
                batch_data[(dataset, bs)]["loss"].append(rows[chosen_lr_wd_row]["loss"])
    print(
        f"\n--- Batch-size table (LR {lr_chosen}, WD {wd_chosen}, row {chosen_lr_wd_row}): Dice, MAE, Loss ---"
    )
    for bs in [8, 16]:
        d_dice = mean_std(batch_data[(duke_name, bs)]["dice"])
        d_mae = mean_std(batch_data[(duke_name, bs)]["mae"])
        d_loss = mean_std(batch_data[(duke_name, bs)]["loss"])
        u_dice = mean_std(batch_data[(umn_name, bs)]["dice"])
        u_mae = mean_std(batch_data[(umn_name, bs)]["mae"])
        u_loss = mean_std(batch_data[(umn_name, bs)]["loss"])
        fmt_mae = lambda m, s: (
            f"{m:.3f} $\\pm$ {s:.3f}" if m is not None and s is not None else "—"
        )
        fmt_loss = r
        print(
            f"  BS={bs}: Duke Dice {r(*d_dice)} MAE {fmt_mae(*d_mae)} Loss {fmt_loss(*d_loss)}  |  UMN Dice {r(*u_dice)} MAE {fmt_mae(*u_mae)} Loss {fmt_loss(*u_loss)}"
        )
    print("\n--- LaTeX batch-size table rows (heatcell rank by column: 5=best) ---")

    # Assign heatcell rank 1-5 per column (Duke Dice, Duke MAE, Duke Loss, UMN Dice, UMN MAE, UMN Loss)
    def rank_heatcell(vals, higher_better):
        """vals = [(m,s), ...] for the 2 rows (BS=8, BS=16). Return (rank_bs8, rank_bs16)."""
        means = [v[0] for v in vals if v[0] is not None]
        if len(means) != 2:
            return (5, 5)
        order = sorted(range(2), key=lambda i: means[i], reverse=higher_better)
        ranks = [0, 0]
        for rk, idx in enumerate(order):
            ranks[idx] = 5 - rk if len(order) == 2 else (4, 3)[idx]
        return tuple(ranks)

    d_dice_8, d_dice_16 = mean_std(batch_data[(duke_name, 8)]["dice"]), mean_std(
        batch_data[(duke_name, 16)]["dice"]
    )
    d_mae_8, d_mae_16 = mean_std(batch_data[(duke_name, 8)]["mae"]), mean_std(
        batch_data[(duke_name, 16)]["mae"]
    )
    d_loss_8, d_loss_16 = mean_std(batch_data[(duke_name, 8)]["loss"]), mean_std(
        batch_data[(duke_name, 16)]["loss"]
    )
    u_dice_8, u_dice_16 = mean_std(batch_data[(umn_name, 8)]["dice"]), mean_std(
        batch_data[(umn_name, 16)]["dice"]
    )
    u_mae_8, u_mae_16 = mean_std(batch_data[(umn_name, 8)]["mae"]), mean_std(
        batch_data[(umn_name, 16)]["mae"]
    )
    u_loss_8, u_loss_16 = mean_std(batch_data[(umn_name, 8)]["loss"]), mean_std(
        batch_data[(umn_name, 16)]["loss"]
    )
    rd = rank_heatcell([d_dice_8, d_dice_16], True)
    rm_d = rank_heatcell([d_mae_8, d_mae_16], False)
    rl_d = rank_heatcell([d_loss_8, d_loss_16], False)
    ru = rank_heatcell([u_dice_8, u_dice_16], True)
    rm_u = rank_heatcell([u_mae_8, u_mae_16], False)
    rl_u = rank_heatcell([u_loss_8, u_loss_16], False)

    def mae_fmt(m, s):
        if m is None:
            return "—"
        if m < 0.001:
            return f"{m:.4f} $\\pm$ {s:.4f}" if s is not None else f"{m:.4f}"
        return f"{m:.3f} $\\pm$ {s:.3f}" if s is not None else f"{m:.3f}"

    print(
        f"  8  & \\heatcell{{{rd[0]}}}{{{r(*d_dice_8)}}} & \\heatcell{{{rm_d[0]}}}{{{mae_fmt(*d_mae_8)}}} & \\heatcell{{{rl_d[0]}}}{{{r(*d_loss_8)}}} & \\heatcell{{{ru[0]}}}{{{r(*u_dice_8)}}} & \\heatcell{{{rm_u[0]}}}{{{mae_fmt(*u_mae_8)}}} & \\heatcell{{{rl_u[0]}}}{{{r(*u_loss_8)}}} \\\\"
    )
    print(
        f"  16 & \\heatcell{{{rd[1]}}}{{{r(*d_dice_16)}}} & \\heatcell{{{rm_d[1]}}}{{{mae_fmt(*d_mae_16)}}} & \\heatcell{{{rl_d[1]}}}{{{r(*d_loss_16)}}} & \\heatcell{{{ru[1]}}}{{{r(*u_dice_16)}}} & \\heatcell{{{rm_u[1]}}}{{{mae_fmt(*u_mae_16)}}} & \\heatcell{{{rl_u[1]}}}{{{r(*u_loss_16)}}} \\\\"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse ablation test results for LR/WD table."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_DATASETS),
        default="unet",
        help="Model: unet, nestedunet, or lfunet (default: unet)",
    )
    args = parser.parse_args()
    chosen = MODEL_CHOSEN_ROW.get(args.model, 0)
    main(MODEL_DATASETS[args.model], chosen_lr_wd_row=chosen)
