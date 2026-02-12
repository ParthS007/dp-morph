#!/usr/bin/env python3
"""
Parse unet or nestedunet base_morph test results for the morphology table.
Duke: Dice over 7 layers L1--L7. UMN: Dice over L1 (fluid) only.
Output: mean ± std per config, and heatcell ranks (1-5) for LaTeX.

Usage: python parse_morph_table.py [--model unet|nestedunet|lfunet]
"""
import argparse
import ast
import csv
import os
from collections import defaultdict

RESULTS_ROOT = "results"
MODEL_CONFIG = {
    "unet": {
        "batch_size": 16,
        "base": ["unet-duke-base", "unet-umn-base"],
        "morph": ["unet-duke-base_morph", "unet-umn-base_morph"],
        "batchsize_ref": "tab:unet-oct-batchsize",
    },
    "nestedunet": {
        "batch_size": 8,
        "base": ["nestedunet-duke-base", "nestedunet-umn-base"],
        "morph": ["nestedunet-duke-base_morph", "nestedunet-umn-base_morph"],
        "batchsize_ref": "tab:nestedunet-oct-batchsize",
    },
    "lfunet": {
        "batch_size": 8,
        "base": ["lfunet-duke-base", "lfunet-umn-base"],
        "morph": ["lfunet-duke-base_morph", "lfunet-umn-base_morph"],
        "batchsize_ref": "tab:lfunet-oct-batchsize",
        "no_morph_row": 1,
    },
}
# no_morph_row: row index in base results.csv (0 = LR 1e-3 WD 1e-4, 1 = LR 1e-3 WD 1e-9). Default 0.
# Config: (op, k, smart_suffix for dir name)
CONFIGS = [
    ("Open", 3, ""),
    ("Open", 3, "_layers3-4-5"),
    ("Open", 5, ""),
    ("Open", 5, "_layers3-4-5"),
    ("Close", 3, ""),
    ("Close", 3, "_layers3-4-5"),
    ("Close", 5, ""),
    ("Close", 5, "_layers3-4-5"),
    ("Both", 3, ""),
    ("Both", 3, "_layers3-4-5"),
    ("Both", 5, ""),
    ("Both", 5, "_layers3-4-5"),
]


def dice_duke_7(dice_all_str):
    arr = ast.literal_eval(dice_all_str)
    return sum(arr[1:8]) / 7.0


def dice_umn_l1(dice_all_str):
    arr = ast.literal_eval(dice_all_str)
    return arr[1]


def load_row(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        row = next(r, None)
    return row


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return None, None
    m = sum(vals) / n
    s = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, s


def rank_heat(vals, higher_better=True):
    """Return rank 1-5 for each value (5 = best). Sorted so best is first (r=0)."""
    n = len(vals)
    sorted_vals = sorted(enumerate(vals), key=lambda x: x[1], reverse=higher_better)
    ranks = [0] * n
    for r, (i, _) in enumerate(sorted_vals):
        heat = max(1, min(5, round(5 - r * 4 / max(1, n - 1))))
        ranks[i] = heat
    return ranks


def main(model="unet"):
    cfg = MODEL_CONFIG[model]
    bs = cfg["batch_size"]
    base_dirs = cfg["base"]
    morph_dirs = cfg["morph"]
    batchsize_ref = cfg["batchsize_ref"]
    duke_base, umn_base = base_dirs[0], base_dirs[1]
    duke_morph, umn_morph = morph_dirs[0], morph_dirs[1]

    # No morph: from base (not morph); row index = no_morph_row (0 = 1e-3/1e-4, 1 = 1e-3/1e-9)
    no_morph_row_idx = cfg.get("no_morph_row", 0)
    no_morph_duke = []
    no_morph_umn = []
    for run in [1, 2]:
        for dataset in base_dirs:
            path = os.path.join(
                RESULTS_ROOT, dataset, f"bs{bs}_run{run}", "test", "results.csv"
            )
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows or len(rows) <= no_morph_row_idx:
                continue
            r0 = rows[no_morph_row_idx]
            dice_all = r0.get("Dice_All", "")
            mae = float(r0.get("MAE", 0))
            if "duke" in dataset:
                no_morph_duke.append((dice_duke_7(dice_all), mae))
            else:
                no_morph_umn.append((dice_umn_l1(dice_all), mae))
    # Aggregate no morph: Duke (dice from run1, run2), UMN same
    if len(no_morph_duke) >= 2:
        d_dice = mean_std([x[0] for x in no_morph_duke])
        d_mae = mean_std([x[1] for x in no_morph_duke])
    else:
        d_dice, d_mae = (None, None), (None, None)
    if len(no_morph_umn) >= 2:
        u_dice = mean_std([x[0] for x in no_morph_umn])
        u_mae = mean_std([x[1] for x in no_morph_umn])
    else:
        u_dice, u_mae = (None, None), (None, None)

    # Morph configs
    rows_data = []
    for op, k, suf in CONFIGS:
        dir_suffix = f"{op.lower()}_k{k}{suf}"
        duke_vals = []
        umn_vals = []
        duke_mae = []
        umn_mae = []
        for run in [1, 2]:
            for dataset in morph_dirs:
                path = os.path.join(
                    RESULTS_ROOT,
                    dataset,
                    f"bs{bs}_run{run}_{dir_suffix}",
                    "test",
                    "results.csv",
                )
                r = load_row(path)
                if r is None:
                    continue
                dice_all = r.get("Dice_All", "")
                mae = float(r.get("MAE", 0))
                if "duke" in dataset:
                    duke_vals.append(dice_duke_7(dice_all))
                    duke_mae.append(mae)
                else:
                    umn_vals.append(dice_umn_l1(dice_all))
                    umn_mae.append(mae)
        rows_data.append(
            {
                "label": f"{op} $k$={k} ({'smart' if suf else 'normal'})",
                "label_short": ("smart" if suf else "normal"),
                "duke_dice": (
                    mean_std(duke_vals)
                    if len(duke_vals) >= 2
                    else (
                        (sum(duke_vals) / len(duke_vals), 0)
                        if duke_vals
                        else (None, None)
                    )
                ),
                "duke_mae": (
                    mean_std(duke_mae)
                    if len(duke_mae) >= 2
                    else (
                        (sum(duke_mae) / len(duke_mae), 0) if duke_mae else (None, None)
                    )
                ),
                "umn_dice": (
                    mean_std(umn_vals)
                    if len(umn_vals) >= 2
                    else (
                        (sum(umn_vals) / len(umn_vals), 0) if umn_vals else (None, None)
                    )
                ),
                "umn_mae": (
                    mean_std(umn_mae)
                    if len(umn_mae) >= 2
                    else (sum(umn_mae) / len(umn_mae), 0) if umn_mae else (None, None)
                ),
            }
        )

    # Build full table: no morph + 12 rows
    all_dice_duke = [d_dice[0]] + [r["duke_dice"][0] for r in rows_data]
    all_mae_duke = [d_mae[0]] + [r["duke_mae"][0] for r in rows_data]
    all_dice_umn = [u_dice[0]] + [r["umn_dice"][0] for r in rows_data]
    all_mae_umn = [u_mae[0]] + [r["umn_mae"][0] for r in rows_data]
    rank_dice_d = rank_heat(all_dice_duke, higher_better=True)
    rank_mae_d = rank_heat(all_mae_duke, higher_better=False)
    rank_dice_u = rank_heat(all_dice_umn, higher_better=True)
    rank_mae_u = rank_heat(all_mae_umn, higher_better=False)

    def fmt(m, s, is_mae=False):
        if m is None:
            return "—", 1
        if is_mae:
            return (
                f"{m:.3f} $\\pm$ {s:.2f}" if s >= 0.01 else f"{m:.3f} $\\pm$ {s:.2f}"
            ), m
        return f"{m:.2f} $\\pm$ {s:.2f}", m

    # Print no morph
    rd, rm_d, ru, rm_u = rank_dice_d[0], rank_mae_d[0], rank_dice_u[0], rank_mae_u[0]
    d_str, _ = fmt(d_dice[0], d_dice[1])
    d_mae_str, _ = fmt(d_mae[0], d_mae[1], is_mae=True)
    u_str, _ = fmt(u_dice[0], u_dice[1])
    u_mae_str, _ = fmt(u_mae[0], u_mae[1], is_mae=True)
    print(
        f"No morph (Table~\\ref{{{batchsize_ref}}}) & \\heatcell{{{rd}}}{{{d_str}}} & \\heatcell{{{rm_d}}}{{{d_mae_str}}} & \\heatcell{{{ru}}}{{{u_str}}} & \\heatcell{{{rm_u}}}{{{u_mae_str}}} \\\\"
    )

    for i, r in enumerate(rows_data):
        rd, rm_d, ru, rm_u = (
            rank_dice_d[i + 1],
            rank_mae_d[i + 1],
            rank_dice_u[i + 1],
            rank_mae_u[i + 1],
        )
        norm = "normal" if r["label_short"] == "normal" else "smart"
        d_str, _ = fmt(r["duke_dice"][0], r["duke_dice"][1])
        d_mae_str, _ = fmt(r["duke_mae"][0], r["duke_mae"][1], is_mae=True)
        u_str, _ = fmt(r["umn_dice"][0], r["umn_dice"][1])
        u_mae_str, _ = fmt(r["umn_mae"][0], r["umn_mae"][1], is_mae=True)
        op_name = r["label"].split()[0]
        k_val = r["label"].split("$k$=")[1].split()[0].rstrip(")")
        print(
            f"{op_name} $k$={k_val} ({norm}) & \\heatcell{{{rd}}}{{{d_str}}} & \\heatcell{{{rm_d}}}{{{d_mae_str}}} & \\heatcell{{{ru}}}{{{u_str}}} & \\heatcell{{{rm_u}}}{{{u_mae_str}}} \\\\"
        )

    # Also print raw means for verification
    print("\n--- No morph ---")
    print("Duke:", d_dice, d_mae)
    print("UMN:", u_dice, u_mae)
    print("\n--- Morph rows (Duke dice, Duke MAE, UMN dice, UMN MAE) ---")
    for r in rows_data:
        print(r["label"], r["duke_dice"], r["duke_mae"], r["umn_dice"], r["umn_mae"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse morphology table results.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIG),
        default="unet",
        help="unet, nestedunet, or lfunet",
    )
    args = parser.parse_args()
    main(model=args.model)
