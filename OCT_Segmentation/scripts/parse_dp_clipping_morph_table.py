#!/usr/bin/env python3
"""
Parse U-Net, U-Net++, or LF-UNet DP clipping + morph table (ε=200), two panels: Duke (no morph + smart morph), UMN (no morph + normal morph).

U-Net: BS=16. Duke: Smart morph = Open k=3 layers 3-5 (open_k3_layers3-4-5). UMN: Normal morph = Open k=3 all layers (open_k3).
U-Net++: BS=8. Duke: Smart morph = Close k=3 layers 3-5 (close_k3_layers3-4-5). UMN: Normal morph = Close k=3 all layers (close_k3).
LF-UNet: BS=8. Duke: Smart morph = Both k=3 layers 3-5 (both_k3_layers3-4-5). UMN: Normal morph = Open k=3 all layers (open_k3).

Output: LaTeX for two minipages (Duke | UMN), same structure as Table in 2_unet.tex / 3_nestedunet.tex / 4_lfunet.tex.

Usage: python parse_dp_clipping_morph_table.py [--model unet|nestedunet|lfunet]
"""
import argparse
import ast
import csv
import os

RESULTS_ROOT = "results"
EPS = 200
MODEL_CONFIG = {
    "unet": {
        "bs": 16,
        "base_duke": "unet-duke-base",
        "base_umn": "unet-umn-base",
        "batchsize_ref": "tab:unet-oct-batchsize",
        "strategies": [
            (
                "AUTO-S",
                "unet-duke-dp_automatic",
                "unet-umn-dp_automatic",
                "unet-duke-dp_automatic_morph",
                "unet-umn-dp_automatic_morph",
            ),
            (
                "Flat clip.",
                "unet-duke-dp_flat",
                "unet-umn-dp_flat",
                "unet-duke-dp_flat_morph",
                "unet-umn-dp_flat_morph",
            ),
            (
                "NSGD",
                "unet-duke-dp_normalized_sgd",
                "unet-umn-dp_normalized_sgd",
                "unet-duke-dp_normalized_sgd_morph",
                "unet-umn-dp_normalized_sgd_morph",
            ),
            (
                "PSAC",
                "unet-duke-dp_psac",
                "unet-umn-dp_psac",
                "unet-duke-dp_psac_morph",
                "unet-umn-dp_psac_morph",
            ),
        ],
        "morph_suffix": "open_k3_layers3-4-5",
        "morph_label": "Smart morph",
        "normal_morph_suffix": "open_k3",
        "normal_morph_label": "Normal morph",
    },
    "nestedunet": {
        "bs": 8,
        "base_duke": "nestedunet-duke-base",
        "base_umn": "nestedunet-umn-base",
        "batchsize_ref": "tab:nestedunet-oct-batchsize",
        "strategies": [
            (
                "AUTO-S",
                "nestedunet-duke-dp_automatic",
                "nestedunet-umn-dp_automatic",
                "nestedunet-duke-dp_automatic_morph",
                "nestedunet-umn-dp_automatic_morph",
            ),
            (
                "Flat clip.",
                "nestedunet-duke-dp_flat",
                "nestedunet-umn-dp_flat",
                "nestedunet-duke-dp_flat_morph",
                "nestedunet-umn-dp_flat_morph",
            ),
            (
                "NSGD",
                "nestedunet-duke-dp_normalized_sgd",
                "nestedunet-umn-dp_normalized_sgd",
                "nestedunet-duke-dp_normalized_sgd_morph",
                "nestedunet-umn-dp_normalized_sgd_morph",
            ),
            (
                "PSAC",
                "nestedunet-duke-dp_psac",
                "nestedunet-umn-dp_psac",
                "nestedunet-duke-dp_psac_morph",
                "nestedunet-umn-dp_psac_morph",
            ),
        ],
        "morph_suffix": "close_k3_layers3-4-5",
        "morph_label": "Smart morph",
        "normal_morph_suffix": "close_k3",
        "normal_morph_label": "Normal morph",
    },
    "lfunet": {
        "bs": 8,
        "base_duke": "lfunet-duke-base",
        "base_umn": "lfunet-umn-base",
        "batchsize_ref": "tab:lfunet-oct-batchsize",
        "strategies": [
            (
                "AUTO-S",
                "lfunet-duke-dp_automatic",
                "lfunet-umn-dp_automatic",
                "lfunet-duke-dp_automatic_morph",
                "lfunet-umn-dp_automatic_morph",
            ),
            (
                "Flat clip.",
                "lfunet-duke-dp_flat",
                "lfunet-umn-dp_flat",
                "lfunet-duke-dp_flat_morph",
                "lfunet-umn-dp_flat_morph",
            ),
            (
                "NSGD",
                "lfunet-duke-dp_normalized_sgd",
                "lfunet-umn-dp_normalized_sgd",
                "lfunet-duke-dp_normalized_sgd_morph",
                "lfunet-umn-dp_normalized_sgd_morph",
            ),
            (
                "PSAC",
                "lfunet-duke-dp_psac",
                "lfunet-umn-dp_psac",
                "lfunet-duke-dp_psac_morph",
                "lfunet-umn-dp_psac_morph",
            ),
        ],
        "morph_suffix": "both_k3_layers3-4-5",
        "morph_label": "Smart morph",
        "normal_morph_suffix": "open_k3",
        "normal_morph_label": "Normal morph",
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


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return None, None
    m = sum(vals) / n
    s = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, s


def fmt(m, s, decimals=2):
    if m is None:
        return "---"
    if s is not None and s < 0.005:
        s = 0.00
    return f"{m:.2f} $\\pm$ {s:.2f}" if s is not None else f"{m:.2f}"


def collect_dp_no_morph(prefix_duke, prefix_umn, bs):
    """Collect Dice and MAE for DP no-morph from bs*_eps200_run1, run2."""
    duke_dice, duke_mae = [], []
    umn_dice, umn_mae = [], []
    for run in [1, 2]:
        path_d = os.path.join(
            RESULTS_ROOT,
            prefix_duke,
            f"bs{bs}_eps{EPS}_run{run}",
            "test",
            "results.csv",
        )
        path_u = os.path.join(
            RESULTS_ROOT, prefix_umn, f"bs{bs}_eps{EPS}_run{run}", "test", "results.csv"
        )
        row_d = load_first_row(path_d)
        row_u = load_first_row(path_u)
        if row_d and row_d.get("Dice_All"):
            duke_dice.append(dice_duke_7(row_d["Dice_All"]))
            duke_mae.append(float(row_d.get("MAE", 0)))
        if row_u and row_u.get("Dice_All"):
            umn_dice.append(dice_umn_l1(row_u["Dice_All"]))
            umn_mae.append(float(row_u.get("MAE", 0)))
    return (mean_std(duke_dice), mean_std(duke_mae)), (
        mean_std(umn_dice),
        mean_std(umn_mae),
    )


def collect_dp_smart_morph_duke(prefix_duke_morph, bs, morph_suffix):
    """Collect Duke only: smart morph from dp_*_morph (open_k3 or close_k3 L3-5)."""
    duke_dice, duke_mae = [], []
    for run in [1, 2]:
        path = os.path.join(
            RESULTS_ROOT,
            prefix_duke_morph,
            f"bs{bs}_eps{EPS}_run{run}_{morph_suffix}",
            "test",
            "results.csv",
        )
        row = load_first_row(path)
        if row and row.get("Dice_All"):
            duke_dice.append(dice_duke_7(row["Dice_All"]))
            duke_mae.append(float(row.get("MAE", 0)))
    return mean_std(duke_dice), mean_std(duke_mae)


def collect_dp_normal_morph_umn(prefix_umn_morph, bs, normal_morph_suffix):
    """Collect UMN: normal morph from dp_*_morph (open_k3 or close_k3 all layers)."""
    umn_dice, umn_mae = [], []
    for run in [1, 2]:
        path = os.path.join(
            RESULTS_ROOT,
            prefix_umn_morph,
            f"bs{bs}_eps{EPS}_run{run}_{normal_morph_suffix}",
            "test",
            "results.csv",
        )
        row = load_first_row(path)
        if row and row.get("Dice_All"):
            umn_dice.append(dice_umn_l1(row["Dice_All"]))
            umn_mae.append(float(row.get("MAE", 0)))
    return mean_std(umn_dice), mean_std(umn_mae)


def main(model="unet"):
    cfg = MODEL_CONFIG[model]
    bs = cfg["bs"]
    batchsize_ref = cfg["batchsize_ref"]
    strategies = cfg["strategies"]
    morph_suffix = cfg["morph_suffix"]
    morph_label = cfg["morph_label"]
    normal_morph_suffix = cfg["normal_morph_suffix"]
    normal_morph_label = cfg["normal_morph_label"]
    base_d, base_u = cfg["base_duke"], cfg["base_umn"]

    np_duke_dice, np_duke_mae = [], []
    np_umn_dice, np_umn_mae = [], []
    for run in [1, 2]:
        for base, d_list, m_list in [
            (base_d, np_duke_dice, np_duke_mae),
            (base_u, np_umn_dice, np_umn_mae),
        ]:
            path = os.path.join(
                RESULTS_ROOT, base, f"bs{bs}_run{run}", "test", "results.csv"
            )
            row = load_first_row(path)
            if row and row.get("Dice_All"):
                if "duke" in base:
                    d_list.append(dice_duke_7(row["Dice_All"]))
                else:
                    d_list.append(dice_umn_l1(row["Dice_All"]))
                m_list.append(float(row.get("MAE", 0)))
    np_duke = (mean_std(np_duke_dice), mean_std(np_duke_mae))
    np_umn = (mean_std(np_umn_dice), mean_std(np_umn_mae))

    no_morph_rows = []
    smart_morph_rows = []
    normal_morph_rows = []
    for name, pduke, pumn, pduke_morph, pumn_morph in strategies:
        (dd, dm), (ud, um) = collect_dp_no_morph(pduke, pumn, bs)
        no_morph_rows.append((name, dd, dm, ud, um))
        sd, sm = collect_dp_smart_morph_duke(pduke_morph, bs, morph_suffix)
        smart_morph_rows.append((name, sd, sm))
        nud, num = collect_dp_normal_morph_umn(pumn_morph, bs, normal_morph_suffix)
        normal_morph_rows.append((name, nud, num))

    # Ranks per panel: Duke 9 (np + 4 no-morph + 4 smart), UMN 9 (np + 4 no-morph + 4 normal)
    all_duke_dice = (
        [np_duke[0][0]]
        + [r[1][0] for r in no_morph_rows]
        + [r[1][0] for r in smart_morph_rows]
    )
    all_duke_mae = (
        [np_duke[1][0]]
        + [r[2][0] for r in no_morph_rows]
        + [r[2][0] for r in smart_morph_rows]
    )
    all_umn_dice = (
        [np_umn[0][0]]
        + [r[3][0] for r in no_morph_rows]
        + [r[1][0] for r in normal_morph_rows]
    )
    all_umn_mae = (
        [np_umn[1][0]]
        + [r[4][0] for r in no_morph_rows]
        + [r[2][0] for r in normal_morph_rows]
    )

    def rank_vals(vals, higher_better=True):
        key = lambda i: (
            vals[i]
            if vals[i] is not None
            else (-float("inf") if higher_better else float("inf"))
        )
        idx = sorted(range(len(vals)), key=key, reverse=higher_better)
        r = [0] * len(vals)
        n = len(vals)
        for i, pos in enumerate(idx):
            r[pos] = (5 - i) if n <= 5 else max(1, 5 - (i // 2))
        return r

    r_duke_d = rank_vals(all_duke_dice, True)
    r_duke_m = rank_vals(all_duke_mae, False)
    r_umn_d = rank_vals(all_umn_dice, True)
    r_umn_m = rank_vals(all_umn_mae, False)

    np_d_d = fmt(np_duke[0][0], np_duke[0][1])
    np_d_m = fmt(np_duke[1][0], np_duke[1][1])
    np_u_d = fmt(np_umn[0][0], np_umn[0][1])
    np_u_m = fmt(np_umn[1][0], np_umn[1][1])

    # Two-panel output (Duke | UMN), same structure as 2_unet.tex / 3_nestedunet.tex
    print(
        f"% LaTeX: two minipages (Duke | UMN). BS={bs}, Duke: no morph + {morph_label}; UMN: no morph + {normal_morph_label}."
    )
    print("% --- Duke panel ---")
    print(
        f"Non-private (Table~\\ref{{{batchsize_ref}}}) & - & \\heatcell{{{r_duke_d[0]}}}{{{np_d_d}}} & \\heatcell{{{r_duke_m[0]}}}{{{np_d_m}}} \\\\"
    )
    print("\\midrule")
    for i, (name, dd, dm, ud, um) in enumerate(no_morph_rows):
        rd, rm_d = r_duke_d[1 + i], r_duke_m[1 + i]
        d_d, d_m = fmt(dd[0], dd[1]), fmt(dm[0], dm[1])
        print(
            f"{name} & No morph & \\heatcell{{{rd}}}{{{d_d}}} & \\heatcell{{{rm_d}}}{{{d_m}}} \\\\"
        )
    print("\\midrule")
    for i, (name, sd, sm) in enumerate(smart_morph_rows):
        rd, rm_d = r_duke_d[5 + i], r_duke_m[5 + i]
        s_d, s_m = fmt(sd[0], sd[1]), fmt(sm[0], sm[1])
        print(
            f"{name} & {morph_label} & \\heatcell{{{rd}}}{{{s_d}}} & \\heatcell{{{rm_d}}}{{{s_m}}} \\\\"
        )
    print("% --- UMN panel ---")
    print(
        f"Non-private (Table~\\ref{{{batchsize_ref}}}) & - & \\heatcell{{{r_umn_d[0]}}}{{{np_u_d}}} & \\heatcell{{{r_umn_m[0]}}}{{{np_u_m}}} \\\\"
    )
    print("\\midrule")
    for i, (name, dd, dm, ud, um) in enumerate(no_morph_rows):
        ru, rm_u = r_umn_d[1 + i], r_umn_m[1 + i]
        u_d, u_m = fmt(ud[0], ud[1]), fmt(um[0], um[1])
        print(
            f"{name} & No morph & \\heatcell{{{ru}}}{{{u_d}}} & \\heatcell{{{rm_u}}}{{{u_m}}} \\\\"
        )
    print("\\midrule")
    for i, (name, nud, num) in enumerate(normal_morph_rows):
        ru, rm_u = r_umn_d[5 + i], r_umn_m[5 + i]
        u_d, u_m = fmt(nud[0], nud[1]), fmt(num[0], num[1])
        print(
            f"{name} & {normal_morph_label} & \\heatcell{{{ru}}}{{{u_d}}} & \\heatcell{{{rm_u}}}{{{u_m}}} \\\\"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse DP clipping + morph merged table."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIG),
        default="unet",
        help="unet, nestedunet, or lfunet",
    )
    args = parser.parse_args()
    main(model=args.model)
