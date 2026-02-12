#!/usr/bin/env python3
"""
Parse U-Net or U-Net++ per-layer Dice from test results.

U-Net (--model unet): Duke only, BS=16. No morph, Open k=3 (all), Open k=3 (smart).
U-Net++ (--model nestedunet): Duke (9 layers) + UMN (2 layers), BS=8. No morph, Close k=3 (all), Close k=3 (L3-5).

Output: LaTeX for per-layer Dice table. Dice_All has 9 values (L0--L8) for Duke; UMN uses L0, L1 only.

Usage: python parse_per_layer_dice.py [--model unet|nestedunet|lfunet]
"""
import argparse
import ast
import csv
import os

RESULTS_ROOT = "results"
LR_TARGET = 0.001
MODEL_CONFIG = {
    "unet": {
        "batch_size": 16,
        "base": "unet-duke-base",
        "morph": "unet-duke-base_morph",
        "op": "open",
        "layers_label": "Open k=3 (smart)",
    },
    "nestedunet": {
        "batch_size": 8,
        "base_duke": "nestedunet-duke-base",
        "base_umn": "nestedunet-umn-base",
        "morph_duke": "nestedunet-duke-base_morph",
        "morph_umn": "nestedunet-umn-base_morph",
        "op": "close",
        "layers_label": "Close k=3 (L3--5)",
    },
    "lfunet": {
        "batch_size": 8,
        "base": "lfunet-duke-base",
        "morph": "lfunet-duke-base_morph",
        "op": "both",
        "layers_label": "Both k=3 (L3--5)",
        "no_morph_row": 1,
    },
}


def get_dice_all_list(csv_path, lr=LR_TARGET, batch_size=16, row_index=None):
    """Return list of Dice_All arrays (one per matching row). LR, Batch_Size match. If row_index is set, keep only that row from matches."""
    if not os.path.isfile(csv_path):
        return []
    out = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lr_val = float(row.get("Learning_Rate", 0))
                bs_val = int(float(row.get("Batch_Size", 0)))
            except (ValueError, TypeError):
                continue
            if lr_val != lr or bs_val != batch_size:
                continue
            dice_all = row.get("Dice_All", "")
            if not dice_all:
                continue
            arr = ast.literal_eval(dice_all)
            if len(arr) >= 9:
                out.append(arr[:9])
    if row_index is not None and len(out) > row_index:
        return [out[row_index]]
    return out


def mean_std_per_layer(lists_of_nine):
    """lists_of_nine: list of 9-element lists. Return (mean_per_layer, std_per_layer), each length 9."""
    if not lists_of_nine:
        return None, None
    n = len(lists_of_nine)
    means = []
    stds = []
    for layer_idx in range(9):
        vals = [x[layer_idx] for x in lists_of_nine]
        m = sum(vals) / n
        s = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
        means.append(m)
        stds.append(s)
    return means, stds


def rank_heat_three(vals, higher_better=True):
    """Rank 3 values into 1--5 (5=best). Maps worst->1, mid->3, best->5."""
    sorted_idx = sorted(range(3), key=lambda i: vals[i], reverse=higher_better)
    ranks = [0, 0, 0]
    ranks[sorted_idx[0]] = 5
    ranks[sorted_idx[1]] = 3
    ranks[sorted_idx[2]] = 1
    return ranks


def _load_three_configs(
    base_dir, morph_dir, op_name, batch_size, no_morph_row_index=None
):
    """Load Dice_All lists for no_morph, op (all layers), op (layers 3-5). Returns (no_means, no_stds), (all_means, all_stds), (smart_means, smart_stds). no_morph_row_index: use that row from base CSV when multiple match (e.g. lfunet=1)."""
    no_morph = []
    for run in [1, 2]:
        path = os.path.join(
            RESULTS_ROOT, base_dir, f"bs{batch_size}_run{run}", "test", "results.csv"
        )
        no_morph.extend(
            get_dice_all_list(path, batch_size=batch_size, row_index=no_morph_row_index)
        )
    no_means, no_stds = mean_std_per_layer(no_morph)

    all_layers = []
    for run in [1, 2]:
        path = os.path.join(
            RESULTS_ROOT,
            morph_dir,
            f"bs{batch_size}_run{run}_{op_name}_k3",
            "test",
            "results.csv",
        )
        all_layers.extend(get_dice_all_list(path, batch_size=batch_size))
    all_means, all_stds = mean_std_per_layer(all_layers)

    smart = []
    for run in [1, 2]:
        path = os.path.join(
            RESULTS_ROOT,
            morph_dir,
            f"bs{batch_size}_run{run}_{op_name}_k3_layers3-4-5",
            "test",
            "results.csv",
        )
        smart.extend(get_dice_all_list(path, batch_size=batch_size))
    smart_means, smart_stds = mean_std_per_layer(smart)
    return (no_means, no_stds), (all_means, all_stds), (smart_means, smart_stds)


def _print_latex_block(
    no_means,
    no_stds,
    all_means,
    all_stds,
    smart_means,
    smart_stds,
    n_layers,
    col2_label,
    col3_label,
):
    def fmt(m, s):
        if m is None:
            return "—"
        s = s if s >= 0.005 else 0.00
        return f"{m:.2f} $\\pm$ {s:.2f}"

    for layer in range(n_layers):
        m0, s0 = no_means[layer], no_stds[layer]
        m1, s1 = all_means[layer], all_stds[layer]
        m2, s2 = smart_means[layer], smart_stds[layer]
        r0, r1, r2 = rank_heat_three([m0, m1, m2], higher_better=True)
        print(
            f"L{layer} & \\heatcell{{{r0}}}{{{fmt(m0, s0)}}} & \\heatcell{{{r1}}}{{{fmt(m1, s1)}}} & \\heatcell{{{r2}}}{{{fmt(m2, s2)}}} \\\\"
        )


def main_unet():
    cfg = MODEL_CONFIG["unet"]
    bs = cfg["batch_size"]
    no_m, no_s, all_m, all_s, smart_m, smart_s = _load_three_configs(
        cfg["base"], cfg["morph"], cfg["op"], bs
    )
    if no_m is None or all_m is None or smart_m is None:
        print("Missing data")
        return
    print("% LaTeX body for per-layer Dice table (U-Net Duke). Paste into 2_unet.tex.")
    print("% Layer & No morph & Open k=3 (all) & Open k=3 (smart) \\\\")
    _print_latex_block(
        no_m,
        no_s,
        all_m,
        all_s,
        smart_m,
        smart_s,
        9,
        "Open k=3 (all)",
        "Open k=3 (smart)",
    )


def main_lfunet():
    """LF-UNet: Duke only, BS=8. No morph (row 1), Both k=3 (all), Both k=3 (L3-5)."""
    cfg = MODEL_CONFIG["lfunet"]
    bs = cfg["batch_size"]
    no_morph_row = cfg.get("no_morph_row", 0)
    (no_m, no_s), (all_m, all_s), (smart_m, smart_s) = _load_three_configs(
        cfg["base"], cfg["morph"], cfg["op"], bs, no_morph_row_index=no_morph_row
    )
    if no_m is None or all_m is None or smart_m is None:
        print("Missing data")
        return
    print(
        "% LaTeX body for per-layer Dice table (LF-UNet Duke). Paste into 4_lfunet.tex."
    )
    print("% Layer & No morph & Both k=3 (all) & Both k=3 (L3--5) \\\\")
    _print_latex_block(
        no_m,
        no_s,
        all_m,
        all_s,
        smart_m,
        smart_s,
        9,
        "Both k=3 (all)",
        cfg["layers_label"],
    )


def main_nestedunet():
    cfg = MODEL_CONFIG["nestedunet"]
    bs = cfg["batch_size"]
    op = cfg["op"]
    # Duke: 9 layers
    no_d, all_d, smart_d = _load_three_configs(
        cfg["base_duke"], cfg["morph_duke"], op, bs
    )
    if no_d[0] is None or all_d[0] is None or smart_d[0] is None:
        print("Missing Duke data")
        return
    print("% (a) Duke: Layer & No morph & Close k=3 (all) & Close k=3 (L3--5) \\\\")
    _print_latex_block(
        no_d[0],
        no_d[1],
        all_d[0],
        all_d[1],
        smart_d[0],
        smart_d[1],
        9,
        "Close k=3 (all)",
        cfg["layers_label"],
    )
    # UMN: 2 layers (L0, L1 only)
    no_u = []
    for run in [1, 2]:
        path = os.path.join(
            RESULTS_ROOT, cfg["base_umn"], f"bs{bs}_run{run}", "test", "results.csv"
        )
        rows = get_dice_all_list(path, batch_size=bs)
        no_u.extend(rows)
    all_u = []
    for run in [1, 2]:
        path = os.path.join(
            RESULTS_ROOT,
            cfg["morph_umn"],
            f"bs{bs}_run{run}_{op}_k3",
            "test",
            "results.csv",
        )
        all_u.extend(get_dice_all_list(path, batch_size=bs))
    smart_u = []
    for run in [1, 2]:
        path = os.path.join(
            RESULTS_ROOT,
            cfg["morph_umn"],
            f"bs{bs}_run{run}_{op}_k3_layers3-4-5",
            "test",
            "results.csv",
        )
        smart_u.extend(get_dice_all_list(path, batch_size=bs))
    no_means_u, no_stds_u = mean_std_per_layer(no_u)
    all_means_u, all_stds_u = mean_std_per_layer(all_u)
    smart_means_u, smart_stds_u = mean_std_per_layer(smart_u)
    if no_means_u is None:
        print("Missing UMN data")
        return
    print("\n% (b) UMN: Layer & No morph & Close k=3 (all) & Close k=3 (L3--5) \\\\")
    _print_latex_block(
        no_means_u,
        no_stds_u,
        all_means_u,
        all_stds_u,
        smart_means_u,
        smart_stds_u,
        2,
        "Close k=3 (all)",
        cfg["layers_label"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Parse per-layer Dice for LaTeX table."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIG),
        default="unet",
        help="unet, nestedunet, or lfunet",
    )
    args = parser.parse_args()
    if args.model == "unet":
        main_unet()
    elif args.model == "lfunet":
        main_lfunet()
    else:
        main_nestedunet()


if __name__ == "__main__":
    main()
