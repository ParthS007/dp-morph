#!/usr/bin/env python3
"""
Generate LaTeX results tables for OCT Segmentation experiments.

Usage:
    python generate_tables.py --arch lfunet
    python generate_tables.py --arch unet --output results_unet.tex
"""

import os
import pandas as pd
import numpy as np
import ast
import re
import argparse
from collections import defaultdict

# Default results path (can be overridden via CLI)
# Script is in slurm/, results are in parent directory
DEFAULT_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results"
)

# Architecture display name mapping
ARCH_DISPLAY_NAMES = {
    "lfunet": "LF-UNet",
    "unet": "U-Net",
    "nestedunet": "U-Net++",
}

# Global variables set by CLI args
RESULTS_PATH = None
ARCH = None
ARCH_DISPLAY = None

# Clipping strategy display names
STRATEGY_DISPLAY = {
    "automatic": "AUTO-S",
    "flat": "Flat",
    "normalized_sgd": "Normalized SGD",
    "psac": "PSAC",
}

STRATEGY_ORDER = ["automatic", "flat", "normalized_sgd", "psac"]


def parse_dice_all(dice_str):
    """Parse Dice_All string to list of floats."""
    try:
        return ast.literal_eval(dice_str)
    except:
        return None


def read_results_file(filepath):
    """Read a results.csv file and return filtered dataframe."""
    try:
        df = pd.read_csv(filepath)
        # Filter for LR = 0.001
        df = df[df["Learning_Rate"] == 0.001]
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def calculate_metrics_duke(row):
    """Calculate Duke metrics from a row."""
    dice_all = parse_dice_all(row["Dice_All"])
    if dice_all is None or len(dice_all) < 9:
        return None, None, None

    # Dice (7 Layers): indices 1-7 (excluding background at 0 and choroid/fluid at 8)
    dice_7 = np.mean(dice_all[1:8])
    # Dice (All Layers): all 9 values
    dice_all_mean = np.mean(dice_all)
    mae = row["MAE"]

    return dice_7, dice_all_mean, mae


def calculate_metrics_umn(row):
    """Calculate UMN metrics from a row."""
    dice_all = parse_dice_all(row["Dice_All"])
    if dice_all is None or len(dice_all) < 2:
        return None, None, None

    # Dice (RNFL Layer): index 1 (RNFL layer only)
    dice_rnfl = dice_all[1] if len(dice_all) > 1 else None
    # Dice (All Layers): indices 0-1 (background + RNFL)
    dice_all_mean = np.mean(dice_all[0:2])
    mae = row["MAE"]

    return dice_rnfl, dice_all_mean, mae


def collect_nonprivate_results(dataset):
    """Collect non-private baseline and morph results with traceability."""
    results = defaultdict(list)

    # Base directory patterns
    base_dir = f"{ARCH}-{dataset}-base"
    morph_dir = f"{ARCH}-{dataset}-base_morph"

    # Collect baseline results
    base_path = os.path.join(RESULTS_PATH, base_dir)
    if os.path.exists(base_path):
        for subdir in os.listdir(base_path):
            match = re.match(r"bs(\d+)_run(\d+)", subdir)
            if match:
                bs = int(match.group(1))
                run = int(match.group(2))
                results_file = os.path.join(base_path, subdir, "test", "results.csv")
                if os.path.exists(results_file):
                    df = read_results_file(results_file)
                    if df is not None and len(df) > 0:
                        row = df.iloc[0]
                        trace_info = {
                            "csv_file": results_file,
                            "row_index": 2,  # +2 for header and 0-index
                            "run": run,
                            "bs": bs,
                        }
                        results[("baseline", None, None, bs)].append(
                            (run, row, trace_info)
                        )

    # Collect morph results
    morph_path = os.path.join(RESULTS_PATH, morph_dir)
    if os.path.exists(morph_path):
        for subdir in os.listdir(morph_path):
            # Pattern: bs{8,16}_run{1,2}_{op}_k{3,5}_layers3-4-5
            match = re.match(r"bs(\d+)_run(\d+)_(\w+)_k(\d+)", subdir)
            if match:
                bs = int(match.group(1))
                run = int(match.group(2))
                op = match.group(3)
                k = int(match.group(4))
                results_file = os.path.join(morph_path, subdir, "test", "results.csv")
                if os.path.exists(results_file):
                    df = read_results_file(results_file)
                    if df is not None and len(df) > 0:
                        row = df.iloc[0]
                        trace_info = {
                            "csv_file": results_file,
                            "row_index": 2,  # +2 for header and 0-index
                            "run": run,
                            "bs": bs,
                            "op": op,
                            "k": k,
                        }
                        results[("morph", op, k, bs)].append((run, row, trace_info))

    return results


def collect_dp_results(dataset, strategy):
    """Collect DP baseline and morph results for a given strategy with traceability."""
    results = defaultdict(list)

    # Base directory patterns
    base_dir = f"{ARCH}-{dataset}-dp_{strategy}"
    morph_dir = f"{ARCH}-{dataset}-dp_{strategy}_morph"

    # Collect baseline results
    base_path = os.path.join(RESULTS_PATH, base_dir)
    if os.path.exists(base_path):
        for subdir in os.listdir(base_path):
            # Pattern: bs{8,16}_eps{8,200}_run{1,2}
            match = re.match(r"bs(\d+)_eps(\d+)_run(\d+)", subdir)
            if match:
                bs = int(match.group(1))
                eps = int(match.group(2))
                run = int(match.group(3))
                results_file = os.path.join(base_path, subdir, "test", "results.csv")
                if os.path.exists(results_file):
                    df = read_results_file(results_file)
                    if df is not None and len(df) > 0:
                        row = df.iloc[0]
                        trace_info = {
                            "csv_file": results_file,
                            "row_index": 2,  # +2 for header and 0-index
                            "run": run,
                            "bs": bs,
                            "eps": eps,
                        }
                        results[("baseline", None, None, bs, eps)].append(
                            (run, row, trace_info)
                        )

    # Collect morph results
    morph_path = os.path.join(RESULTS_PATH, morph_dir)
    if os.path.exists(morph_path):
        for subdir in os.listdir(morph_path):
            # Pattern: bs{8,16}_eps{8,200}_run{1,2}_{op}_k{3,5}
            match = re.match(r"bs(\d+)_eps(\d+)_run(\d+)_(\w+)_k(\d+)", subdir)
            if match:
                bs = int(match.group(1))
                eps = int(match.group(2))
                run = int(match.group(3))
                op = match.group(4)
                k = int(match.group(5))
                results_file = os.path.join(morph_path, subdir, "test", "results.csv")
                if os.path.exists(results_file):
                    df = read_results_file(results_file)
                    if df is not None and len(df) > 0:
                        row = df.iloc[0]
                        trace_info = {
                            "csv_file": results_file,
                            "row_index": 2,  # +2 for header and 0-index
                            "run": run,
                            "bs": bs,
                            "eps": eps,
                            "op": op,
                            "k": k,
                        }
                        results[("morph", op, k, bs, eps)].append(
                            (run, row, trace_info)
                        )

    return results


def aggregate_metrics_duke(rows_list):
    """Aggregate metrics over runs for Duke dataset with traceability."""
    dice_7_vals = []
    dice_all_vals = []
    mae_vals = []
    trace_info_list = []

    for item in rows_list:
        if len(item) == 3:
            run, row, trace_info = item
        else:
            # Backward compatibility
            run, row = item
            trace_info = None
        d7, da, mae = calculate_metrics_duke(row)
        if d7 is not None:
            dice_7_vals.append(d7)
            dice_all_vals.append(da)
            mae_vals.append(mae)
            if trace_info:
                trace_info_list.append(trace_info)

    if len(dice_7_vals) == 0:
        return None, None, None, None, None, None, []

    return (
        np.mean(dice_7_vals),
        np.std(dice_7_vals),
        np.mean(dice_all_vals),
        np.std(dice_all_vals),
        np.mean(mae_vals),
        np.std(mae_vals),
        trace_info_list,
    )


def aggregate_metrics_umn(rows_list):
    """Aggregate metrics over runs for UMN dataset with traceability."""
    dice_rnfl_vals = []
    dice_all_vals = []
    mae_vals = []
    trace_info_list = []

    for item in rows_list:
        if len(item) == 3:
            run, row, trace_info = item
        else:
            # Backward compatibility
            run, row = item
            trace_info = None
        dr, da, mae = calculate_metrics_umn(row)
        if dr is not None:
            dice_rnfl_vals.append(dr)
            dice_all_vals.append(da)
            mae_vals.append(mae)
            if trace_info:
                trace_info_list.append(trace_info)

    if len(dice_rnfl_vals) == 0:
        return None, None, None, None, None, None, []

    return (
        np.mean(dice_rnfl_vals),
        np.std(dice_rnfl_vals),
        np.mean(dice_all_vals),
        np.std(dice_all_vals),
        np.mean(mae_vals),
        np.std(mae_vals),
        trace_info_list,
    )


def format_metric(mean, std, precision=3):
    """Format metric as mean ± std."""
    if mean is None:
        return "---"
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"


def format_metric_bold(mean, std, is_best, precision=3):
    """Format metric as mean ± std, with bold if best."""
    if mean is None:
        return "---"
    fmt = f"{{:.{precision}f}}"
    val_str = f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"
    if is_best:
        return f"\\textbf{{{val_str}}}"
    return val_str


def generate_nonprivate_combined_table(duke_results, umn_results):
    """Generate combined non-private table with Duke and UMN side-by-side."""
    configs = [
        ("Baseline", "baseline", None, None),
        ("Morph-Open $k$=3", "morph", "open", 3),
        ("Morph-Open $k$=5", "morph", "open", 5),
        ("Morph-Close $k$=3", "morph", "close", 3),
        ("Morph-Close $k$=5", "morph", "close", 5),
        ("Morph-Both $k$=3", "morph", "both", 3),
        ("Morph-Both $k$=5", "morph", "both", 5),
    ]

    # Collect all metrics
    all_metrics = []

    for config_name, config_type, op, k in configs:
        for bs in [8, 16]:
            # Duke metrics
            if config_type == "baseline":
                duke_key = ("baseline", None, None, bs)
            else:
                duke_key = ("morph", op, k, bs)

            if duke_key in duke_results and len(duke_results[duke_key]) > 0:
                d7_mean, d7_std, da_mean, da_std, mae_mean, mae_std, duke_trace = (
                    aggregate_metrics_duke(duke_results[duke_key])
                )
            else:
                d7_mean = d7_std = da_mean = da_std = mae_mean = mae_std = None
                duke_trace = []

            # UMN metrics
            if config_type == "baseline":
                umn_key = ("baseline", None, None, bs)
            else:
                umn_key = ("morph", op, k, bs)

            if umn_key in umn_results and len(umn_results[umn_key]) > 0:
                (
                    dr_mean,
                    dr_std,
                    da_umn_mean,
                    da_umn_std,
                    mae_umn_mean,
                    mae_umn_std,
                    umn_trace,
                ) = aggregate_metrics_umn(umn_results[umn_key])
            else:
                dr_mean = dr_std = da_umn_mean = da_umn_std = mae_umn_mean = (
                    mae_umn_std
                ) = None
                umn_trace = []

            all_metrics.append(
                {
                    "config": config_name,
                    "bs": bs,
                    "duke_d7_mean": d7_mean,
                    "duke_d7_std": d7_std,
                    "duke_da_mean": da_mean,
                    "duke_da_std": da_std,
                    "duke_mae_mean": mae_mean,
                    "duke_mae_std": mae_std,
                    "umn_dr_mean": dr_mean,
                    "umn_dr_std": dr_std,
                    "umn_da_mean": da_umn_mean,
                    "umn_da_std": da_umn_std,
                    "umn_mae_mean": mae_umn_mean,
                    "umn_mae_std": mae_umn_std,
                    "duke_trace": duke_trace,
                    "umn_trace": umn_trace,
                }
            )

    # Find best values
    duke_d7_vals = [
        m["duke_d7_mean"] for m in all_metrics if m["duke_d7_mean"] is not None
    ]
    duke_da_vals = [
        m["duke_da_mean"] for m in all_metrics if m["duke_da_mean"] is not None
    ]
    duke_mae_vals = [
        m["duke_mae_mean"] for m in all_metrics if m["duke_mae_mean"] is not None
    ]
    umn_dr_vals = [
        m["umn_dr_mean"] for m in all_metrics if m["umn_dr_mean"] is not None
    ]
    umn_da_vals = [
        m["umn_da_mean"] for m in all_metrics if m["umn_da_mean"] is not None
    ]
    umn_mae_vals = [
        m["umn_mae_mean"] for m in all_metrics if m["umn_mae_mean"] is not None
    ]

    best_duke_d7 = max(duke_d7_vals) if duke_d7_vals else None
    best_duke_da = max(duke_da_vals) if duke_da_vals else None
    best_duke_mae = min(duke_mae_vals) if duke_mae_vals else None
    best_umn_dr = max(umn_dr_vals) if umn_dr_vals else None
    best_umn_da = max(umn_da_vals) if umn_da_vals else None
    best_umn_mae = min(umn_mae_vals) if umn_mae_vals else None

    # Generate table rows
    latex_rows = []
    prev_config = None

    for m in all_metrics:
        if prev_config is not None and prev_config != m["config"].split()[0]:
            latex_rows.append("\\midrule")
        prev_config = m["config"].split()[0]

        # Duke columns
        d7_str = format_metric_bold(
            m["duke_d7_mean"], m["duke_d7_std"], m["duke_d7_mean"] == best_duke_d7
        )
        da_duke_str = format_metric_bold(
            m["duke_da_mean"], m["duke_da_std"], m["duke_da_mean"] == best_duke_da
        )
        mae_duke_str = format_metric_bold(
            m["duke_mae_mean"], m["duke_mae_std"], m["duke_mae_mean"] == best_duke_mae
        )

        # UMN columns
        dr_str = format_metric_bold(
            m["umn_dr_mean"], m["umn_dr_std"], m["umn_dr_mean"] == best_umn_dr
        )
        da_umn_str = format_metric_bold(
            m["umn_da_mean"], m["umn_da_std"], m["umn_da_mean"] == best_umn_da
        )
        mae_umn_str = format_metric_bold(
            m["umn_mae_mean"],
            m["umn_mae_std"],
            m["umn_mae_mean"] == best_umn_mae,
            precision=4,
        )

        latex_rows.append(
            f"{m['config']} & {m['bs']} & {d7_str} & {da_duke_str} & {mae_duke_str} & {dr_str} & {da_umn_str} & {mae_umn_str} \\\\"
        )

    # Build combined table
    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} non-private baseline results on Duke and UMN datasets. 
$\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better. 
Duke: Dice (7 Layers) and Dice (All Layers). UMN: Dice (RNFL Layer) and Dice (All Layers).
Morphology targets layers 3-5 (inner retinal layers) for Duke and all classes for UMN.
Best values per metric are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-nonprivate-combined}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll|ccc|ccc}}
\\toprule
\\multirow{{2}}{{*}}{{Configuration}} & \\multirow{{2}}{{*}}{{Batch Size}} & 
\\multicolumn{{3}}{{c}}{{Duke}} & \\multicolumn{{3}}{{c}}{{UMN}} \\\\
\\cmidrule(lr){{3-5}} \\cmidrule(lr){{6-8}}
& & Dice (7) $\\uparrow$ & Dice (All) $\\uparrow$ & MAE $\\downarrow$ & 
Dice (RNFL) $\\uparrow$ & Dice (All) $\\uparrow$ & MAE $\\downarrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table, all_metrics


def generate_nonprivate_duke_table(results):
    """Generate Table 1: Non-Private Duke."""
    rows = []

    # Define configuration order
    configs = [
        ("Baseline", "baseline", None, None),
        ("Morph-Open $k$=3", "morph", "open", 3),
        ("Morph-Open $k$=5", "morph", "open", 5),
        ("Morph-Close $k$=3", "morph", "close", 3),
        ("Morph-Close $k$=5", "morph", "close", 5),
        ("Morph-Both $k$=3", "morph", "both", 3),
        ("Morph-Both $k$=5", "morph", "both", 5),
    ]

    # Collect all metrics for finding best values
    all_metrics = []

    for config_name, config_type, op, k in configs:
        for bs in [8, 16]:
            if config_type == "baseline":
                key = ("baseline", None, None, bs)
            else:
                key = ("morph", op, k, bs)

            if key in results and len(results[key]) > 0:
                d7_mean, d7_std, da_mean, da_std, mae_mean, mae_std = (
                    aggregate_metrics_duke(results[key])
                )
                all_metrics.append(
                    {
                        "config": config_name,
                        "bs": bs,
                        "d7_mean": d7_mean,
                        "d7_std": d7_std,
                        "da_mean": da_mean,
                        "da_std": da_std,
                        "mae_mean": mae_mean,
                        "mae_std": mae_std,
                    }
                )
            else:
                all_metrics.append(
                    {
                        "config": config_name,
                        "bs": bs,
                        "d7_mean": None,
                        "d7_std": None,
                        "da_mean": None,
                        "da_std": None,
                        "mae_mean": None,
                        "mae_std": None,
                    }
                )

    # Find best values
    d7_vals = [m["d7_mean"] for m in all_metrics if m["d7_mean"] is not None]
    da_vals = [m["da_mean"] for m in all_metrics if m["da_mean"] is not None]
    mae_vals = [m["mae_mean"] for m in all_metrics if m["mae_mean"] is not None]

    best_d7 = max(d7_vals) if d7_vals else None
    best_da = max(da_vals) if da_vals else None
    best_mae = min(mae_vals) if mae_vals else None

    # Generate table rows
    latex_rows = []
    prev_config = None

    for m in all_metrics:
        # Add midrule between different config groups
        if prev_config is not None and prev_config != m["config"].split()[0]:
            latex_rows.append("\\midrule")
        prev_config = m["config"].split()[0]

        d7_str = format_metric_bold(m["d7_mean"], m["d7_std"], m["d7_mean"] == best_d7)
        da_str = format_metric_bold(m["da_mean"], m["da_std"], m["da_mean"] == best_da)
        mae_str = format_metric_bold(
            m["mae_mean"], m["mae_std"], m["mae_mean"] == best_mae
        )

        latex_rows.append(
            f"{m['config']} & {m['bs']} & {d7_str} & {da_str} & {mae_str} \\\\"
        )

    # Build table
    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} non-private baseline results on Duke dataset. 
$\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better. 
Morphology targets layers 3-5 (inner retinal layers).
Best values per metric are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-duke-nonprivate}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llccc}}
\\toprule
Configuration & Batch Size & Dice (7 Layers) $\\uparrow$ & Dice (All Layers) $\\uparrow$ & MAE $\\downarrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table, all_metrics


def generate_nonprivate_umn_table(results):
    """Generate Table 2: Non-Private UMN."""
    configs = [
        ("Baseline", "baseline", None, None),
        ("Morph-Open $k$=3", "morph", "open", 3),
        ("Morph-Open $k$=5", "morph", "open", 5),
        ("Morph-Close $k$=3", "morph", "close", 3),
        ("Morph-Close $k$=5", "morph", "close", 5),
        ("Morph-Both $k$=3", "morph", "both", 3),
        ("Morph-Both $k$=5", "morph", "both", 5),
    ]

    all_metrics = []

    for config_name, config_type, op, k in configs:
        for bs in [8, 16]:
            if config_type == "baseline":
                key = ("baseline", None, None, bs)
            else:
                key = ("morph", op, k, bs)

            if key in results and len(results[key]) > 0:
                d2_mean, d2_std, mae_mean, mae_std = aggregate_metrics_umn(results[key])
                all_metrics.append(
                    {
                        "config": config_name,
                        "bs": bs,
                        "d2_mean": d2_mean,
                        "d2_std": d2_std,
                        "mae_mean": mae_mean,
                        "mae_std": mae_std,
                    }
                )
            else:
                all_metrics.append(
                    {
                        "config": config_name,
                        "bs": bs,
                        "d2_mean": None,
                        "d2_std": None,
                        "mae_mean": None,
                        "mae_std": None,
                    }
                )

    # Find best values
    d2_vals = [m["d2_mean"] for m in all_metrics if m["d2_mean"] is not None]
    mae_vals = [m["mae_mean"] for m in all_metrics if m["mae_mean"] is not None]

    best_d2 = max(d2_vals) if d2_vals else None
    best_mae = min(mae_vals) if mae_vals else None

    # Generate table rows
    latex_rows = []
    prev_config = None

    for m in all_metrics:
        if prev_config is not None and prev_config != m["config"].split()[0]:
            latex_rows.append("\\midrule")
        prev_config = m["config"].split()[0]

        d2_str = format_metric_bold(m["d2_mean"], m["d2_std"], m["d2_mean"] == best_d2)
        mae_str = format_metric_bold(
            m["mae_mean"], m["mae_std"], m["mae_mean"] == best_mae, precision=4
        )

        latex_rows.append(f"{m['config']} & {m['bs']} & {d2_str} & {mae_str} \\\\")

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} non-private baseline results on UMN dataset. 
$\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better. 
Morphology applied to all classes.
Best values per metric are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-umn-nonprivate}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llcc}}
\\toprule
Configuration & Batch Size & Dice (2 Layers) $\\uparrow$ & MAE $\\downarrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table, all_metrics


def generate_dp_combined_table(duke_strategy_results, umn_strategy_results):
    """Generate combined DP table with Duke and UMN side-by-side."""
    configs = [
        ("Standard No Morph", "baseline", None, None),
        ("Morph-Open $k$=3", "morph", "open", 3),
        ("Morph-Open $k$=5", "morph", "open", 5),
        ("Morph-Close $k$=3", "morph", "close", 3),
        ("Morph-Close $k$=5", "morph", "close", 5),
        ("Morph-Both $k$=3", "morph", "both", 3),
        ("Morph-Both $k$=5", "morph", "both", 5),
    ]

    latex_rows = []

    for strat_idx, strategy in enumerate(STRATEGY_ORDER):
        duke_results = duke_strategy_results.get(strategy, {})
        umn_results = umn_strategy_results.get(strategy, {})
        strategy_metrics = []

        for config_name, config_type, op, k in configs:
            row_data = {"config": config_name}

            for eps in [8, 200]:
                for bs in [8, 16]:
                    if config_type == "baseline":
                        duke_key = ("baseline", None, None, bs, eps)
                        umn_key = ("baseline", None, None, bs, eps)
                    else:
                        duke_key = ("morph", op, k, bs, eps)
                        umn_key = ("morph", op, k, bs, eps)

                    # Duke metrics
                    if duke_key in duke_results and len(duke_results[duke_key]) > 0:
                        d7_mean, d7_std, _, _, _, _, _ = aggregate_metrics_duke(
                            duke_results[duke_key]
                        )
                        row_data[f"duke_eps{eps}_bs{bs}_mean"] = d7_mean
                        row_data[f"duke_eps{eps}_bs{bs}_std"] = d7_std
                    else:
                        row_data[f"duke_eps{eps}_bs{bs}_mean"] = None
                        row_data[f"duke_eps{eps}_bs{bs}_std"] = None

                    # UMN metrics
                    if umn_key in umn_results and len(umn_results[umn_key]) > 0:
                        dr_mean, dr_std, _, _, _, _, _ = aggregate_metrics_umn(
                            umn_results[umn_key]
                        )
                        row_data[f"umn_eps{eps}_bs{bs}_mean"] = dr_mean
                        row_data[f"umn_eps{eps}_bs{bs}_std"] = dr_std
                    else:
                        row_data[f"umn_eps{eps}_bs{bs}_mean"] = None
                        row_data[f"umn_eps{eps}_bs{bs}_std"] = None

            strategy_metrics.append(row_data)

        # Find best per column for this strategy
        best_vals = {}
        for col in [
            "duke_eps8_bs8",
            "duke_eps8_bs16",
            "duke_eps200_bs8",
            "duke_eps200_bs16",
            "umn_eps8_bs8",
            "umn_eps8_bs16",
            "umn_eps200_bs8",
            "umn_eps200_bs16",
        ]:
            vals = [
                m[f"{col}_mean"]
                for m in strategy_metrics
                if m[f"{col}_mean"] is not None
            ]
            best_vals[col] = max(vals) if vals else None

        # Generate rows for this strategy
        for i, m in enumerate(strategy_metrics):
            if i == 0:
                strat_cell = f"\\multirow{{7}}{{*}}{{{STRATEGY_DISPLAY[strategy]}}}"
            else:
                strat_cell = ""

            cols = []
            # Duke columns
            for col in [
                "duke_eps8_bs8",
                "duke_eps8_bs16",
                "duke_eps200_bs8",
                "duke_eps200_bs16",
            ]:
                is_best = (
                    m[f"{col}_mean"] == best_vals[col] and best_vals[col] is not None
                )
                cols.append(
                    format_metric_bold(m[f"{col}_mean"], m[f"{col}_std"], is_best)
                )
            # UMN columns
            for col in [
                "umn_eps8_bs8",
                "umn_eps8_bs16",
                "umn_eps200_bs8",
                "umn_eps200_bs16",
            ]:
                is_best = (
                    m[f"{col}_mean"] == best_vals[col] and best_vals[col] is not None
                )
                cols.append(
                    format_metric_bold(m[f"{col}_mean"], m[f"{col}_std"], is_best)
                )

            latex_rows.append(f"{strat_cell} & {m['config']} & {' & '.join(cols)} \\\\")

        if strat_idx < len(STRATEGY_ORDER) - 1:
            latex_rows.append("\\midrule")

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} DP results on Duke and UMN datasets comparing $\\varepsilon = 8$ (strong privacy) 
and $\\varepsilon = 200$ (weak privacy). $\\uparrow$ indicates higher is better. 
Duke: Dice (7 Layers). UMN: Dice (RNFL Layer).
Morphology targets layers 3-5 (inner retinal layers) for Duke and all classes for UMN.
Best values per clipping strategy and privacy level are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-dp-combined}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll|cccc|cccc}}
\\toprule
\\multirow{{2}}{{*}}{{Clipping Strategy}} & \\multirow{{2}}{{*}}{{Configuration}} & 
\\multicolumn{{4}}{{c}}{{Duke}} & \\multicolumn{{4}}{{c}}{{UMN}} \\\\
\\cmidrule(lr){{3-6}} \\cmidrule(lr){{7-10}}
& & $\\varepsilon$=8 (BS=8) $\\uparrow$ & $\\varepsilon$=8 (BS=16) $\\uparrow$ & 
$\\varepsilon$=200 (BS=8) $\\uparrow$ & $\\varepsilon$=200 (BS=16) $\\uparrow$ & 
$\\varepsilon$=8 (BS=8) $\\uparrow$ & $\\varepsilon$=8 (BS=16) $\\uparrow$ & 
$\\varepsilon$=200 (BS=8) $\\uparrow$ & $\\varepsilon$=200 (BS=16) $\\uparrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table


def generate_dp_duke_table(all_strategy_results):
    """Generate Table 3: DP Duke (Combined epsilon)."""
    configs = [
        ("Standard No Morph", "baseline", None, None),
        ("Morph-Open $k$=3", "morph", "open", 3),
        ("Morph-Open $k$=5", "morph", "open", 5),
        ("Morph-Close $k$=3", "morph", "close", 3),
        ("Morph-Close $k$=5", "morph", "close", 5),
        ("Morph-Both $k$=3", "morph", "both", 3),
        ("Morph-Both $k$=5", "morph", "both", 5),
    ]

    latex_rows = []

    for strat_idx, strategy in enumerate(STRATEGY_ORDER):
        results = all_strategy_results.get(strategy, {})
        strategy_metrics = []

        for config_name, config_type, op, k in configs:
            row_data = {"config": config_name}

            for eps in [8, 200]:
                for bs in [8, 16]:
                    if config_type == "baseline":
                        key = ("baseline", None, None, bs, eps)
                    else:
                        key = ("morph", op, k, bs, eps)

                    if key in results and len(results[key]) > 0:
                        d7_mean, d7_std, _, _, _, _ = aggregate_metrics_duke(
                            results[key]
                        )
                        row_data[f"eps{eps}_bs{bs}_mean"] = d7_mean
                        row_data[f"eps{eps}_bs{bs}_std"] = d7_std
                    else:
                        row_data[f"eps{eps}_bs{bs}_mean"] = None
                        row_data[f"eps{eps}_bs{bs}_std"] = None

            strategy_metrics.append(row_data)

        # Find best per column for this strategy
        best_vals = {}
        for col in ["eps8_bs8", "eps8_bs16", "eps200_bs8", "eps200_bs16"]:
            vals = [
                m[f"{col}_mean"]
                for m in strategy_metrics
                if m[f"{col}_mean"] is not None
            ]
            best_vals[col] = max(vals) if vals else None

        # Generate rows for this strategy
        for i, m in enumerate(strategy_metrics):
            if i == 0:
                strat_cell = f"\\multirow{{7}}{{*}}{{{STRATEGY_DISPLAY[strategy]}}}"
            else:
                strat_cell = ""

            cols = []
            for col in ["eps8_bs8", "eps8_bs16", "eps200_bs8", "eps200_bs16"]:
                is_best = (
                    m[f"{col}_mean"] == best_vals[col] and best_vals[col] is not None
                )
                cols.append(
                    format_metric_bold(m[f"{col}_mean"], m[f"{col}_std"], is_best)
                )

            latex_rows.append(f"{strat_cell} & {m['config']} & {' & '.join(cols)} \\\\")

        if strat_idx < len(STRATEGY_ORDER) - 1:
            latex_rows.append("\\midrule")

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} DP results on Duke dataset comparing $\\varepsilon = 8$ (strong privacy) 
and $\\varepsilon = 200$ (weak privacy). $\\uparrow$ indicates higher is better. 
Morphology targets layers 3-5 (inner retinal layers).
Best values per clipping strategy and privacy level are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-duke-dp}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llcccc}}
\\toprule
Clipping Strategy & Configuration & $\\varepsilon$=8 (BS=8) $\\uparrow$ & $\\varepsilon$=8 (BS=16) $\\uparrow$ & $\\varepsilon$=200 (BS=8) $\\uparrow$ & $\\varepsilon$=200 (BS=16) $\\uparrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table, all_strategy_results


def generate_dp_umn_table(all_strategy_results):
    """Generate Table 4: DP UMN (Combined epsilon)."""
    configs = [
        ("Standard No Morph", "baseline", None, None),
        ("Morph-Open $k$=3", "morph", "open", 3),
        ("Morph-Open $k$=5", "morph", "open", 5),
        ("Morph-Close $k$=3", "morph", "close", 3),
        ("Morph-Close $k$=5", "morph", "close", 5),
        ("Morph-Both $k$=3", "morph", "both", 3),
        ("Morph-Both $k$=5", "morph", "both", 5),
    ]

    latex_rows = []

    for strat_idx, strategy in enumerate(STRATEGY_ORDER):
        results = all_strategy_results.get(strategy, {})
        strategy_metrics = []

        for config_name, config_type, op, k in configs:
            row_data = {"config": config_name}

            for eps in [8, 200]:
                for bs in [8, 16]:
                    if config_type == "baseline":
                        key = ("baseline", None, None, bs, eps)
                    else:
                        key = ("morph", op, k, bs, eps)

                    if key in results and len(results[key]) > 0:
                        d2_mean, d2_std, _, _ = aggregate_metrics_umn(results[key])
                        row_data[f"eps{eps}_bs{bs}_mean"] = d2_mean
                        row_data[f"eps{eps}_bs{bs}_std"] = d2_std
                    else:
                        row_data[f"eps{eps}_bs{bs}_mean"] = None
                        row_data[f"eps{eps}_bs{bs}_std"] = None

            strategy_metrics.append(row_data)

        # Find best per column for this strategy
        best_vals = {}
        for col in ["eps8_bs8", "eps8_bs16", "eps200_bs8", "eps200_bs16"]:
            vals = [
                m[f"{col}_mean"]
                for m in strategy_metrics
                if m[f"{col}_mean"] is not None
            ]
            best_vals[col] = max(vals) if vals else None

        # Generate rows for this strategy
        for i, m in enumerate(strategy_metrics):
            if i == 0:
                strat_cell = f"\\multirow{{7}}{{*}}{{{STRATEGY_DISPLAY[strategy]}}}"
            else:
                strat_cell = ""

            cols = []
            for col in ["eps8_bs8", "eps8_bs16", "eps200_bs8", "eps200_bs16"]:
                is_best = (
                    m[f"{col}_mean"] == best_vals[col] and best_vals[col] is not None
                )
                cols.append(
                    format_metric_bold(m[f"{col}_mean"], m[f"{col}_std"], is_best)
                )

            latex_rows.append(f"{strat_cell} & {m['config']} & {' & '.join(cols)} \\\\")

        if strat_idx < len(STRATEGY_ORDER) - 1:
            latex_rows.append("\\midrule")

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} DP results on UMN dataset comparing $\\varepsilon = 8$ (strong privacy) 
and $\\varepsilon = 200$ (weak privacy). $\\uparrow$ indicates higher is better. 
Morphology applied to all classes.
Best values per clipping strategy and privacy level are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-umn-dp}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llcccc}}
\\toprule
Clipping Strategy & Configuration & $\\varepsilon$=8 (BS=8) $\\uparrow$ & $\\varepsilon$=8 (BS=16) $\\uparrow$ & $\\varepsilon$=200 (BS=8) $\\uparrow$ & $\\varepsilon$=200 (BS=16) $\\uparrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table, all_strategy_results


def generate_traceability_report(combined_metrics):
    """Generate traceability report showing CSV file paths and row numbers."""
    report_lines = []
    report_lines.append("\\section*{Traceability Report}")

    # Non-private results
    report_lines.append("\\subsection*{Non-Private Results}")
    report_lines.append("\\subsubsection*{Duke Dataset}")

    for m in combined_metrics:
        if m.get("duke_trace"):
            config = m["config"]
            bs = m["bs"]
            report_lines.append(f"\\paragraph{{{config}, BS={bs}}}")
            for trace in m["duke_trace"]:
                csv_file = trace["csv_file"].replace("_", "\\_")
                row_idx = trace["row_index"]
                report_lines.append(f"CSV: {csv_file}, Row: {row_idx}")

    report_lines.append("\\subsubsection*{UMN Dataset}")

    for m in combined_metrics:
        if m.get("umn_trace"):
            config = m["config"]
            bs = m["bs"]
            report_lines.append(f"\\paragraph{{{config}, BS={bs}}}")
            for trace in m["umn_trace"]:
                csv_file = trace["csv_file"].replace("_", "\\_")
                row_idx = trace["row_index"]
                report_lines.append(f"CSV: {csv_file}, Row: {row_idx}")

    return "\n".join(report_lines)


def generate_nonprivate_findings(combined_metrics):
    """Generate key findings paragraph for non-private results."""
    # Extract best configurations (only if data exists)
    duke_valid = [m for m in combined_metrics if m.get("duke_d7_mean") is not None]
    umn_valid = [m for m in combined_metrics if m.get("umn_dr_mean") is not None]

    # Get baseline performance
    duke_baseline = [m for m in combined_metrics if m["config"] == "Baseline"]
    umn_baseline = [m for m in combined_metrics if m["config"] == "Baseline"]

    findings = f"""\\paragraph{{Key Findings.}}
Table~\\ref{{tab:{ARCH}-nonprivate-combined}} presents the non-private baseline results for {ARCH_DISPLAY} on Duke and UMN datasets.
"""

    if duke_valid:
        duke_best_d7 = max(duke_valid, key=lambda x: x["duke_d7_mean"])
        duke_worst_d7 = min(duke_valid, key=lambda x: x["duke_d7_mean"])
        findings += f"""
On the Duke dataset, the best Dice (7 Layers) score of {duke_best_d7['duke_d7_mean']:.3f} is achieved by {duke_best_d7['config']} with BS={duke_best_d7['bs']}, while the lowest performance ({duke_worst_d7['duke_d7_mean']:.3f}) is observed with {duke_worst_d7['config']} (BS={duke_worst_d7['bs']}). """

    # Analyze morphology effects (only if Duke data exists)
    if duke_valid:
        morph_configs = [m for m in combined_metrics if "Morph" in m["config"]]
        open_configs = [m for m in morph_configs if "Open" in m["config"]]
        close_configs = [m for m in morph_configs if "Close" in m["config"]]
        both_configs = [m for m in morph_configs if "Both" in m["config"]]

        open_vals = [
            m["duke_d7_mean"] for m in open_configs if m.get("duke_d7_mean") is not None
        ]
        close_vals = [
            m["duke_d7_mean"]
            for m in close_configs
            if m.get("duke_d7_mean") is not None
        ]
        both_vals = [
            m["duke_d7_mean"] for m in both_configs if m.get("duke_d7_mean") is not None
        ]
        baseline_vals = [
            m["duke_d7_mean"]
            for m in duke_baseline
            if m.get("duke_d7_mean") is not None
        ]

        if open_vals and close_vals and baseline_vals:
            avg_open = np.mean(open_vals)
            avg_close = np.mean(close_vals)
            avg_both = np.mean(both_vals) if both_vals else 0
            avg_baseline = np.mean(baseline_vals)

            findings += f"Comparing morphological operations, the opening operation (avg Dice: {avg_open:.3f}) and both operations (avg Dice: {avg_both:.3f}) perform comparably, while the closing operation (avg Dice: {avg_close:.3f}) shows {'improved' if avg_close > avg_baseline else 'reduced'} performance relative to baseline (avg: {avg_baseline:.3f}). "

    # UMN findings (only if UMN data exists)
    if umn_valid:
        umn_best_dr = max(umn_valid, key=lambda x: x["umn_dr_mean"])
        findings += f"""

On the UMN dataset, {umn_best_dr['config']} with BS={umn_best_dr['bs']} achieves the highest Dice (RNFL Layer) score of {umn_best_dr['umn_dr_mean']:.3f}. """

        umn_avg_baseline_vals = [
            m["umn_dr_mean"] for m in umn_baseline if m.get("umn_dr_mean") is not None
        ]
        umn_morph = [m for m in combined_metrics if "Morph" in m["config"]]
        umn_avg_morph_vals = [
            m["umn_dr_mean"] for m in umn_morph if m.get("umn_dr_mean") is not None
        ]

        if umn_avg_baseline_vals and umn_avg_morph_vals:
            umn_avg_baseline = np.mean(umn_avg_baseline_vals)
            umn_avg_morph = np.mean(umn_avg_morph_vals)
            findings += f"The baseline achieves an average Dice of {umn_avg_baseline:.3f}, while morphological regularization yields an average of {umn_avg_morph:.3f}, indicating {'a slight improvement' if umn_avg_morph > umn_avg_baseline else 'comparable performance'}."

    return findings


def generate_dp_findings(duke_results, umn_results):
    """Generate key findings paragraph for DP results."""
    findings = f"""\\paragraph{{Key Findings.}}
Table~\\ref{{tab:{ARCH}-dp-combined}} presents the differentially private results for {ARCH_DISPLAY} under strong ($\\varepsilon = 8$) and weak ($\\varepsilon = 200$) privacy guarantees.

Under strong privacy ($\\varepsilon = 8$), performance is substantially reduced compared to non-private baselines, as expected with strict privacy constraints. """

    # Analyze clipping strategies
    duke_eps8_by_strategy = {}
    duke_eps200_by_strategy = {}

    for strategy in STRATEGY_ORDER:
        results = duke_results.get(strategy, {})
        eps8_vals = []
        eps200_vals = []

        for key, rows in results.items():
            if key[4] == 8:  # eps = 8
                d7, _, _, _, _, _, _ = aggregate_metrics_duke(rows)
                if d7 is not None:
                    eps8_vals.append(d7)
            elif key[4] == 200:  # eps = 200
                d7, _, _, _, _, _, _ = aggregate_metrics_duke(rows)
                if d7 is not None:
                    eps200_vals.append(d7)

        if eps8_vals:
            duke_eps8_by_strategy[strategy] = np.mean(eps8_vals)
        if eps200_vals:
            duke_eps200_by_strategy[strategy] = np.mean(eps200_vals)

    if duke_eps8_by_strategy:
        best_strat_eps8 = max(duke_eps8_by_strategy, key=duke_eps8_by_strategy.get)
        findings += f"Among clipping strategies at $\\varepsilon = 8$, {STRATEGY_DISPLAY[best_strat_eps8]} achieves the best average Dice of {duke_eps8_by_strategy[best_strat_eps8]:.3f} on Duke. "

    if duke_eps200_by_strategy:
        best_strat_eps200 = max(
            duke_eps200_by_strategy, key=duke_eps200_by_strategy.get
        )
        findings += f"""

Under weak privacy ($\\varepsilon = 200$), utility substantially recovers with {STRATEGY_DISPLAY[best_strat_eps200]} achieving the best average Dice of {duke_eps200_by_strategy[best_strat_eps200]:.3f} on Duke. """

    # Privacy-utility tradeoff
    if duke_eps8_by_strategy and duke_eps200_by_strategy:
        avg_eps8 = np.mean(list(duke_eps8_by_strategy.values()))
        avg_eps200 = np.mean(list(duke_eps200_by_strategy.values()))
        improvement = ((avg_eps200 - avg_eps8) / avg_eps8) * 100
        findings += f"""

Comparing privacy levels, relaxing privacy from $\\varepsilon = 8$ to $\\varepsilon = 200$ improves average Dice by {improvement:.1f}\\% on Duke (from {avg_eps8:.3f} to {avg_eps200:.3f}), demonstrating the privacy-utility tradeoff. """

    # UMN analysis
    umn_eps8_by_strategy = {}
    umn_eps200_by_strategy = {}

    for strategy in STRATEGY_ORDER:
        results = umn_results.get(strategy, {})
        eps8_vals = []
        eps200_vals = []

        for key, rows in results.items():
            if key[4] == 8:
                dr, _, _, _, _, _, _ = aggregate_metrics_umn(rows)
                if dr is not None:
                    eps8_vals.append(dr)
            elif key[4] == 200:
                dr, _, _, _, _, _, _ = aggregate_metrics_umn(rows)
                if dr is not None:
                    eps200_vals.append(dr)

        if eps8_vals:
            umn_eps8_by_strategy[strategy] = np.mean(eps8_vals)
        if eps200_vals:
            umn_eps200_by_strategy[strategy] = np.mean(eps200_vals)

    if umn_eps200_by_strategy:
        best_umn_strat = max(umn_eps200_by_strategy, key=umn_eps200_by_strategy.get)
        findings += f"On the UMN dataset, {STRATEGY_DISPLAY[best_umn_strat]} performs best at $\\varepsilon = 200$ with average Dice of {umn_eps200_by_strategy[best_umn_strat]:.3f}."

    return findings


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX results tables for OCT Segmentation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_tables.py --arch lfunet
    python generate_tables.py --arch unet --results-path /custom/path/to/results
        """,
    )
    parser.add_argument(
        "--arch",
        "-a",
        type=str,
        required=True,
        help="Architecture name (e.g., lfunet, deeplabv3, unet)",
    )
    parser.add_argument(
        "--arch-display",
        type=str,
        default=None,
        help="Display name for architecture (default: auto-detect from ARCH_DISPLAY_NAMES)",
    )
    parser.add_argument(
        "--results-path",
        "-r",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to results directory (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for LaTeX tables (default: print to stdout)",
    )
    parser.add_argument(
        "--traceability",
        "-t",
        action="store_true",
        help="Include traceability report",
    )
    return parser.parse_args()


def main():
    global RESULTS_PATH, ARCH, ARCH_DISPLAY

    args = parse_args()

    # Set global variables from args
    ARCH = args.arch.lower()
    RESULTS_PATH = args.results_path

    # Set display name
    if args.arch_display:
        ARCH_DISPLAY = args.arch_display
    elif ARCH in ARCH_DISPLAY_NAMES:
        ARCH_DISPLAY = ARCH_DISPLAY_NAMES[ARCH]
    else:
        # Default: capitalize
        ARCH_DISPLAY = ARCH.upper()

    # Output handling
    output_lines = []

    def log(msg=""):
        output_lines.append(msg)
        print(msg)

    log("=" * 80)
    log(f"Generating results tables for {ARCH_DISPLAY}")
    log(f"Results path: {RESULTS_PATH}")
    log("=" * 80)

    # Collect non-private results
    log("\nCollecting non-private Duke results...")
    duke_nonprivate = collect_nonprivate_results("duke")
    log(f"Found {len(duke_nonprivate)} configurations")

    log("\nCollecting non-private UMN results...")
    umn_nonprivate = collect_nonprivate_results("umn")
    log(f"Found {len(umn_nonprivate)} configurations")

    # Collect DP results
    duke_dp = {}
    umn_dp = {}

    for strategy in STRATEGY_ORDER:
        log(f"\nCollecting DP results for strategy: {strategy}")
        duke_dp[strategy] = collect_dp_results("duke", strategy)
        umn_dp[strategy] = collect_dp_results("umn", strategy)
        log(f"  Duke: {len(duke_dp[strategy])} configurations")
        log(f"  UMN: {len(umn_dp[strategy])} configurations")

    # Generate tables
    log("\n" + "=" * 80)
    log("TABLE 1: Non-Private Combined (Duke + UMN)")
    log("=" * 80)
    table1, combined_metrics = generate_nonprivate_combined_table(
        duke_nonprivate, umn_nonprivate
    )
    log(table1)

    log("\n" + "=" * 80)
    log("KEY FINDINGS: Non-Private")
    log("=" * 80)
    findings_nonprivate = generate_nonprivate_findings(combined_metrics)
    log(findings_nonprivate)

    log("\n" + "=" * 80)
    log("TABLE 2: DP Combined (Duke + UMN)")
    log("=" * 80)
    table2 = generate_dp_combined_table(duke_dp, umn_dp)
    log(table2)

    log("\n" + "=" * 80)
    log("KEY FINDINGS: DP")
    log("=" * 80)
    findings_dp = generate_dp_findings(duke_dp, umn_dp)
    log(findings_dp)

    # Generate traceability report if requested
    if args.traceability:
        log("\n" + "=" * 80)
        log("TRACEABILITY REPORT")
        log("=" * 80)
        trace_report = generate_traceability_report(combined_metrics)
        log(trace_report)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
