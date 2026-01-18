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
DEFAULT_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results")

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
        return None, None

    # Dice (2 Layers): indices 0-1 (background + RNFL)
    dice_2 = np.mean(dice_all[0:2])
    mae = row["MAE"]

    return dice_2, mae


def collect_nonprivate_results(dataset):
    """Collect non-private baseline and morph results."""
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
                        results[("baseline", None, None, bs)].append((run, row))

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
                        results[("morph", op, k, bs)].append((run, row))

    return results


def collect_dp_results(dataset, strategy):
    """Collect DP baseline and morph results for a given strategy."""
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
                        results[("baseline", None, None, bs, eps)].append((run, row))

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
                        results[("morph", op, k, bs, eps)].append((run, row))

    return results


def aggregate_metrics_duke(rows_list):
    """Aggregate metrics over runs for Duke dataset."""
    dice_7_vals = []
    dice_all_vals = []
    mae_vals = []

    for run, row in rows_list:
        d7, da, mae = calculate_metrics_duke(row)
        if d7 is not None:
            dice_7_vals.append(d7)
            dice_all_vals.append(da)
            mae_vals.append(mae)

    if len(dice_7_vals) == 0:
        return None, None, None, None, None, None

    return (
        np.mean(dice_7_vals),
        np.std(dice_7_vals),
        np.mean(dice_all_vals),
        np.std(dice_all_vals),
        np.mean(mae_vals),
        np.std(mae_vals),
    )


def aggregate_metrics_umn(rows_list):
    """Aggregate metrics over runs for UMN dataset."""
    dice_2_vals = []
    mae_vals = []

    for run, row in rows_list:
        d2, mae = calculate_metrics_umn(row)
        if d2 is not None:
            dice_2_vals.append(d2)
            mae_vals.append(mae)

    if len(dice_2_vals) == 0:
        return None, None, None, None

    return (
        np.mean(dice_2_vals),
        np.std(dice_2_vals),
        np.mean(mae_vals),
        np.std(mae_vals),
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


def generate_nonprivate_findings(duke_metrics, umn_metrics):
    """Generate key findings paragraph for non-private results."""
    # Extract best configurations
    duke_best_d7 = max(
        [m for m in duke_metrics if m["d7_mean"] is not None],
        key=lambda x: x["d7_mean"],
    )
    duke_worst_d7 = min(
        [m for m in duke_metrics if m["d7_mean"] is not None],
        key=lambda x: x["d7_mean"],
    )

    umn_best_d2 = max(
        [m for m in umn_metrics if m["d2_mean"] is not None], key=lambda x: x["d2_mean"]
    )
    umn_worst_d2 = min(
        [m for m in umn_metrics if m["d2_mean"] is not None], key=lambda x: x["d2_mean"]
    )

    # Get baseline performance
    duke_baseline = [m for m in duke_metrics if m["config"] == "Baseline"]
    umn_baseline = [m for m in umn_metrics if m["config"] == "Baseline"]

    findings = f"""\\paragraph{{Key Findings.}}
Tables~\\ref{{tab:{ARCH}-duke-nonprivate}} and~\\ref{{tab:{ARCH}-umn-nonprivate}} present the non-private baseline results for {ARCH_DISPLAY} on Duke and UMN datasets respectively.

On the Duke dataset, the best Dice (7 Layers) score of {duke_best_d7['d7_mean']:.3f} is achieved by {duke_best_d7['config']} with BS={duke_best_d7['bs']}, while the lowest performance ({duke_worst_d7['d7_mean']:.3f}) is observed with {duke_worst_d7['config']} (BS={duke_worst_d7['bs']}). """

    # Analyze morphology effects
    morph_configs = [m for m in duke_metrics if "Morph" in m["config"]]
    open_configs = [m for m in morph_configs if "Open" in m["config"]]
    close_configs = [m for m in morph_configs if "Close" in m["config"]]
    both_configs = [m for m in morph_configs if "Both" in m["config"]]

    if open_configs and close_configs:
        avg_open = np.mean(
            [m["d7_mean"] for m in open_configs if m["d7_mean"] is not None]
        )
        avg_close = np.mean(
            [m["d7_mean"] for m in close_configs if m["d7_mean"] is not None]
        )
        avg_both = np.mean(
            [m["d7_mean"] for m in both_configs if m["d7_mean"] is not None]
        )
        avg_baseline = np.mean(
            [m["d7_mean"] for m in duke_baseline if m["d7_mean"] is not None]
        )

        findings += f"Comparing morphological operations, the opening operation (avg Dice: {avg_open:.3f}) and both operations (avg Dice: {avg_both:.3f}) perform comparably, while the closing operation (avg Dice: {avg_close:.3f}) shows {'improved' if avg_close > avg_baseline else 'reduced'} performance relative to baseline (avg: {avg_baseline:.3f}). "

    findings += f"""

On the UMN dataset, {umn_best_d2['config']} with BS={umn_best_d2['bs']} achieves the highest Dice (2 Layers) score of {umn_best_d2['d2_mean']:.3f}. """

    umn_avg_baseline = np.mean(
        [m["d2_mean"] for m in umn_baseline if m["d2_mean"] is not None]
    )
    umn_morph = [m for m in umn_metrics if "Morph" in m["config"]]
    umn_avg_morph = np.mean(
        [m["d2_mean"] for m in umn_morph if m["d2_mean"] is not None]
    )

    findings += f"The baseline achieves an average Dice of {umn_avg_baseline:.3f}, while morphological regularization yields an average of {umn_avg_morph:.3f}, indicating {'a slight improvement' if umn_avg_morph > umn_avg_baseline else 'comparable performance'}."

    return findings


def generate_dp_findings(duke_results, umn_results):
    """Generate key findings paragraph for DP results."""
    findings = f"""\\paragraph{{Key Findings.}}
Tables~\\ref{{tab:{ARCH}-duke-dp}} and~\\ref{{tab:{ARCH}-umn-dp}} present the differentially private results for {ARCH_DISPLAY} under strong ($\\varepsilon = 8$) and weak ($\\varepsilon = 200$) privacy guarantees.

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
                d7, _, _, _, _, _ = aggregate_metrics_duke(rows)
                if d7 is not None:
                    eps8_vals.append(d7)
            elif key[4] == 200:  # eps = 200
                d7, _, _, _, _, _ = aggregate_metrics_duke(rows)
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
                d2, _, _, _ = aggregate_metrics_umn(rows)
                if d2 is not None:
                    eps8_vals.append(d2)
            elif key[4] == 200:
                d2, _, _, _ = aggregate_metrics_umn(rows)
                if d2 is not None:
                    eps200_vals.append(d2)

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
    python generate_tables.py --arch deeplabv3 --output deeplabv3_tables.tex
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
    log("TABLE 1: Non-Private Duke")
    log("=" * 80)
    table1, duke_metrics = generate_nonprivate_duke_table(duke_nonprivate)
    log(table1)

    log("\n" + "=" * 80)
    log("TABLE 2: Non-Private UMN")
    log("=" * 80)
    table2, umn_metrics = generate_nonprivate_umn_table(umn_nonprivate)
    log(table2)

    log("\n" + "=" * 80)
    log("KEY FINDINGS: Non-Private")
    log("=" * 80)
    findings_nonprivate = generate_nonprivate_findings(duke_metrics, umn_metrics)
    log(findings_nonprivate)

    log("\n" + "=" * 80)
    log("TABLE 3: DP Duke")
    log("=" * 80)
    table3, _ = generate_dp_duke_table(duke_dp)
    log(table3)

    log("\n" + "=" * 80)
    log("TABLE 4: DP UMN")
    log("=" * 80)
    table4, _ = generate_dp_umn_table(umn_dp)
    log(table4)

    log("\n" + "=" * 80)
    log("KEY FINDINGS: DP")
    log("=" * 80)
    findings_dp = generate_dp_findings(duke_dp, umn_dp)
    log(findings_dp)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
