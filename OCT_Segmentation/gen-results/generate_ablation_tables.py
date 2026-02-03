#!/usr/bin/env python3
"""
Generate LaTeX ablation study tables for learning rate and weight decay.

Usage:
    python generate_ablation_tables.py --arch lfunet
    python generate_ablation_tables.py --arch unet --output ablation_unet.tex
"""

import os
import pandas as pd
import numpy as np
import ast
import re
import argparse
from collections import defaultdict
from pathlib import Path

# Default paths (ablation .txt files live in slurm/, script in gen-results/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_PATH = os.path.join(_script_dir, "..", "results")
DEFAULT_SLURM_DIR = os.path.join(_script_dir, "..", "slurm")

# Architecture display name mapping
ARCH_DISPLAY_NAMES = {
    "lfunet": "LF-UNet",
    "unet": "U-Net",
    "nestedunet": "U-Net++",
}

# Model name mapping (from ablation files to directory names)
MODEL_NAME_MAP = {
    "LFUNet": "lfunet",
    "unet": "unet",
    "NestedUNet": "nestedunet",
}

# Global variables
RESULTS_PATH = None
ARCH = None
ARCH_DISPLAY = None
SLURM_DIR = None


def parse_dice_all(dice_str):
    """Parse Dice_All string to list of floats."""
    try:
        return ast.literal_eval(dice_str)
    except:
        return None


def parse_ablation_file(ablation_file):
    """
    Parse ablation text file to extract weight_decay mappings.
    Returns dict: (model, dataset, bs, run, lr) -> list of weight_decay values
    Since same (model, dataset, bs, run, lr) can have different weight_decay,
    we store all possible values.
    """
    mapping = {}

    if not os.path.exists(ablation_file):
        return mapping

    with open(ablation_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse command line
            # Example: python train-one-gpu.py --model_name LFUNet --dataset Duke --morphology False --DPSGD False --batch_size 8 --learning_rate 0.001 --weight_decay 0.0001 --run_number 1

            # Extract model_name
            model_match = re.search(r"--model_name\s+(\w+)", line)
            dataset_match = re.search(r"--dataset\s+(\w+)", line)
            bs_match = re.search(r"--batch_size\s+(\d+)", line)
            lr_match = re.search(r"--learning_rate\s+([\d.e-]+)", line)
            wd_match = re.search(r"--weight_decay\s+([\d.e-]+)", line)
            run_match = re.search(r"--run_number\s+(\d+)", line)

            if all(
                [model_match, dataset_match, bs_match, lr_match, wd_match, run_match]
            ):
                model = model_match.group(1)
                dataset = dataset_match.group(1).lower()
                bs = int(bs_match.group(1))
                lr = float(lr_match.group(1))
                wd_str = wd_match.group(1)
                run = int(run_match.group(1))

                # Convert weight_decay string to float
                if "e-" in wd_str.lower():
                    wd = float(wd_str)
                else:
                    wd = float(wd_str)

                # Normalize model name
                model_lower = MODEL_NAME_MAP.get(model, model.lower())

                key = (model_lower, dataset, bs, run, lr)
                if key not in mapping:
                    mapping[key] = []
                mapping[key].append(wd)

    return mapping


def load_ablation_mappings(arch):
    """
    Load weight_decay mappings from all ablation files for an architecture.
    Returns dict: (model, dataset, bs, run, lr) -> list of (wd, order_index) tuples
    where order_index is the position in the ablation file for that key.
    """
    all_mappings = {}
    order_tracker = {}  # Track order for each key

    datasets = ["duke", "umn"]

    for dataset in datasets:
        ablation_file = os.path.join(
            DEFAULT_SLURM_DIR, f"{arch}-{dataset}-ablation.txt"
        )
        if not os.path.exists(ablation_file):
            continue

        with open(ablation_file, "r") as f:
            for order_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                model_match = re.search(r"--model_name\s+(\w+)", line)
                dataset_match = re.search(r"--dataset\s+(\w+)", line)
                bs_match = re.search(r"--batch_size\s+(\d+)", line)
                lr_match = re.search(r"--learning_rate\s+([\d.e-]+)", line)
                wd_match = re.search(r"--weight_decay\s+([\d.e-]+)", line)
                run_match = re.search(r"--run_number\s+(\d+)", line)

                if all(
                    [
                        model_match,
                        dataset_match,
                        bs_match,
                        lr_match,
                        wd_match,
                        run_match,
                    ]
                ):
                    model = MODEL_NAME_MAP.get(
                        model_match.group(1), model_match.group(1).lower()
                    )
                    dataset_lower = dataset_match.group(1).lower()
                    bs = int(bs_match.group(1))
                    lr = float(lr_match.group(1))
                    wd_str = wd_match.group(1)
                    run = int(run_match.group(1))

                    wd = float(wd_str)
                    key = (model, dataset_lower, bs, run, lr)

                    if key not in all_mappings:
                        all_mappings[key] = []
                        order_tracker[key] = 0

                    all_mappings[key].append((wd, order_tracker[key]))
                    order_tracker[key] += 1

    return all_mappings


def calculate_metrics_duke(row):
    """Calculate Duke metrics from a row."""
    dice_all = parse_dice_all(row["Dice_All"])
    if dice_all is None or len(dice_all) < 9:
        return None, None

    # Dice (7 Layers): indices 1-7 (excluding background at 0 and choroid/fluid at 8)
    dice_7 = np.mean(dice_all[1:8])
    # Dice (All Layers): all 9 values
    dice_all_mean = np.mean(dice_all)

    return dice_7, dice_all_mean


def calculate_metrics_umn(row):
    """Calculate UMN metrics from a row."""
    dice_all = parse_dice_all(row["Dice_All"])
    if dice_all is None or len(dice_all) < 2:
        return None, None

    # Dice (1 Layer - RNFL): index 1 (RNFL layer)
    dice_rnfl = dice_all[1] if len(dice_all) > 1 else None
    # Dice (All Layers): indices 0-1 (background + RNFL)
    dice_all_mean = np.mean(dice_all[0:2])

    return dice_rnfl, dice_all_mean


def collect_ablation_results(dataset, wd_mapping):
    """Collect ablation study results from base directories."""
    results = defaultdict(list)

    base_dir = f"{ARCH}-{dataset}-base"
    base_path = os.path.join(RESULTS_PATH, base_dir)

    if not os.path.exists(base_path):
        return results

    for subdir in os.listdir(base_path):
        match = re.match(r"bs(\d+)_run(\d+)", subdir)
        if match:
            bs = int(match.group(1))
            run = int(match.group(2))
            results_file = os.path.join(base_path, subdir, "test", "results.csv")

            if os.path.exists(results_file):
                try:
                    df = pd.read_csv(results_file)

                    # Process each row in the CSV
                    # Track order of rows with same LR to match with ablation file order
                    lr_order_tracker = defaultdict(int)

                    for idx, row in df.iterrows():
                        lr = row["Learning_Rate"]

                        # Check if weight_decay is already in CSV
                        if "Weight_Decay" in row and pd.notna(row["Weight_Decay"]):
                            wd = float(row["Weight_Decay"])
                        else:
                            # Match to weight_decay using mapping and order
                            key = (ARCH, dataset, bs, run, lr)
                            wd_list = wd_mapping.get(key, [])

                            # Match by order: use the weight_decay at the same position
                            if wd_list:
                                order_idx = lr_order_tracker[(lr, bs, run)]
                                if order_idx < len(wd_list):
                                    wd, _ = wd_list[
                                        order_idx
                                    ]  # Get weight_decay, ignore stored order
                                    lr_order_tracker[(lr, bs, run)] += 1
                                else:
                                    # Fallback: use first weight_decay
                                    wd, _ = wd_list[0]
                            else:
                                wd = None

                        if wd is not None:
                            # Store with traceability info
                            trace_info = {
                                "csv_file": results_file,
                                "row_index": idx + 2,  # +2 for header and 0-index
                                "lr": lr,
                                "wd": wd,
                                "bs": bs,
                                "run": run,
                            }

                            if dataset == "duke":
                                dice_7, dice_all = calculate_metrics_duke(row)
                                if dice_7 is not None:
                                    results[(lr, wd, bs)].append(
                                        {
                                            "run": run,
                                            "dice_7": dice_7,
                                            "dice_all": dice_all,
                                            "trace": trace_info,
                                        }
                                    )
                            else:  # umn
                                dice_rnfl, dice_all = calculate_metrics_umn(row)
                                if dice_rnfl is not None:
                                    results[(lr, wd, bs)].append(
                                        {
                                            "run": run,
                                            "dice_rnfl": dice_rnfl,
                                            "dice_all": dice_all,
                                            "trace": trace_info,
                                        }
                                    )

                except Exception as e:
                    print(f"Error processing {results_file}: {e}")

    return results


def aggregate_metrics_duke(rows_list):
    """Aggregate Duke metrics over runs."""
    dice_7_vals = []
    dice_all_vals = []
    trace_info_list = []

    for row_data in rows_list:
        dice_7_vals.append(row_data["dice_7"])
        dice_all_vals.append(row_data["dice_all"])
        trace_info_list.append(row_data["trace"])

    if len(dice_7_vals) == 0:
        return None, None, None, None, []

    return (
        np.mean(dice_7_vals),
        np.std(dice_7_vals),
        np.mean(dice_all_vals),
        np.std(dice_all_vals),
        trace_info_list,
    )


def aggregate_metrics_umn(rows_list):
    """Aggregate UMN metrics over runs."""
    dice_rnfl_vals = []
    dice_all_vals = []
    trace_info_list = []

    for row_data in rows_list:
        dice_rnfl_vals.append(row_data["dice_rnfl"])
        dice_all_vals.append(row_data["dice_all"])
        trace_info_list.append(row_data["trace"])

    if len(dice_rnfl_vals) == 0:
        return None, None, None, None, []

    return (
        np.mean(dice_rnfl_vals),
        np.std(dice_rnfl_vals),
        np.mean(dice_all_vals),
        np.std(dice_all_vals),
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


def format_wd(wd):
    """Format weight decay for display."""
    if wd == 1e-4:
        return "$1 \\times 10^{-4}$"
    elif abs(wd - 1e-9) < 1e-10:
        return "$1 \\times 10^{-9}$"
    else:
        return (
            f"${wd:.0e}$".replace("e-0", " \\times 10^{-").replace(
                "e+", " \\times 10^{"
            )
            + "}"
        )


def format_lr(lr):
    """Format learning rate for display."""
    if lr == 0.001:
        return "$1 \\times 10^{-3}$"
    elif lr == 0.0005:
        return "$5 \\times 10^{-4}$"
    else:
        return (
            f"${lr:.0e}$".replace("e-0", " \\times 10^{-").replace(
                "e+", " \\times 10^{"
            )
            + "}"
        )


def generate_ablation_tables(duke_results, umn_results):
    """Generate side-by-side ablation tables for Duke and UMN."""

    # Learning rates and weight decays to test
    learning_rates = [0.001, 0.0005]
    weight_decays = [1e-4, 1e-9]
    batch_sizes = [8, 16]

    # Collect all metrics
    duke_metrics = []
    umn_metrics = []

    for lr in learning_rates:
        for wd in weight_decays:
            for bs in batch_sizes:
                key = (lr, wd, bs)

                # Duke metrics
                if key in duke_results and len(duke_results[key]) > 0:
                    d7_mean, d7_std, da_mean, da_std, trace = aggregate_metrics_duke(
                        duke_results[key]
                    )
                    duke_metrics.append(
                        {
                            "lr": lr,
                            "wd": wd,
                            "bs": bs,
                            "d7_mean": d7_mean,
                            "d7_std": d7_std,
                            "da_mean": da_mean,
                            "da_std": da_std,
                            "trace": trace,
                        }
                    )
                else:
                    duke_metrics.append(
                        {
                            "lr": lr,
                            "wd": wd,
                            "bs": bs,
                            "d7_mean": None,
                            "d7_std": None,
                            "da_mean": None,
                            "da_std": None,
                            "trace": [],
                        }
                    )

                # UMN metrics
                if key in umn_results and len(umn_results[key]) > 0:
                    dr_mean, dr_std, da_mean, da_std, trace = aggregate_metrics_umn(
                        umn_results[key]
                    )
                    umn_metrics.append(
                        {
                            "lr": lr,
                            "wd": wd,
                            "bs": bs,
                            "dr_mean": dr_mean,
                            "dr_std": dr_std,
                            "da_mean": da_mean,
                            "da_std": da_std,
                            "trace": trace,
                        }
                    )
                else:
                    umn_metrics.append(
                        {
                            "lr": lr,
                            "wd": wd,
                            "bs": bs,
                            "dr_mean": None,
                            "dr_std": None,
                            "da_mean": None,
                            "da_std": None,
                            "trace": [],
                        }
                    )

    # Find best values for each metric
    duke_d7_vals = [m["d7_mean"] for m in duke_metrics if m["d7_mean"] is not None]
    duke_da_vals = [m["da_mean"] for m in duke_metrics if m["da_mean"] is not None]
    umn_dr_vals = [m["dr_mean"] for m in umn_metrics if m["dr_mean"] is not None]
    umn_da_vals = [m["da_mean"] for m in umn_metrics if m["da_mean"] is not None]

    best_duke_d7 = max(duke_d7_vals) if duke_d7_vals else None
    best_duke_da = max(duke_da_vals) if duke_da_vals else None
    best_umn_dr = max(umn_dr_vals) if umn_dr_vals else None
    best_umn_da = max(umn_da_vals) if umn_da_vals else None

    # Generate table rows
    latex_rows = []

    for lr in learning_rates:
        for wd in weight_decays:
            row_parts = []

            # LR and WD columns
            lr_str = format_lr(lr)
            wd_str = format_wd(wd)
            row_parts.append(f"{lr_str} & {wd_str}")

            # Duke columns (BS=8 and BS=16)
            for bs in batch_sizes:
                duke_m = next(
                    (
                        m
                        for m in duke_metrics
                        if m["lr"] == lr and m["wd"] == wd and m["bs"] == bs
                    ),
                    None,
                )
                if duke_m:
                    d7_str = format_metric_bold(
                        duke_m["d7_mean"],
                        duke_m["d7_std"],
                        duke_m["d7_mean"] == best_duke_d7,
                    )
                    da_str = format_metric_bold(
                        duke_m["da_mean"],
                        duke_m["da_std"],
                        duke_m["da_mean"] == best_duke_da,
                    )
                    row_parts.append(f"{d7_str} & {da_str}")
                else:
                    row_parts.append("--- & ---")

            # UMN columns (BS=8 and BS=16)
            for bs in batch_sizes:
                umn_m = next(
                    (
                        m
                        for m in umn_metrics
                        if m["lr"] == lr and m["wd"] == wd and m["bs"] == bs
                    ),
                    None,
                )
                if umn_m:
                    dr_str = format_metric_bold(
                        umn_m["dr_mean"],
                        umn_m["dr_std"],
                        umn_m["dr_mean"] == best_umn_dr,
                    )
                    da_str = format_metric_bold(
                        umn_m["da_mean"],
                        umn_m["da_std"],
                        umn_m["da_mean"] == best_umn_da,
                    )
                    row_parts.append(f"{dr_str} & {da_str}")
                else:
                    row_parts.append("--- & ---")

            latex_rows.append(" & ".join(row_parts) + " \\\\")

        # Add midrule between different learning rates
        if lr != learning_rates[-1]:
            latex_rows.append("\\midrule")

    # Build side-by-side table
    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} ablation study results for learning rate and weight decay on Duke and UMN datasets. 
$\\uparrow$ indicates higher is better. 
Duke: Dice (7 Layers) and Dice (All Layers). 
UMN: Dice (RNFL Layer) and Dice (All Layers).
Best values per metric are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-ablation}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll|cc|cc|cc|cc}}
\\toprule
\\multirow{{2}}{{*}}{{Learning Rate}} & \\multirow{{2}}{{*}}{{Weight Decay}} & 
\\multicolumn{{2}}{{c}}{{Duke (BS=8)}} & \\multicolumn{{2}}{{c}}{{Duke (BS=16)}} & 
\\multicolumn{{2}}{{c}}{{UMN (BS=8)}} & \\multicolumn{{2}}{{c}}{{UMN (BS=16)}} \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}} \\cmidrule(lr){{7-8}} \\cmidrule(lr){{9-10}}
& & Dice (7) $\\uparrow$ & Dice (All) $\\uparrow$ & Dice (7) $\\uparrow$ & Dice (All) $\\uparrow$ & 
Dice (RNFL) $\\uparrow$ & Dice (All) $\\uparrow$ & Dice (RNFL) $\\uparrow$ & Dice (All) $\\uparrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table, duke_metrics, umn_metrics


def generate_traceability_report(duke_metrics, umn_metrics):
    """Generate traceability report showing CSV file paths and row numbers."""
    report_lines = []
    report_lines.append("\\section*{Traceability Report}")
    report_lines.append("\\subsection*{Duke Dataset}")

    for m in duke_metrics:
        if m["trace"]:
            lr = m["lr"]
            wd = m["wd"]
            bs = m["bs"]
            report_lines.append(f"\\paragraph{{LR={lr}, WD={wd}, BS={bs}}}")
            for trace in m["trace"]:
                csv_file = trace["csv_file"].replace("_", "\\_")
                row_idx = trace["row_index"]
                report_lines.append(f"CSV: {csv_file}, Row: {row_idx}")

    report_lines.append("\\subsection*{UMN Dataset}")

    for m in umn_metrics:
        if m["trace"]:
            lr = m["lr"]
            wd = m["wd"]
            bs = m["bs"]
            report_lines.append(f"\\paragraph{{LR={lr}, WD={wd}, BS={bs}}}")
            for trace in m["trace"]:
                csv_file = trace["csv_file"].replace("_", "\\_")
                row_idx = trace["row_index"]
                report_lines.append(f"CSV: {csv_file}, Row: {row_idx}")

    return "\n".join(report_lines)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX ablation study tables for OCT Segmentation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_ablation_tables.py --arch lfunet
    python generate_ablation_tables.py --arch unet --output ablation_unet.tex
        """,
    )
    parser.add_argument(
        "--arch",
        "-a",
        type=str,
        required=True,
        help="Architecture name (e.g., lfunet, unet, nestedunet)",
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
    global RESULTS_PATH, ARCH, ARCH_DISPLAY, SLURM_DIR

    args = parse_args()

    # Set global variables
    ARCH = args.arch.lower()
    RESULTS_PATH = args.results_path
    SLURM_DIR = DEFAULT_SLURM_DIR

    # Set display name
    if args.arch_display:
        ARCH_DISPLAY = args.arch_display
    elif ARCH in ARCH_DISPLAY_NAMES:
        ARCH_DISPLAY = ARCH_DISPLAY_NAMES[ARCH]
    else:
        ARCH_DISPLAY = ARCH.upper()

    # Output handling
    output_lines = []

    def log(msg=""):
        output_lines.append(msg)
        print(msg)

    log("=" * 80)
    log(f"Generating ablation study tables for {ARCH_DISPLAY}")
    log(f"Results path: {RESULTS_PATH}")
    log("=" * 80)

    # Load weight_decay mappings from ablation files
    log("\nLoading weight_decay mappings from ablation files...")
    wd_mapping = load_ablation_mappings(ARCH)
    log(f"Found {len(wd_mapping)} weight_decay mappings")

    # Collect results
    log("\nCollecting Duke ablation results...")
    duke_results = collect_ablation_results("duke", wd_mapping)
    log(f"Found {len(duke_results)} Duke configurations")

    log("\nCollecting UMN ablation results...")
    umn_results = collect_ablation_results("umn", wd_mapping)
    log(f"Found {len(umn_results)} UMN configurations")

    # Generate tables
    log("\n" + "=" * 80)
    log("GENERATING ABLATION TABLES")
    log("=" * 80)
    table, duke_metrics, umn_metrics = generate_ablation_tables(
        duke_results, umn_results
    )
    log(table)

    # Generate traceability report if requested
    if args.traceability:
        log("\n" + "=" * 80)
        log("TRACEABILITY REPORT")
        log("=" * 80)
        trace_report = generate_traceability_report(duke_metrics, umn_metrics)
        log(trace_report)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
