#!/usr/bin/env python3
"""
Generate all SLURM experiment scripts for dp-morph OCT Segmentation.

This script generates .txt command files and .sh SLURM scripts for all experiments:
- Baseline ablation (LR × WD grid, no DP, no morph)
- Non-DP with morphology
- DP without morphology (all clipping strategies)
- DP with morphology (all clipping strategies)

Usage:
    python generate_experiments.py           # Generate all experiments
    python generate_experiments.py --dry-run # Show what would be generated
"""

import os
import argparse
from pathlib import Path

# Base paths
SLURM_DIR = Path(__file__).parent
PROJECT_ROOT = "/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya"

# Experiment configurations
MODELS = ["unet", "NestedUNet", "LFUNet"]
MODEL_NAMES = {"unet": "unet", "NestedUNet": "nestedunet", "LFUNet": "lfunet"}

DATASETS = ["Duke", "UMN"]
BATCH_SIZES = [8, 16]
EPSILONS = [8, 200]
RUNS = [1, 2]
MORPH_OPERATIONS = ["open", "close", "both"]
KERNEL_SIZES = [3, 5]
CLIPPING_STRATEGIES = ["flat", "automatic", "psac", "normalized_sgd"]
CLIPPING_NAMES = {
    "flat": "flat",
    "automatic": "automatic",
    "psac": "psac",
    "normalized_sgd": "nsgd",
}

# Ablation study configurations (baseline only)
LEARNING_RATES = [0.001, 0.0005]  # 1e-3, 5e-4
WEIGHT_DECAYS = [1e-4, 1e-9]

# Default fixed hyperparameters
DEFAULT_LR = 0.001
DEFAULT_WD = 1e-4


def generate_sh_script(
    job_name, log_dir, array_size, txt_file, time="06:00:00", partition="a100-80g"
):
    """Generate SLURM shell script content."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={PROJECT_ROOT}/code/dp-morph/OCT_Segmentation/slurm/logs/{log_dir}/{job_name}_%A_%a.out
#SBATCH --error={PROJECT_ROOT}/code/dp-morph/OCT_Segmentation/slurm/logs/{log_dir}/{job_name}_%A_%a.err
#SBATCH --time={time}
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition={partition}
#SBATCH --qos=gpu6hours
#SBATCH --array=1-{array_size}%16

# Create logs directory if it doesn't exist
mkdir -p {PROJECT_ROOT}/code/dp-morph/OCT_Segmentation/slurm/logs/{log_dir}

# Navigate to project directory
cd {PROJECT_ROOT}

# Activate virtual environment
source .venv/bin/activate

# Navigate to OCT_Segmentation code directory
cd code/dp-morph/OCT_Segmentation

# Get the command for this array task
COMMANDS_FILE="{PROJECT_ROOT}/code/dp-morph/OCT_Segmentation/slurm/{txt_file}"
COMMAND=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMMANDS_FILE")

echo "=========================================="
echo "Job Name: {job_name}"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Command: $COMMAND"
echo "=========================================="

# Execute the command
eval $COMMAND

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
"""


def generate_baseline_ablation_commands(model, dataset):
    """
    Generate baseline ablation commands (no DP, no morph).
    Tests all combinations of LR and WD.
    """
    commands = []
    for lr in LEARNING_RATES:
        for wd in WEIGHT_DECAYS:
            for bs in BATCH_SIZES:
                for run in RUNS:
                    cmd = (
                        f"python train-one-gpu.py "
                        f"--model_name {model} "
                        f"--dataset {dataset} "
                        f"--morphology False "
                        f"--DPSGD False "
                        f"--batch_size {bs} "
                        f"--learning_rate {lr} "
                        f"--weight_decay {wd} "
                        f"--run_number {run}"
                    )
                    commands.append(cmd)
    return commands


def generate_base_morph_commands(model, dataset):
    """Generate base + morphology commands (no DP) - uses SMART morphology (layers 3,4,5)."""
    commands = []
    for operation in MORPH_OPERATIONS:
        for kernel_size in KERNEL_SIZES:
            for bs in BATCH_SIZES:
                for run in RUNS:
                    cmd = (
                        f"python train-one-gpu.py "
                        f"--model_name {model} "
                        f"--dataset {dataset} "
                        f"--morphology True "
                        f"--operation {operation} "
                        f"--kernel_size {kernel_size} "
                        f"--smart_morphology True "
                        f"--morph_layers 3,4,5 "
                        f"--DPSGD False "
                        f"--batch_size {bs} "
                        f"--run_number {run}"
                    )
                    commands.append(cmd)
    return commands


def generate_base_morph_all_layers_commands(model, dataset):
    """Generate base + morphology commands (no DP) - morphology applied to ALL layers."""
    commands = []
    for operation in MORPH_OPERATIONS:
        for kernel_size in KERNEL_SIZES:
            for bs in BATCH_SIZES:
                for run in RUNS:
                    cmd = (
                        f"python train-one-gpu.py "
                        f"--model_name {model} "
                        f"--dataset {dataset} "
                        f"--morphology True "
                        f"--operation {operation} "
                        f"--kernel_size {kernel_size} "
                        f"--smart_morphology False "
                        f"--DPSGD False "
                        f"--batch_size {bs} "
                        f"--run_number {run}"
                    )
                    commands.append(cmd)
    return commands


def generate_dp_commands(model, dataset, clipping):
    """Generate DP commands (no morph)."""
    commands = []
    for epsilon in EPSILONS:
        for bs in BATCH_SIZES:
            for run in RUNS:
                cmd = (
                    f"python train-one-gpu.py "
                    f"--model_name {model} "
                    f"--dataset {dataset} "
                    f"--morphology False "
                    f"--DPSGD True "
                    f"--epsilon {epsilon} "
                    f"--clipping {clipping} "
                    f"--batch_size {bs} "
                    f"--run_number {run}"
                )
                commands.append(cmd)
    return commands


def generate_dp_morph_commands(model, dataset, clipping):
    """Generate DP + morphology commands - uses NORMAL morphology (all layers)."""
    commands = []
    for operation in MORPH_OPERATIONS:
        for kernel_size in KERNEL_SIZES:
            for epsilon in EPSILONS:
                for bs in BATCH_SIZES:
                    for run in RUNS:
                        cmd = (
                            f"python train-one-gpu.py "
                            f"--model_name {model} "
                            f"--dataset {dataset} "
                            f"--morphology True "
                            f"--operation {operation} "
                            f"--kernel_size {kernel_size} "
                            f"--smart_morphology False "
                            f"--DPSGD True "
                            f"--epsilon {epsilon} "
                            f"--clipping {clipping} "
                            f"--batch_size {bs} "
                            f"--run_number {run}"
                        )
                        commands.append(cmd)
    return commands


def generate_dp_morph_smart_commands(model, dataset, clipping):
    """Generate DP + morphology commands - uses SMART morphology (layers 3,4,5)."""
    commands = []
    for operation in MORPH_OPERATIONS:
        for kernel_size in KERNEL_SIZES:
            for epsilon in EPSILONS:
                for bs in BATCH_SIZES:
                    for run in RUNS:
                        cmd = (
                            f"python train-one-gpu.py "
                            f"--model_name {model} "
                            f"--dataset {dataset} "
                            f"--morphology True "
                            f"--operation {operation} "
                            f"--kernel_size {kernel_size} "
                            f"--smart_morphology True "
                            f"--morph_layers 3,4,5 "
                            f"--DPSGD True "
                            f"--epsilon {epsilon} "
                            f"--clipping {clipping} "
                            f"--batch_size {bs} "
                            f"--run_number {run}"
                        )
                        commands.append(cmd)
    return commands


def write_files(job_name, log_dir, txt_file, sh_file, commands, dry_run=False):
    """Write command file and SLURM script."""
    if dry_run:
        print(f"  Would create: {txt_file} ({len(commands)} commands)")
        print(f"  Would create: {sh_file}")
        return 2

    # Write txt file
    with open(SLURM_DIR / txt_file, "w") as f:
        f.write("\n".join(commands) + "\n")

    # Write sh file
    with open(SLURM_DIR / sh_file, "w") as f:
        f.write(generate_sh_script(job_name, log_dir, len(commands), txt_file))

    return 2


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM experiment scripts")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without creating files",
    )
    parser.add_argument(
        "--dataset",
        choices=["Duke", "UMN", "all"],
        default="all",
        help="Which dataset(s) to generate",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    total_experiments = 0
    files_created = 0

    print("=" * 60)
    print("dp-morph OCT Segmentation Experiment Generator")
    print("=" * 60)

    for dataset in datasets:
        dataset_lower = dataset.lower()
        print(f"\n{'='*40}")
        print(f"Dataset: {dataset}")
        print(f"{'='*40}")

        for model in MODELS:
            model_short = MODEL_NAMES[model]

            # ========================================
            # 1. Baseline Ablation (LR × WD grid)
            # Note: This includes the "base" case (LR=1e-3, WD=1e-4)
            # ========================================
            job_name = f"{model_short}_{dataset_lower}_ablation"
            log_dir = f"{model_short}-{dataset_lower}-ablation"
            txt_file = f"{model_short}-{dataset_lower}-ablation.txt"
            sh_file = f"{model_short}_{dataset_lower}_ablation.sh"

            commands = generate_baseline_ablation_commands(model, dataset)
            files_created += write_files(
                job_name, log_dir, txt_file, sh_file, commands, args.dry_run
            )
            total_experiments += len(commands)
            print(f"  {model_short} ablation: {len(commands)} experiments")

            # ========================================
            # 2. Base + Morphology (no DP)
            # ========================================
            job_name = f"{model_short}_{dataset_lower}_base_morph"
            log_dir = f"{model_short}-{dataset_lower}-base-morph"
            txt_file = f"{model_short}-{dataset_lower}-base-morph.txt"
            sh_file = f"{model_short}_{dataset_lower}_base_morph.sh"

            commands = generate_base_morph_commands(model, dataset)
            files_created += write_files(
                job_name, log_dir, txt_file, sh_file, commands, args.dry_run
            )
            total_experiments += len(commands)
            print(f"  {model_short} base_morph: {len(commands)} experiments")

            # ========================================
            # 3. Base + Morphology ALL LAYERS (no DP) - new jobs only
            # ========================================
            job_name = f"{model_short}_{dataset_lower}_base_morph_all_layers"
            log_dir = f"{model_short}-{dataset_lower}-base-morph-all-layers"
            txt_file = f"{model_short}-{dataset_lower}-base-morph-all-layers.txt"
            sh_file = f"{model_short}_{dataset_lower}_base_morph_all_layers.sh"

            commands = generate_base_morph_all_layers_commands(model, dataset)
            files_created += write_files(
                job_name, log_dir, txt_file, sh_file, commands, args.dry_run
            )
            total_experiments += len(commands)
            print(f"  {model_short} base_morph_all_layers: {len(commands)} experiments")

            # ========================================
            # 4. DP experiments (all clipping strategies)
            # ========================================
            for clipping in CLIPPING_STRATEGIES:
                clipping_short = CLIPPING_NAMES[clipping]

                # DP without morph
                job_name = f"{model_short}_{dataset_lower}_dp_{clipping_short}"
                log_dir = f"{model_short}-{dataset_lower}-dp-{clipping_short}"
                txt_file = f"{model_short}-{dataset_lower}-dp-{clipping_short}.txt"
                sh_file = f"{model_short}_{dataset_lower}_dp_{clipping_short}.sh"

                commands = generate_dp_commands(model, dataset, clipping)
                files_created += write_files(
                    job_name, log_dir, txt_file, sh_file, commands, args.dry_run
                )
                total_experiments += len(commands)
                print(
                    f"  {model_short} dp_{clipping_short}: {len(commands)} experiments"
                )

                # DP with morph
                job_name = f"{model_short}_{dataset_lower}_dp_{clipping_short}_morph"
                log_dir = f"{model_short}-{dataset_lower}-dp-{clipping_short}-morph"
                txt_file = (
                    f"{model_short}-{dataset_lower}-dp-{clipping_short}-morph.txt"
                )
                sh_file = f"{model_short}_{dataset_lower}_dp_{clipping_short}_morph.sh"

                commands = generate_dp_morph_commands(model, dataset, clipping)
                files_created += write_files(
                    job_name, log_dir, txt_file, sh_file, commands, args.dry_run
                )
                total_experiments += len(commands)
                print(
                    f"  {model_short} dp_{clipping_short}_morph: {len(commands)} experiments"
                )

                # DP with morph (SMART - layers 3,4,5) - new jobs only
                job_name = (
                    f"{model_short}_{dataset_lower}_dp_{clipping_short}_morph_smart"
                )
                log_dir = (
                    f"{model_short}-{dataset_lower}-dp-{clipping_short}-morph-smart"
                )
                txt_file = (
                    f"{model_short}-{dataset_lower}-dp-{clipping_short}-morph-smart.txt"
                )
                sh_file = (
                    f"{model_short}_{dataset_lower}_dp_{clipping_short}_morph_smart.sh"
                )

                commands = generate_dp_morph_smart_commands(model, dataset, clipping)
                files_created += write_files(
                    job_name, log_dir, txt_file, sh_file, commands, args.dry_run
                )
                total_experiments += len(commands)
                print(
                    f"  {model_short} dp_{clipping_short}_morph_smart: {len(commands)} experiments"
                )

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Files {'would be ' if args.dry_run else ''}created: {files_created}")

    # Breakdown by experiment type
    ablation_per_model = (
        len(LEARNING_RATES) * len(WEIGHT_DECAYS) * len(BATCH_SIZES) * len(RUNS)
    )
    morph_per_model = (
        len(MORPH_OPERATIONS) * len(KERNEL_SIZES) * len(BATCH_SIZES) * len(RUNS)
    )
    dp_per_model = len(EPSILONS) * len(BATCH_SIZES) * len(RUNS)
    dp_morph_per_model = (
        len(MORPH_OPERATIONS)
        * len(KERNEL_SIZES)
        * len(EPSILONS)
        * len(BATCH_SIZES)
        * len(RUNS)
    )

    print(f"\nPer model per dataset:")
    print(f"  Ablation (LR×WD):  {ablation_per_model} (includes base case)")
    print(f"  Base + Morph:      {morph_per_model}")
    print(f"  Base + Morph (all layers): {morph_per_model} (new jobs only)")
    print(f"  DP (per clipping): {dp_per_model}")
    print(f"  DP + Morph:        {dp_morph_per_model}")
    print(f"  DP + Morph (smart): {dp_morph_per_model} (new jobs only)")

    if not args.dry_run:
        print(f"\nFiles written to: {SLURM_DIR}")
        print("\nTo submit all experiments for a dataset:")
        print("  for f in *_duke_*.sh; do sbatch $f; done")
        print(
            "\nTo submit ONLY the new morphology experiments (all-layers base, smart DP):"
        )
        print(
            "  for f in *_base_morph_all_layers.sh *_dp_*_morph_smart.sh; do sbatch $f; done"
        )


if __name__ == "__main__":
    main()
