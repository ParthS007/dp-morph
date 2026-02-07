#!/usr/bin/env python3
"""
Generate SLURM experiment scripts for epsilon sweep experiments.

This script generates experiments to study how Dice score varies with epsilon
(privacy budget) from 8 to 200.

Configuration:
- Clipping Strategy: automatic (AUTO-S)
- Batch Size: 8
- Runs: 1 only
- Morphology: model-dependent (U-Net/LFU-Net: smart morph both k=3; NestedUNet: normal morph close k=3)
- Epsilons: 8, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200
- Results Base: results_epsilon_sweep

Usage:
    python generate_epsilon_sweep.py           # Generate all experiments
    python generate_epsilon_sweep.py --dry-run # Show what would be generated
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

# Fixed parameters for epsilon sweep
BATCH_SIZE = 8
RUN_NUMBER = 1
CLIPPING_STRATEGY = "automatic"
RESULTS_BASE = "results_epsilon_sweep"

# Epsilon values to test (8 to 200)
EPSILONS = [8, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]


def generate_sh_script(job_name, log_dir, array_size, txt_file, time="06:00:00"):
    """Generate SLURM shell script content."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={PROJECT_ROOT}/code/dp-morph/OCT_Segmentation/slurm/epsilon_sweep/logs/{log_dir}/{job_name}_%A_%a.out
#SBATCH --error={PROJECT_ROOT}/code/dp-morph/OCT_Segmentation/slurm/epsilon_sweep/logs/{log_dir}/{job_name}_%A_%a.err
#SBATCH --time={time}
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=gpu6hours
#SBATCH --array=1-{array_size}%16

# Create logs directory if it doesn't exist
mkdir -p {PROJECT_ROOT}/code/dp-morph/OCT_Segmentation/slurm/epsilon_sweep/logs/{log_dir}

# Navigate to project directory
cd {PROJECT_ROOT}

# Activate virtual environment
source .venv/bin/activate

# Navigate to OCT_Segmentation code directory
cd code/dp-morph/OCT_Segmentation

# Get the command for this array task
COMMANDS_FILE="{PROJECT_ROOT}/code/dp-morph/OCT_Segmentation/slurm/epsilon_sweep/{txt_file}"
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


def get_morphology_args(model):
    """Return morphology flags for the given model."""
    if model == "NestedUNet":
        return (
            "--morphology True "
            "--operation close "
            "--kernel_size 3 "
            "--smart_morphology False "
        )
    # U-Net and LFUNet: smart morphology, both operation, k=3
    return (
        "--morphology True "
        "--operation both "
        "--kernel_size 3 "
        "--smart_morphology True "
        "--morph_layers 3,4,5 "
    )


def generate_epsilon_sweep_commands(model, dataset):
    """Generate epsilon sweep commands for a given model and dataset (with morphology)."""
    morph_args = get_morphology_args(model)
    commands = []
    for epsilon in EPSILONS:
        cmd = (
            f"python train-one-gpu.py "
            f"--model_name {model} "
            f"--dataset {dataset} "
            f"{morph_args}"
            f"--DPSGD True "
            f"--epsilon {epsilon} "
            f"--clipping {CLIPPING_STRATEGY} "
            f"--batch_size {BATCH_SIZE} "
            f"--run_number {RUN_NUMBER} "
            f"--results_base {RESULTS_BASE}"
        )
        commands.append(cmd)
    return commands


def generate_epsilon_sweep_commands_no_morph(model, dataset):
    """Generate epsilon sweep commands for DP without morphology (same epsilons, batch size 8)."""
    commands = []
    for epsilon in EPSILONS:
        cmd = (
            f"python train-one-gpu.py "
            f"--model_name {model} "
            f"--dataset {dataset} "
            f"--morphology False "
            f"--DPSGD True "
            f"--epsilon {epsilon} "
            f"--clipping {CLIPPING_STRATEGY} "
            f"--batch_size {BATCH_SIZE} "
            f"--run_number {RUN_NUMBER} "
            f"--results_base {RESULTS_BASE}"
        )
        commands.append(cmd)
    return commands


def write_files(job_name, log_dir, txt_file, sh_file, commands, dry_run=False):
    """Write command file and SLURM script."""
    if dry_run:
        print(f"  Would create: {txt_file} ({len(commands)} commands)")
        print(f"  Would create: {sh_file}")
        for i, cmd in enumerate(commands[:3], 1):
            print(f"    Command {i}: {cmd}")
        if len(commands) > 3:
            print(f"    ... and {len(commands) - 3} more commands")
        return 2

    # Write txt file
    txt_path = SLURM_DIR / txt_file
    with open(txt_path, "w") as f:
        f.write("\n".join(commands) + "\n")
    print(f"  Created: {txt_file} ({len(commands)} commands)")

    # Write sh file
    sh_path = SLURM_DIR / sh_file
    with open(sh_path, "w") as f:
        f.write(generate_sh_script(job_name, log_dir, len(commands), txt_file))
    print(f"  Created: {sh_file}")

    return 2


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for epsilon sweep experiments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without creating files",
    )
    args = parser.parse_args()

    total_experiments = 0
    files_created = 0

    print("=" * 60)
    print("Epsilon Sweep Experiment Generator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Clipping Strategy: {CLIPPING_STRATEGY}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Run Number: {RUN_NUMBER}")
    print(f"  Results Base: {RESULTS_BASE}")
    print(f"  Epsilons: {EPSILONS}")
    print(f"  Models: {list(MODEL_NAMES.values())}")
    print(f"  Datasets: {DATASETS}")

    # --- DP with morphology (existing) ---
    print("\n--- DP with morphology ---")
    for model in MODELS:
        model_short = MODEL_NAMES[model]

        for dataset in DATASETS:
            dataset_lower = dataset.lower()

            job_name = f"{model_short}_{dataset_lower}_eps_sweep"
            log_dir = f"{model_short}-{dataset_lower}-eps-sweep"
            txt_file = f"{model_short}-{dataset_lower}-eps-sweep.txt"
            sh_file = f"{model_short}_{dataset_lower}_eps_sweep.sh"

            print(f"\n{model_short} - {dataset} (morph):")

            commands = generate_epsilon_sweep_commands(model, dataset)
            files_created += write_files(
                job_name, log_dir, txt_file, sh_file, commands, args.dry_run
            )
            total_experiments += len(commands)

    # --- DP without morphology (same epsilons, batch size 8) ---
    print("\n--- DP without morphology ---")
    for model in MODELS:
        model_short = MODEL_NAMES[model]

        for dataset in DATASETS:
            dataset_lower = dataset.lower()

            job_name = f"{model_short}_{dataset_lower}_eps_sweep_no_morph"
            log_dir = f"{model_short}-{dataset_lower}-eps-sweep-no-morph"
            txt_file = f"{model_short}-{dataset_lower}-eps-sweep-no-morph.txt"
            sh_file = f"{model_short}_{dataset_lower}_eps_sweep_no_morph.sh"

            print(f"\n{model_short} - {dataset} (no morph):")

            commands = generate_epsilon_sweep_commands_no_morph(model, dataset)
            files_created += write_files(
                job_name, log_dir, txt_file, sh_file, commands, args.dry_run
            )
            total_experiments += len(commands)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Files {'would be ' if args.dry_run else ''}created: {files_created}")
    print(
        f"\nExperiments per model per dataset: {len(EPSILONS)} (morph + no_morph each)"
    )

    if not args.dry_run:
        print(f"\nFiles written to: {SLURM_DIR}")
        print("\nTo submit DP with morphology:")
        print("  for f in *_eps_sweep.sh; do [[ $f != *no_morph* ]] && sbatch $f; done")
        print("\nTo submit DP without morphology only:")
        print("  for f in *_eps_sweep_no_morph.sh; do sbatch $f; done")
        print("\nOr submit no_morph individually:")
        for model in MODELS:
            model_short = MODEL_NAMES[model]
            for dataset in DATASETS:
                dataset_lower = dataset.lower()
                print(f"  sbatch {model_short}_{dataset_lower}_eps_sweep_no_morph.sh")


if __name__ == "__main__":
    main()
