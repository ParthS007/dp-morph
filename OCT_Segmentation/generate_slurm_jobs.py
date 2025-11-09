#!/usr/bin/env python3
"""
Generate SLURM job array scripts for OCT Segmentation experimental matrix.

This script generates SLURM job array scripts according to the experimental strategy:
- 3 models (UNet, NestedUNet, LFUNet)
- 2 datasets (Duke, UMN)
- 2 privacy modes (Non-DP, DP with 4 clipping strategies)
- 4 clipping strategies (base/flat, automatic, psac, normalized_sgd) - DP only
- 2 morphology conditions (with, without)
- 2 epsilon values (8, 200) - DP only

Total: 108 experiments (54 per dataset)

Uses SLURM job arrays for efficient scheduling.
"""

from pathlib import Path
from typing import List, Dict

# Experimental configuration
MODELS = ["unet", "NestedUNet", "LFUNet"]
DATASETS = ["Duke", "UMN"]
EPSILON_VALUES = [8, 200]
CLIPPING_STRATEGIES = {
    "base": "flat",  # base uses 'flat' which doesn't pass clipping arg
    "automatic": "automatic",
    "psac": "psac",
    "normalized_sgd": "normalized_sgd",
}

# Dataset-specific parameters
DATASET_CONFIG = {"Duke": {"n_classes": 9}, "UMN": {"n_classes": 2}}

# SLURM configuration
SLURM_CONFIG = {
    "partition": "a100-80g",
    "qos": "gpu6hours",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 4,
    "gres": "gpu:1",
    "mem": "64G",
    "time": "06:00:00",  # Will be adjusted for test runs
    "max_concurrent": 20,  # Limit concurrent array tasks
}

# Training parameters (defaults, can be overridden)
DEFAULT_TRAINING_PARAMS = {
    "learning_rate": 5e-4,
    "batch_sizes": [16, 32, 48],  # Multiple batch sizes
    "num_runs": 3,  # Number of runs per experiment
    "weight_decay": 1e-9,
    "image_size": 224,
    "operation": "both",
    "kernel_size": 3,
    "delta": 1e-5,
    "max_grad_norm": 1.0,
}


def build_training_command(
    model: str,
    dataset: str,
    dpsgd: bool,
    clipping: str,
    epsilon: float,
    morphology: bool,
    iterations: int,
    batch_size: int,
    run_number: int,
) -> str:
    """Build the Python training command for a single experiment."""
    n_classes = DATASET_CONFIG[dataset]["n_classes"]

    python_args = [
        f"--dataset {dataset}",
        f"--n_classes {n_classes}",
        f"--model_name {model}",
        f"--batch_size {batch_size}",
        f"--num_iterations {iterations}",
        f"--learning_rate {DEFAULT_TRAINING_PARAMS['learning_rate']}",
        f"--weight_decay {DEFAULT_TRAINING_PARAMS['weight_decay']}",
        f"--image_size {DEFAULT_TRAINING_PARAMS['image_size']}",
        f"--morphology {str(morphology)}",
        f"--run_number {run_number}",
    ]

    if morphology:
        python_args.append(f"--operation {DEFAULT_TRAINING_PARAMS['operation']}")
        python_args.append(f"--kernel_size {DEFAULT_TRAINING_PARAMS['kernel_size']}")

    if dpsgd:
        python_args.append("--DPSGD True")
        python_args.append(f"--epsilon {epsilon}")
        python_args.append(f"--clipping {CLIPPING_STRATEGIES[clipping]}")
    else:
        python_args.append("--DPSGD False")
        python_args.append("--clipping flat")  # Not used but set for consistency

    python_cmd = " ".join(python_args)
    return f"python train-one-gpu.py {python_cmd}"


def generate_experiment_name(
    model: str,
    dataset: str,
    dpsgd: bool,
    clipping: str,
    epsilon: float,
    morphology: bool,
    test_run: bool = False,
) -> str:
    """Generate a descriptive name for the experiment."""
    job_suffix = f"{model.lower()}_{dataset.lower()}"
    if dpsgd:
        clipping_name = clipping if clipping != "base" else "base"
        job_suffix += f"_dp_{clipping_name}_eps{int(epsilon)}"
    else:
        job_suffix += "_nodp"

    if morphology:
        job_suffix += "_morph"
    else:
        job_suffix += "_nomorph"

    if test_run:
        job_suffix += "_test"

    return f"oct_{job_suffix}"


def generate_all_experiments(
    test_run: bool = False,
    iterations: int = None,
    model_filter: str = None,
    clipping_filter: str = None,
    num_runs: int = None,
    batch_sizes: List[int] = None,
) -> List[Dict]:
    """
    Generate all experiment configurations.

    Args:
        test_run: Whether this is a test run
        iterations: Number of iterations
        model_filter: If provided, only generate experiments for this model
        clipping_filter: If provided, only generate experiments for this clipping strategy (DP only)

    Returns:
        List of experiment configuration dictionaries
    """
    if iterations is None:
        iterations = 20 if test_run else 200

    # Use provided batch_sizes and num_runs or defaults
    batch_sizes_to_use = (
        batch_sizes if batch_sizes else DEFAULT_TRAINING_PARAMS["batch_sizes"]
    )
    num_runs_to_use = num_runs if num_runs else DEFAULT_TRAINING_PARAMS["num_runs"]

    # Filter models if specified
    models_to_use = (
        [model_filter] if model_filter and model_filter in MODELS else MODELS
    )

    # Filter clipping strategies if specified
    clipping_to_use = (
        [clipping_filter]
        if clipping_filter and clipping_filter in CLIPPING_STRATEGIES.keys()
        else list(CLIPPING_STRATEGIES.keys())
    )

    experiments = []

    # Non-DP experiments (2 per model per dataset) - only if no clipping filter
    if not clipping_filter:
        for dataset in DATASETS:
            for model in models_to_use:
                for morphology in [False, True]:
                    # Add batch sizes and runs
                    for batch_size in batch_sizes_to_use:
                        for run_number in range(1, num_runs_to_use + 1):
                            config = {
                                "model": model,
                                "dataset": dataset,
                                "dpsgd": False,
                                "clipping": "none",
                                "epsilon": 0,
                                "morphology": morphology,
                                "iterations": iterations,
                                "batch_size": batch_size,
                                "run_number": run_number,
                            }
                            experiments.append(config)

    # DP experiments (2 epsilon × 2 morphology per model per dataset per clipping)
    for dataset in DATASETS:
        for model in models_to_use:
            for clipping in clipping_to_use:
                for epsilon in EPSILON_VALUES:
                    for morphology in [False, True]:
                        # Add batch sizes and runs
                        for batch_size in batch_sizes_to_use:
                            for run_number in range(1, num_runs_to_use + 1):
                                config = {
                                    "model": model,
                                    "dataset": dataset,
                                    "dpsgd": True,
                                    "clipping": clipping,
                                    "epsilon": epsilon,
                                    "morphology": morphology,
                                    "iterations": iterations,
                                    "batch_size": batch_size,
                                    "run_number": run_number,
                                }
                                experiments.append(config)

    return experiments


def create_array_job_script(
    num_tasks: int,
    output_dir: Path,
    test_run: bool = False,
    model_filter: str = None,
    clipping_filter: str = None,
) -> str:
    """Create the SLURM array job script."""
    time_limit = "01:00:00" if test_run else SLURM_CONFIG["time"]
    suffix_parts = []
    if model_filter:
        suffix_parts.append(model_filter.lower())
    if clipping_filter:
        suffix_parts.append(clipping_filter.lower())
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    job_name = (
        f"oct_experiments{suffix}_test" if test_run else f"oct_experiments{suffix}"
    )
    max_concurrent = SLURM_CONFIG["max_concurrent"]
    log_dir = output_dir / "logs"

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%A_%a.out
#SBATCH --error={log_dir}/{job_name}_%A_%a.err
#SBATCH --time={time_limit}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus_per_task']}
#SBATCH --gres={SLURM_CONFIG['gres']}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --qos={SLURM_CONFIG['qos']}
#SBATCH --array=1-{num_tasks}%{max_concurrent}

# OCT Segmentation Experimental Matrix
# Array job with {num_tasks} tasks
# Test run: {test_run}
# Max concurrent tasks: {max_concurrent}

# Create logs directory if it doesn't exist
mkdir -p {log_dir}

# Navigate to project root
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to OCT_Segmentation directory
cd code/dp-morph/OCT_Segmentation

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Get the command for this array task
# Use absolute path for commands file
COMMANDS_FILE="{output_dir.resolve()}/commands.cmd"
COMMAND=$(sed -n ${{SLURM_ARRAY_TASK_ID}}p "$COMMANDS_FILE")

echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Command: $COMMAND"
echo "=========================================="

# Execute the command
eval $COMMAND

echo "Task $SLURM_ARRAY_TASK_ID completed!"
"""

    return script_content


def main():
    """Main function to generate SLURM array job scripts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SLURM array job scripts for OCT experiments"
    )
    parser.add_argument(
        "--test", action="store_true", help="Generate test run scripts (20 iterations)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations (default: 20 for test, 200 for full)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="slurm_jobs",
        help="Output directory for SLURM scripts",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent array tasks (default: 20)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=MODELS,
        help="Filter to specific model only (unet, NestedUNet, LFUNet)",
    )
    parser.add_argument(
        "--clipping",
        type=str,
        default=None,
        choices=list(CLIPPING_STRATEGIES.keys()),
        help="Filter to specific clipping strategy only (base, automatic, psac, normalized_sgd) - DP only",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Number of runs per experiment (default: 3, or 20 for test)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to use (default: 16 32 48)",
    )

    args = parser.parse_args()

    test_run = args.test
    iterations = args.iterations if args.iterations else (20 if test_run else 200)
    max_concurrent = (
        args.max_concurrent if args.max_concurrent else SLURM_CONFIG["max_concurrent"]
    )

    # Update max_concurrent in config
    SLURM_CONFIG["max_concurrent"] = max_concurrent

    # Create output directories
    output_dir = Path(args.output_dir)
    if test_run:
        output_dir = output_dir / "test_run"
    else:
        output_dir = output_dir / "full_run"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Generate all experiment configurations
    experiments = generate_all_experiments(
        test_run=test_run,
        iterations=iterations,
        model_filter=args.model,
        clipping_filter=args.clipping,
        num_runs=args.num_runs,
        batch_sizes=args.batch_sizes,
    )

    filter_info = []
    if args.model:
        filter_info.append(f"Model: {args.model}")
    if args.clipping:
        filter_info.append(f"Clipping: {args.clipping}")
    filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

    print(
        f"Generating SLURM array job for {len(experiments)} experiments{filter_str}..."
    )
    print(f"Test run: {test_run}, Iterations: {iterations}")
    print(f"Max concurrent tasks: {max_concurrent}")
    print(f"Output directory: {output_dir}")

    # Generate commands file
    commands_file = output_dir / "commands.cmd"
    experiment_names = []

    with open(commands_file, "w") as f:
        for i, config in enumerate(experiments, 1):
            command = build_training_command(
                model=config["model"],
                dataset=config["dataset"],
                dpsgd=config["dpsgd"],
                clipping=config["clipping"],
                epsilon=config["epsilon"],
                morphology=config["morphology"],
                iterations=config["iterations"],
                batch_size=config["batch_size"],
                run_number=config["run_number"],
            )
            f.write(command + "\n")

            # Generate experiment name for summary
            exp_name = generate_experiment_name(
                model=config["model"],
                dataset=config["dataset"],
                dpsgd=config["dpsgd"],
                clipping=config["clipping"],
                epsilon=config["epsilon"],
                morphology=config["morphology"],
                test_run=test_run,
            )
            experiment_names.append((i, exp_name, config))

            if i % 20 == 0:
                print(f"Generated {i}/{len(experiments)} commands...")

    print(f"Commands file created: {commands_file}")

    # Generate array job script
    array_script = output_dir / "array_job.sh"
    script_content = create_array_job_script(
        num_tasks=len(experiments),
        output_dir=output_dir,
        test_run=test_run,
        model_filter=args.model,
        clipping_filter=args.clipping,
    )
    array_script.write_text(script_content)
    array_script.chmod(0o755)

    print(f"Array job script created: {array_script}")

    # Create summary file
    summary_file = output_dir / "experiments_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"OCT Segmentation Experimental Matrix\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Test Run: {test_run}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Total Experiments: {len(experiments)}\n")
        f.write(f"Max Concurrent Tasks: {max_concurrent}\n\n")
        f.write(
            f"{'Task ID':<8} {'Experiment Name':<60} {'Model':<15} {'Dataset':<8} {'DP':<5} {'Clipping':<15} {'Eps':<6} {'Morph':<6} {'Batch':<6} {'Run':<4}\n"
        )
        f.write(f"{'-'*140}\n")

        for task_id, exp_name, config in experiment_names:
            f.write(
                f"{task_id:<8} {exp_name:<60} {config['model']:<15} {config['dataset']:<8} "
                f"{str(config['dpsgd']):<5} {config['clipping']:<15} "
                f"{str(config['epsilon']):<6} {str(config['morphology']):<6} "
                f"{config['batch_size']:<6} {config['run_number']:<4}\n"
            )

    print(f"Summary written to {summary_file}")

    # Create submission helper script
    submit_script = output_dir / "submit_job.sh"
    submit_content = f"""#!/bin/bash
# Submit SLURM array job

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting SLURM array job with {len(experiments)} tasks..."
echo "Max concurrent tasks: {max_concurrent}"
echo ""

sbatch array_job.sh

echo ""
echo "Job submitted!"
echo "Check status with: squeue -u $USER"
echo "Monitor progress: squeue -j <JOB_ID>"
echo "View logs: ls -lh {output_dir}/logs/"
"""

    submit_script.write_text(submit_content)
    submit_script.chmod(0o755)

    print(f"Submission script created: {submit_script}")
    print(f"\nTo submit the array job, run:")
    print(f"  cd {output_dir}")
    print(f"  bash submit_job.sh")
    print(f"\nOr directly:")
    print(f"  cd {output_dir}")
    print(f"  sbatch array_job.sh")


if __name__ == "__main__":
    main()
