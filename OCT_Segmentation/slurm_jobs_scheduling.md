# SLURM Job Array Scheduling for OCT Segmentation Experiments

This directory contains scripts to schedule all 108 experiments via SLURM job arrays on the HPC cluster. [Job arrays](https://docs.scicore.unibas.ch/HPC%20Cluster/batchcomputing/#array-jobs) are the recommended approach as they are much more efficient and put less stress on the Workload Manager than launching hundreds of individual scripts.

## Overview

The experimental matrix includes:
- 3 Models: UNet, NestedUNet, LFUNet
- 2 Datasets: Duke (9 classes), UMN (2 classes)
- 2 Privacy Modes: Non-DP (6 experiments per dataset), DP-SGD (48 experiments per dataset)
- 4 Clipping Strategies (DP only): base/flat, automatic, psac, normalized_sgd
- 2 Morphology Conditions: with, without
- 2 Epsilon Values (DP only): 8, 200

Total: 108 experiments (54 per dataset)

## How Job Arrays Work

Instead of creating 108 individual SLURM scripts, we use a single array job with 108 tasks:
- One `commands.cmd` file containing all 108 training commands (one per line)
- One `array_job.sh` script that uses `--array=1-108%20` to run tasks 1-108 with max 20 concurrent
- Each task reads its command from `commands.cmd` using `SLURM_ARRAY_TASK_ID`

## Usage

### Step 1: Generate Test Run Scripts

First, generate test run scripts with fewer iterations (20) to verify everything works:

```bash
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/dp-morph/OCT_Segmentation

python generate_slurm_jobs.py --test
```

This will create:
- `slurm_jobs/test_run/` directory
- `commands.cmd` - file with all 108 training commands
- `array_job.sh` - single SLURM array job script
- `submit_job.sh` - helper script to submit the array job
- `experiments_summary.txt` - summary of all experiments

### Step 2: Submit Test Run Array Job

```bash
cd slurm_jobs/test_run
bash submit_job.sh
```

Or directly:
```bash
sbatch array_job.sh
```

This submits one array job with 108 tasks. SLURM will automatically:
- Run up to 20 tasks concurrently (configurable)
- Queue remaining tasks
- Assign each task a unique `SLURM_ARRAY_TASK_ID` (1-108)

### Step 3: Monitor Test Run

Check array job status:
```bash
# Check all your jobs
squeue -u $USER

# Check specific array job (replace JOB_ID with actual ID)
squeue -j JOB_ID

# View array job details
scontrol show job JOB_ID
```

View logs:
```bash
# List all log files
ls -lh slurm_jobs/test_run/logs/

# View specific task log (e.g., task 1)
cat slurm_jobs/test_run/logs/oct_experiments_test_JOBID_1.out

# Monitor all tasks
tail -f slurm_jobs/test_run/logs/oct_experiments_test_*.out
```

### Step 4: Generate Full Run Scripts

After verifying the test run works, generate full run scripts with 200 iterations:

```bash
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/dp-morph/OCT_Segmentation

python generate_slurm_jobs.py
```

This creates `slurm_jobs/full_run/` with all scripts configured for 200 iterations.

### Step 5: Submit Full Run Array Job

```bash
cd slurm_jobs/full_run
bash submit_job.sh
```

## Job Script Structure

The generated array job script:
- Uses partition `a100-80g` with QoS `gpu6hours`
- Requests 1 GPU, 4 CPUs, 64GB RAM per task
- Time limit: 1 hour (test) or 6 hours (full)
- Array: `--array=1-108%20` (108 tasks, max 20 concurrent)
- Activates the conda environment automatically
- Reads command from `commands.cmd` using `SLURM_ARRAY_TASK_ID`
- Saves logs to `slurm_jobs/*/logs/` with format `oct_experiments_JOBID_TASKID.out`

## Results Organization

Results are saved according to the experimental strategy structure:
```
results/
├── Duke/
│   ├── non_dp/
│   │   ├── no_morph/
│   │   └── with_morph/
│   └── dp/
│       ├── base/
│       ├── automatic/
│       ├── psac/
│       └── normalized_sgd/
└── UMN/
    └── (same structure)
```
