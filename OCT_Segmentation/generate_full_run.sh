#!/bin/bash
# Generate full run SLURM jobs for all experiment combinations

cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/dp-morph/OCT_Segmentation

echo "=========================================="
echo "Generating FULL RUN jobs"
echo "=========================================="
echo ""
echo "This will generate jobs for:"
echo "  - 3 models (unet, NestedUNet, LFUNet)"
echo "  - 2 datasets (Duke, UMN)"
echo "  - 3 batch sizes (16, 32, 48)"
echo "  - 2 runs per combination"
echo "  - 3 operations (both, close, open) × 3 kernel sizes (3, 5, 7)"
echo "  - 4 clipping strategies × 2 epsilon values (for DP)"
echo ""
echo "Expected total: ~3,240 experiments"
echo ""

# Activate venv
source /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/.venv/bin/activate

python3 generate_slurm_jobs.py \
    --iterations 200 \
    --output-dir slurm_jobs \
    --max-concurrent 50

echo ""
echo "=========================================="
echo "Full run jobs generated!"
echo "=========================================="
echo ""
echo "To submit, run:"
echo "  cd slurm_jobs/full_run"
echo "  sbatch array_job.sh"
echo ""

