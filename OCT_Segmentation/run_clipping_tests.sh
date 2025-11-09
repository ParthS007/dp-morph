#!/bin/zsh
# Generate and submit test jobs for each clipping strategy
# Each with: 1 run, 200 iterations, batch size 16, with and without morphology

cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/dp-morph/OCT_Segmentation

CLIPPING_STRATEGIES=("base" "automatic" "psac" "normalized_sgd")

for clipping in "${CLIPPING_STRATEGIES[@]}"; do
    echo "=========================================="
    echo "Generating jobs for clipping: $clipping"
    echo "=========================================="

    # Generate jobs (not test run, so goes to full_run directory)
    python generate_slurm_jobs.py \
        --clipping "$clipping" \
        --iterations 200 \
        --num-runs 1 \
        --batch-sizes 16 \
        --output-dir "slurm_jobs/clipping_tests_${clipping}"

    # Submit the job
    cd "slurm_jobs/clipping_tests_${clipping}/full_run"
    echo "Submitting job for $clipping..."
    sbatch array_job.sh
    cd ../../..

    echo "Job submitted for $clipping"
    echo ""
done

echo "All clipping strategy test jobs have been generated and submitted!"
echo "Check status with: squeue -u $USER"

