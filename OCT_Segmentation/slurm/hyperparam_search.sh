#!/bin/bash
#SBATCH --job-name=hp_search
#SBATCH --output=hp_search_%A_%a.out
#SBATCH --error=hp_search_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-80g
#SBATCH --qos=gpu30min
#SBATCH --array=1-6

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya
source .venv/bin/activate
cd code/dp-morph/OCT_Segmentation

# Define hyperparameter combinations
# Format: lr, weight_decay, batch_size, iterations
case $SLURM_ARRAY_TASK_ID in
    1) LR=1e-3; WD=1e-4; BS=16; ITER=100 ;;   # Baseline with more iterations
    2) LR=5e-4; WD=1e-4; BS=16; ITER=100 ;;   # Lower LR, more iterations
    3) LR=2e-3; WD=1e-5; BS=16; ITER=100 ;;   # Higher LR, lower WD
    4) LR=1e-3; WD=1e-5; BS=8;  ITER=100 ;;   # Smaller batch, lower WD
    5) LR=1e-3; WD=1e-4; BS=32; ITER=100 ;;   # Larger batch
    6) LR=5e-4; WD=1e-5; BS=16; ITER=100 ;;   # Conservative LR, many iterations
esac

echo "=========================================="
echo "Hyperparameter Search - Task $SLURM_ARRAY_TASK_ID"
echo "LR=$LR, WD=$WD, BS=$BS, ITER=$ITER"
echo "Started at: $(date)"
echo "=========================================="

python train-one-gpu.py \
    --morphology False \
    --DPSGD False \
    --model_name NestedUNet \
    --dataset Duke \
    --kernel_size 3 \
    --batch_size $BS \
    --num_iterations $ITER \
    --learning_rate $LR \
    --n_classes 9 \
    --weight_decay $WD \
    --image_size 224 \
    --model_should_be_load False

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "=========================================="
