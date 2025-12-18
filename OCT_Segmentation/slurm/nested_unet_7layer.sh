#!/bin/bash
#SBATCH --job-name=nested_unet_7layer
#SBATCH --output=nested_unet_7layer_%j.out
#SBATCH --error=nested_unet_7layer_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-80g
#SBATCH --qos=gpu30min

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya

# Activate virtual environment
source .venv/bin/activate

# Navigate to OCT_Segmentation code directory
cd code/dp-morph/OCT_Segmentation

echo "=========================================="
echo "Job Name: nested_unet_7layer"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Using Shiva's hyperparameters for 7-layer evaluation"
echo "=========================================="

# Execute the training command with Shiva's hyperparameters
python train-one-gpu.py \
    --morphology False \
    --DPSGD False \
    --model_name NestedUNet \
    --dataset Duke \
    --kernel_size 3 \
    --batch_size 16 \
    --num_iterations 100 \
    --learning_rate 1e-3 \
    --n_classes 9 \
    --weight_decay 1e-4 \
    --image_size 224 \
    --model_should_be_load False

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
