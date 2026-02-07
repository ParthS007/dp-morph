#!/bin/bash
#SBATCH --job-name=unet_umn_dp_psac_morph_smart
#SBATCH --output=/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/dp-morph/OCT_Segmentation/slurm/logs/unet-umn-dp-psac-morph-smart/unet_umn_dp_psac_morph_smart_%A_%a.out
#SBATCH --error=/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/dp-morph/OCT_Segmentation/slurm/logs/unet-umn-dp-psac-morph-smart/unet_umn_dp_psac_morph_smart_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-80g
#SBATCH --qos=gpu6hours
#SBATCH --array=1-48%16

# Create logs directory if it doesn't exist
mkdir -p /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/dp-morph/OCT_Segmentation/slurm/logs/unet-umn-dp-psac-morph-smart

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya

# Activate virtual environment
source .venv/bin/activate

# Navigate to OCT_Segmentation code directory
cd code/dp-morph/OCT_Segmentation

# Get the command for this array task
COMMANDS_FILE="/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/dp-morph/OCT_Segmentation/slurm/unet-umn-dp-psac-morph-smart.txt"
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMANDS_FILE")

echo "=========================================="
echo "Job Name: unet_umn_dp_psac_morph_smart"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Command: $COMMAND"
echo "=========================================="

# Execute the command
eval $COMMAND

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
