# Results Organization Structure

## Overview
The training pipeline now automatically runs:
1. Training - Model training on training set
2. Validation - Evaluation on validation set (after training)
3. Test - Evaluation on test set (after training)

All results are organized by stage (validation/test) with separate directories and files.

## CSV File Structure

### Per-Experiment CSV (e.g., `unet_batch16_results.csv`)
- Contains multiple rows, one per run
- Columns include: Model_Name, Dataset, DPSGD, Clipping_Strategy, Epsilon, Morphology, Operation, Kernel_Size, Learning_Rate, Batch_Size, Run_Number, Iterations, Training_Loss, Validation_Loss/Test_Loss, Validation_Dice/Test_Dice, MAE, Dice_All, Per_Layer_Dice, Max_Grad_Norm, Noise_Multiplier, Stage

### Global CSV Files
- `all_results_validation_global.csv` - All validation results from all experiments
- `all_results_test_global.csv` - All test results from all experiments

## Key Features

1. Separate Validation and Test Results
   - Validation results saved in `{experiment_dir}/validation/`
   - Test results saved in `{experiment_dir}/test/`

2. Run-Wise Plot Organization
   - Plots saved in `{stage}/plots/run{1,2,3}/`
   - Each run has its own folder with 20 example plots

3. Batch Size Organization
   - Separate CSV files per batch size: `{model}_batch{16,32,48}_results.csv`
   - Multiple runs append to the same CSV file

4. Global Aggregation
   - Two global CSV files for easy aggregation
   - One for validation, one for test

## Pipeline Flow

1. Training Phase
   - Model trains on training set
   - Validation performed every 2 iterations
   - Best model tracked based on validation dice

2. Validation Phase (after training)
   - Final validation evaluation
   - Results saved to `validation/` directory
   - Plots saved to `validation/plots/run{run_number}/`

3. Test Phase (after training)
   - Test evaluation on held-out test set
   - Results saved to `test/` directory
   - Plots saved to `test/plots/run{run_number}/`

## Metrics Recorded

### Training Metrics
- Training Loss (final)

### Validation Metrics
- Validation Loss
- Validation Dice Score
- MAE (Mean Absolute Error)
- Dice All (per-class dice scores)
- Per Layer Dice (dice per anatomical layer)

### Test Metrics
- Test Loss
- Test Dice Score
- MAE (Mean Absolute Error)
- Dice All (per-class dice scores)
- Per Layer Dice (dice per anatomical layer)

### Privacy Metrics (DP only)
- Epsilon (privacy budget consumed)
- Max Gradient Norm
- Noise Multiplier

