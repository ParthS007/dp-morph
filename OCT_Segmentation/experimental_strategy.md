# OCT Segmentation Experimental Strategy

## Overview
Comprehensive experimental matrix for OCT segmentation with differential privacy and morphological operations.

## Experimental Variables

### 1. Models
- UNet (baseline)
- NestedUNet (UNet++)
- LFUNet

### 2. Datasets
- Duke (9 classes)
- UMN (2 classes)

### 3. DP Clipping Strategies (Opacus 1.5.4)
- Standard - NO clipping argument passed
- automatic: Automatic clipping - `clipping="automatic"`
- psac: Per-Sample Adaptive Clipping - `clipping="psac"`
- normalized_sgd: Normalized SGD clipping - `clipping="normalized_sgd"`

### 4. Morphology
- Without Morphology: Baseline segmentation
- With Morphology: Post-processing with morphological operations
  - Operations: `both`
  - Kernel size: 3 (default)

### 5. Privacy Settings
- Non-DP: Baseline without differential privacy
- DP-SGD: With differential privacy
  - Epsilon (ε): 8, 200 (privacy budget: tight, moderate, relaxed)
  - Delta (δ): 1e-5
  - Max gradient norm: 1.0

## Experimental Matrix

### Total Experiments per Dataset
```
Models: 3 (UNet, NestedUNet, LFUNet)
DP Modes: 2 (Non-DP, DP)
Clipping (DP only): 4 (flat/default, automatic, psac, normalized_sgd)
Morphology: 2 (with, without)
Epsilon (DP only): 2 (8, 200)

Non-DP: 3 models × 2 morphology = 6 experiments
DP: 3 models × 4 clipping × 2 morphology × 2 epsilon = 48 experiments

Total per dataset: 54 experiments
Total (both datasets): 108 experiments
```

## Results Organization

### Directory Structure
```
results/
├── Duke/
│   ├── non_dp/
│   │   ├── no_morph/
│   │   │   ├── unet_results.csv
│   │   │   ├── nestedunet_results.csv
│   │   │   └── lfunet_results.csv
│   │   └── with_morph/
│   │       ├── unet_close_results.csv
│   │       ├── nestedunet_close_results.csv
│   │       └── lfunet_close_results.csv
│   └── dp/
│       ├── base/              # Default clipping (no arg)
│       │   ├── epsilon_8/
│       │   └── epsilon_200/
│       ├── automatic/         # clipping="automatic"
│       ├── psac/              # clipping="psac"
│       └── normalized_sgd/    # clipping="normalized_sgd"
└── UMN/
    └── (same structure)
```

### CSV Results Format
Columns:
- Model_Name
- Dataset
- DPSGD (True/False)
- Clipping_Strategy (base/automatic/psac/normalized_sgd/none)
- Epsilon (8, 200, or 0 for non-DP)
- Morphology (True/False)
- Operation (both/none)
- Kernel_Size
- Learning_Rate
- Batch_Size
- Iterations
- Training_Loss (final)
- Validation_Loss (final)
- Validation_Dice (final)
- MAE (final)
- Dice_All (array)
- Per_Layer_Dice (list)
- Max_Grad_Norm
- Noise_Multiplier
- Privacy_Epsilon_Consumed

## Order

###  Baseline Establishment (Non-DP)
1. All 3 models without morphology on both datasets
2. All 3 models with morphology (both) on both datasets

### DP with Standard Clipping (base - no clipping arg)
1. All 3 models, epsilon=8, 200, no morphology
2. All 3 models, epsilon=8, 200, with morphology (both)

### Advanced Clipping - Automatic
1. All 3 models, epsilon=8, 200, no morphology
2. All 3 models, epsilon=8, 200, with morphology (both)

### Advanced Clipping - PSAC
1. All 3 models, epsilon=8, 200, no morphology
2. All 3 models, epsilon=8, 200, with morphology (both)

### Advanced Clipping - Normalized SGD
1. All 3 models, epsilon=8, 200, no morphology
2. All 3 models, epsilon=8, 200, with morphology (both)


## Training Parameters

### Fixed Parameters
- Learning rate: 5e-4
- Batch size: 16
- Iterations: 200
- Weight decay: 1e-9
- Image size: 224
- Optimizer: Adam
- Loss: Combined Loss (BCE + Dice)

### DP Parameters (when enabled)
- Delta: 1e-5
- Max gradient norm: 1.0
- Epsilon: 8, 200

## Metrics to Track

### Primary Metrics
- Dice Coefficient: Segmentation accuracy
- MAE: Mean Absolute Error
- Validation Loss: Convergence quality

### Privacy Metrics (DP only)
- Epsilon consumed: Final privacy budget
- RDP certificates: Privacy accounting details

### Computational Metrics
- Training time per epoch
- Memory usage
- GPU utilization
