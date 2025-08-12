# OCT_Segmentation

This directory includes all codes that are necessary for generating morph and non-morph results for both DPSGD and non-DPSGD cases.
First, the following packages:
```
pip install torch tqdm numpy matplotlib opacus kornia torchinfo

``` 
If you plan to use fastDP library, install:
```
pip install fast-dp

```

For running the code,you can:

```
python train-one-gpu.py --dataset UMN --n_classes 2 --batch_size 16 --num_iterations 200 --learning_rate 0.0005 --model_name NestedUNet --device cuda

```
For running the fastDP,just replace train_one_gpu.py with train-one-gpu_fast.py.

The arguments are:

| Argument               | Type    | Default          | Description                                                                                      |
|------------------------|---------|------------------|--------------------------------------------------------------------------------------------------|
| `--dataset`            | string  | `Duke`            | Dataset to use. Options: `Duke`, `UMN`                                                          |
| `--batch_size`         | int     | 16               | Number of samples per batch                                                                      |
| `--num_iterations`     | int     | 200              | Number of training iterations                                                                    |
| `--learning_rate`      | float   | 0.0005           | Learning rate for the optimizer                                                                  |
| `--n_classes`          | int     | 9                | Number of output classes                                                                          |
| `--ffc_lambda`         | float   | 0                | Lambda parameter for FFC (if used)                                                               |
| `--weight_decay`       | float   | 1e-9             | Weight decay (L2 regularization)                                                                 |
| `--image_size`         | int     | 224              | Size of the input images                                                                          |
| `--model_name`         | string  | `NestedUNet`     | Model architecture to use. Options: `unet`, `y_net_gen`, `y_net_gen_ffc`, `UNetOrg`, `LFUNet`, `FCN8s`, `NestedUNet`, `SimplifiedFCN8s`, `ConvNet` |
| `--g_ratio`            | float   | 0.5              | Custom parameter (add your specific description)                                                 |
| `--device`             | string  | `cuda`           | Device to run the model on. Options: `cuda`, `cpu`                                               |
| `--in_channels`        | int     | 1                | Number of input channels (e.g., 1 for grayscale images)                                          |
| `--image_dir`          | string  | None             | Path to the directory containing input images                                                   |
| `--DPSGD`              | bool    | False            | Enable Differentially Private SGD                                                               |
| `--test`               | bool    | False            | Run in test mode                                                                                  |
| `--model_should_be_saved` | bool  | False            | Save the trained model                                                                            |
| `--model_should_be_load`  | bool  | False            | Load a saved model                                                                               |
| `--save_dir`           | string  | `./saved_models/`| Directory to save or load models                                                                 |
| `--epsilon`            | float   | 8                | Privacy epsilon value for DPSGD                                                                  |
| `--morphology`         | bool    | True             | Enable morphology operations                                                                     |
| `--operation`          | string  | `close`          | Morphology operation type. Options: `close`, `open`, `both`                                     |
| `--kernel_size`        | int     | 3                | Kernel size for morphology operations                                                           |


Please note: If you set the dataset to Duke, the n_classes parameter should be 9; for UMN, it should be 2.
The remaining parameters for each dataset and model are detailed in the accompanying paper.


