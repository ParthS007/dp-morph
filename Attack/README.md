# Ruuning Attack model

First install the following packages:

```bash
pip install torch torchvision tqdm numpy matplotlib seaborn pandas scikit-learn scipy kornia opacus pillow
```

For running the attacks:

For DUke:

```bash
 python3 main.py --main_dir "DukeData" --n_classes 9 --OUTPUT_CHANNELS 9 --DPSGD True --epsilon 8 --morphology True --operation "open" 
 
```

For UMN:

```bash
python3 main.py --main_dir "UMNData" --n_classes 2 --OUTPUT_CHANNELS 2 --DPSGD True --epsilon 8 --morphology True --operation "open" 
 ```
 For details on defining other parameters such as batch size and more, please refer to our paper.

This code includes multiple attack implementations; however, only the global loss results are reported in the paper. Additional modifications are required for the other attacks.



