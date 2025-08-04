# Hierarchical Perception and Adaptive Enhancement for Multimodal Sentiment Analysis

This repository contains the official PyTorch implementation for the paper "Hierarchical Perception and Adaptive Enhancement for Multimodal Sentiment Analysis" (HPA-MSA).

HPA-MSA is a novel, cognitive-inspired framework for multimodal sentiment analysis that addresses two key limitations in existing models: the lack of hierarchical perception within each modality and the absence of an adaptive mechanism for fusing conflicting or complementary cues.

## Core Features

- **Hierarchical Perception**: A **Unimodal Explicit-Implicit Cue Decoupling (UEICD)** module disentangles dominant (explicit) and subtle (implicit) cues within each modality (text, audio, vision).
- **Adaptive Enhancement**: A **Cross-modal Adaptive Complementary Enhancement (CACE)** module performs a cascaded, perception-driven fusion. It first forms a preliminary judgment based on explicit cues and then uses it to arbitrate and refine the final decision using implicit cues.
- **State-of-the-Art Performance**: Achieves competitive results on popular benchmark datasets like CMU-MOSI, CMU-MOSEI, and CH-SIMS.

## Project Structure

```
HPA-MSA-reproduction/
│
├── configs/                 # YAML configuration files for experiments
│   ├── cmu_mosi_default.yaml
│   └── ...
├── data/                    # Placeholder for datasets
│   └── CMU-MOSI/processed/
├── src/                     # Source code
│   ├── models/              # Model definitions (HPA-MSA, UEICD, CACE)
│   ├── utils/               # Utility functions (loss, metrics)
│   ├── config.py            # Configuration loading system
│   └── datasets.py          # PyTorch Dataset and DataLoader logic
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs
├── train.py                 # Main training and evaluation script
└── README.md                # This file
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repository-link>
cd HPA-MSA-reproduction
```

### 2. Create a Conda Environment (Recommended)

It is highly recommended to use a virtual environment to manage dependencies.

```bash
conda create -n hpamsa python=3.8 -y
conda activate hpamsa
```

### 3. Install Dependencies

Install all required Python packages using pip.

```bash
pip install -r requirements.txt
```

**`requirements.txt` should contain:**
```
torch
numpy
PyYAML
easydict
tqdm
scikit-learn
scipy
matplotlib
```

### 4. Prepare the Dataset

1.  Download the pre-processed dataset file (e.g., `aligned_50.pkl` for CMU-MOSI).
2.  Place the `.pkl` file into the appropriate directory, for example: `data/CMU-MOSI/processed/`.
3.  Ensure the `data_path` in your configuration file (e.g., `configs/cmu_mosi_default.yaml`) points to the correct location of this file.

## Training and Evaluation

The entire workflow is driven by a main training script `train.py` and YAML configuration files located in the `configs/` directory.

### Running an Experiment

To run an experiment, you simply need to specify the path to the desired configuration file using the `--config` argument.

#### 1. Baseline Training

To train the full HPA-MSA model on the CMU-MOSI dataset using the default settings, run:

```bash
python train.py --config configs/cmu_mosi_default.yaml
```

It is recommended to redirect the output to a log file for better tracking and analysis:

```bash
# Create a logs directory first
mkdir -p logs

# Run training and save logs
python train.py --config configs/cmu_mosi_default.yaml > logs/hpa_msa_baseline.log 2>&1 &
```

The script will:
- Load the configuration and dataset.
- Initialize the HPA-MSA model, loss function, and optimizer.
- Start the training loop for the number of epochs specified in the config.
- After each epoch, it will evaluate the model on the validation set and print the performance.
- The model with the best performance on the validation set (based on F1-score) will be saved to the `checkpoints/` directory.
- After training is complete, it will load the best model and perform a final evaluation on the test set.

#### 2. Running Ablation Studies

To run an ablation study, simply use a different configuration file that modifies the parameters of the baseline. For example, to train a model without the orthogonality loss (`L_ortho`):

```bash
python train.py --config configs/hpa_msa_ablation_no_ortho.yaml
```

### Configuration Files (`.yaml`)

All hyperparameters and settings for an experiment are controlled via YAML files in the `configs/` directory. This allows for easy and reproducible experimentation without changing the source code.


## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{anonymous2024hpamsa,
  title={Hierarchical Perception and Adaptive Enhancement for Multimodal Sentiment Analysis},
  author={Anonymous},
  booktitle={Submission},
  year={2024}
}
```