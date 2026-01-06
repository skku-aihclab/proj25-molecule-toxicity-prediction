# MoltiTox: a multimodal fusion model for molecular toxicity prediction

[View Paper](https://doi.org/10.3389/ftox.2025.1720651)

A comprehensive deep learning model for molecular toxicity prediction using a multimodal approach. The model combines molecular graphs, SMILES sequences, 2D structure images, and NMR spectra to predict toxicity across 12 different endpoints from the Tox21 dataset.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Download](#data-download)
- [Data Preprocessing](#data-preprocessing)
- [Running Experiments](#running-experiments)
- [Results](#results)

## Overview

This model leverages pre-trained molecular encoders and multimodal fusion for toxicity prediction, achieving superior performance compared to single-modal baselines on the Tox21 benchmark.

![MoltiTox Architecture](figure/MoltiTox_figure.jpg)

It supports:
- **4 Single-Modal Encoders**: Graph (GNN), SMILES (Transformer), Image (CNN), Spectrum (1D CNN)
- **8 Multimodal Fusion Models**: All pairwise and higher-order combinations of the modalities
- **12 Toxicity Endpoints**: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53

The multimodal models use self-attention fusion mechanisms to effectively combine complementary information from different molecular representations.

## Features

- **Automated Hyperparameter Tuning**: Optuna-based TPE optimization
- **Two-Stage Training**: Train/valid split optimization → Full dataset retraining
- **Pre-trained Backbones**:
  - MoLFormer-XL for SMILES encoding
  - ImageMol (ResNet18) for image encoding
  - CReSS for NMR spectrum encoding
- **Multi-Task Learning**: Simultaneous prediction of 12 toxicity endpoints

## Project Structure

```
[root]/
├── models/
│   └── models.py              # All encoder and classifier models
├── utils/
│   ├── dataset.py             # Dataset classes for all modalities
│   └── attention_analysis.py  # Attention analysis utilities
├── experiments/
│   ├── graph/                 # Graph Encoder
│   ├── smiles/                # SMILES Encoder
│   ├── image/                 # Image Encoder
│   ├── spectrum/              # Spectrum Encoder
│   └── multimodal/            # Multimodal fusion experiments
│       ├── gph_smi/           # Graph + SMILES
│       ├── gph_img/           # Graph + Image
│       ├── gph_spec/          # Graph + Spectrum
│       ├── smi_img/           # SMILES + Image
│       ├── smi_spec/          # SMILES + Spectrum
│       ├── spec_img/          # Image + Spectrum
│       ├── gph_smi_img/       # Graph + SMILES + Image
│       └── moltitox/          # Graph + SMILES + Image + Spectrum
├── data/
│   ├── train.csv              # Training data
│   ├── valid.csv              # Validation data
│   ├── test.csv               # Test data
│   ├── train_spectra.csv      # Training data with spectra only
│   ├── valid_spectra.csv      # Validation data with spectra only
│   ├── test_spectra.csv       # Test data with spectra only
│   ├── images/                # Molecular 2D images
│   └── spectra/               # NMR spectral data
├── checkpoints/
│   ├── encoder/               # Saved encoder weights
│   │   ├── train_only/        # Encoders trained on train set only
│   │   └── train_and_valid/   # Encoders trained on train+valid
│   ├── model/                 # Saved full model weights
│   ├── parameters/            # Best hyperparameters (JSON)
│   └── pretrained_models/     # Pre-trained model checkpoints
├── main.py                    # Run all experiments sequentially
├── requirements.txt           # Python dependencies
└── README.md                  
```


## Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- Transformers (Hugging Face)
- RDKit
- scikit-learn
- optuna
- pandas
- numpy
- PIL
- matplotlib
- seaborn

### Setup

```bash
# Clone the repository
git clone https://github.com/skku-aihclab/proj25-molecule-toxicity-prediction.git
cd proj25-molecule-toxicity-prediction

# Install dependencies
pip install -r requirements.txt
```

See the [Data Download](#data-download) section below for downloading datasets and pre-trained models.

## Data Download

All required data files, pre-trained models, and checkpoints are available on Google Drive:

**Google Drive Link:** https://drive.google.com/drive/folders/13QLMfp9T_C8tiHabWEwB1knieR9ZOZO9?usp=drive_link

### Directory Structure on Google Drive

```
MoltiTox/
├── data/
│   ├── 1st/
│   │   ├── train.csv
│   │   ├── valid.csv
│   │   ├── test.csv
│   │   ├── train_spectra.csv
│   │   ├── valid_spectra.csv
│   │   └── test_spectra.csv
│   ├── 2nd/
│   │   └── [same structure]
│   ├── 3rd/
│   │   └── [same structure]
│   ├── 4th/
│   │   └── [same structure]
│   └── 5th/
│       └── [same structure]
├── checkpoints/
│   ├── 1st/
│   │   ├── encoder/
│   │   ├── model/
│   │   └── parameters/
│   ├── 2nd/
│   │   └── [same structure]
│   ├── 3rd/
│   │   └── [same structure]
│   ├── 4th/
│   │   └── [same structure]
│   └── 5th/
│       └── [same structure]
├── ImageMol.pth
└── 8.pth
```

### Download Instructions

1. **Download Dataset**
   Images and spectra are included in this repository.
   
   However, if you want to use the same data splits as reported in the paper, download the CSV files from Google Drive:
   - Navigate to the `data/` folder on Google Drive
   - Choose one of the 5 splits (1st through 5th)
   - Download all 6 CSV files:
     - `train.csv`, `valid.csv`, `test.csv` (full datasets)
     - `train_spectra.csv`, `valid_spectra.csv`, `test_spectra.csv` (spectra subset only)
   - Place the downloaded CSV files in your local `data/` directory

2. **Download Pre-trained Checkpoints (Optional)**
   - Navigate to the `checkpoints/` folder on Google Drive
   - Choose the corresponding split (1st through 5th) that matches your data
   - Download the entire folder structure (`encoder/`, `model/`, `parameters/`)
   - Place the downloaded folders in your local `checkpoints/` directory
   - The checkpoint structure must match the original:
     ```
     checkpoints/
     ├── encoder/
     │   ├── train_only/
     │   └── train_and_valid/
     ├── model/
     └── parameters/
     ```

3. **Download Pre-trained Backbones**
   - Download `ImageMol.pth` from the Google Drive root
   - Place it in `experiments/image/ImageMol.pth`
   - Download `8.pth` from the Google Drive root
   - Place it in `experiments/spectrum/8.pth`

### Data File Descriptions

- **Standard CSV files** (`train.csv`, `valid.csv`, `test.csv`):
  - Contains all molecules with graph, SMILES, and image modalities
  - Used for single-modal (graph, SMILES, image) and multimodal experiments without spectra

- **Spectra CSV files** (`train_spectra.csv`, `valid_spectra.csv`, `test_spectra.csv`):
  - Contains only molecules with available NMR spectra
  - Subset of the standard datasets
  - Used for spectrum-based and spectrum-inclusive multimodal experiments

## Data Preprocessing

The preprocessing pipeline for the Tox21 dataset is documented in `data/preprocess.ipynb`.

**Note:** The preprocessing notebook requires access to the original spectral databases (NMRShiftDB2, NP-MRD, HMDB). These databases are not included in this repository due to licensing and size constraints.

If you wish to reproduce the preprocessing pipeline or need access to the raw spectral databases, please contact me.

The pre-processed data (CSV files, images, and binary spectra) are already available and can be used directly without running the preprocessing pipeline.

## Running Experiments

### Running All Experiments

To run all experiments sequentially (single-modal + multimodal):

```bash
python -u main.py 2>&1 | tee result.txt
```

This will:
1. Train and test all 4 single-modal models
2. Train and test all 8 multimodal models
3. Save all results to `result.txt`
4. Save model checkpoints to `checkpoints/`
5. Save best hyperparameters to `checkpoints/parameters/`

### Running Individual Experiments

To run a specific experiment:

```bash
# Single-modal examples
cd experiments/graph
python train.py  # Train graph model
python test.py   # Test graph model

cd ../smiles
python train.py  # Train SMILES model
python test.py   # Test SMILES model

# Multimodal examples
cd ../multimodal/gph_smi
python train.py  # Train Graph+SMILES fusion model
python test.py   # Test Graph+SMILES fusion model

cd ../smi_img
python train.py  # Train SMILES+Image fusion model
python test.py   # Test SMILES+Image fusion model
```

### Training Pipeline

Each training script follows this pipeline:

1. **Load Data**
2. **Load Encoders**
3. **Hyperparameter Search**
4. **Save Best Parameters**
5. **Retrain on Full Data**

### Testing Pipeline

Each testing script:
1. Loads test data
2. Loads best hyperparameters
3. Loads trained model checkpoint
4. Evaluates on test set
5. Reports per-task AUC and mean AUC

## Citation

If you use this model in your research, please cite:

```
@article{park2025moltitox,
  title={MoltiTox: a multimodal fusion model for molecular toxicity prediction},
  author={Park, Junwoo and Lee, Sujee},
  journal={Frontiers in Toxicology},
  volume={7},
  pages={1720651},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact:
- Email: jw0528@g.skku.edu

---

**Last Updated**: December 2025
