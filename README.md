# Multi-Modal Molecular Toxicity Prediction Framework

A comprehensive deep learning framework for molecular toxicity prediction using single and multi-modal approaches. The framework combines molecular graphs, SMILES sequences, 2D structure images, and NMR spectra to predict toxicity across 12 different endpoints from the Tox21 dataset.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [References](#references)

## Overview

This framework implements state-of-the-art molecular representation learning techniques to predict molecular toxicity. It supports:
- **4 Single-Modal Encoders**: Graph (GNN), SMILES (Transformer), Image (CNN), Spectrum (1D CNN)
- **10 Multi-Modal Fusion Models**: All pairwise and higher-order combinations of the modalities
- **12 Toxicity Endpoints**: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53

The multi-modal models use self-attention fusion mechanisms to effectively combine complementary information from different molecular representations.

## Features

- ✅ **Modular Architecture**: Easily extensible encoder and fusion architectures
- ✅ **Automated Hyperparameter Tuning**: Optuna-based TPE optimization
- ✅ **Early Stopping**: Prevents overfitting with patience-based validation
- ✅ **Two-Stage Training**: Train/valid split optimization → Full dataset retraining
- ✅ **Pre-trained Backbones**:
  - MoLFormer-XL for SMILES encoding
  - ImageMol (ResNet18) for image encoding
  - CReSS for NMR spectrum encoding
- ✅ **Transfer Learning**: Freeze pre-trained encoders and train fusion layers
- ✅ **Multi-Task Learning**: Simultaneous prediction of 12 toxicity endpoints
- ✅ **Reproducible**: Fixed random seeds (seed=42) for Optuna sampling

## Project Structure

```
my_model/
├── models/
│   ├── models.py              # All encoder and classifier models
│   └── __init__.py
├── utils/
│   ├── dataset.py             # Dataset classes for all modalities
│   └── __init__.py
├── experiments/
│   ├── graph/                 # Graph-based experiments
│   │   ├── train.py
│   │   └── test.py
│   ├── smiles/                # SMILES-based experiments
│   │   ├── train.py
│   │   └── test.py
│   ├── image/                 # Image-based experiments
│   │   ├── train.py
│   │   └── test.py
│   ├── spectrum/              # Spectrum-based experiments
│   │   ├── train.py
│   │   ├── test.py
│   │   └── CReSS/             # CReSS model for NMR encoding
│   └── multimodal/            # Multi-modal fusion experiments
│       ├── gph_smi/           # Graph + SMILES
│       ├── gph_img/           # Graph + Image
│       ├── gph_spec/          # Graph + Spectrum
│       ├── smi_img/           # SMILES + Image
│       ├── smi_spec/          # SMILES + Spectrum
│       ├── spec_img/          # Spectrum + Image
│       ├── gph_smi_img/       # Graph + SMILES + Image
│       └── moltitox/          # Graph + SMILES + Image + Spectrum (Full)
├── data/
│   ├── train.csv              # Training data
│   ├── valid.csv              # Validation data
│   ├── test.csv               # Test data
│   ├── train_spectra.csv      # Training data with spectra
│   ├── valid_spectra.csv      # Validation data with spectra
│   ├── test_spectra.csv       # Test data with spectra
│   ├── images/                # Molecular 2D structure images
│   └── spectra/               # NMR spectral data (.npy files)
├── checkpoints/
│   ├── encoder/               # Saved encoder weights
│   │   ├── train_only/        # Encoders trained on train set only
│   │   └── train_and_valid/   # Encoders trained on train+valid
│   ├── model/                 # Saved full model weights
│   ├── parameters/            # Best hyperparameters (JSON)
│   └── pretrained_models/     # Pre-trained model checkpoints
├── main.py                    # Run all experiments sequentially
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Model Architecture

### Single-Modal Encoders

1. **GraphEncoder** (`models.models.GraphEncoder`)
   - Architecture: Graph Isomorphism Network (GIN) with GINEConv layers
   - Input: Molecular graph with node features (78-dim) and edge features (4-dim)
   - Layers: Multiple GINEConv → BatchNorm → ReLU → Dropout
   - Output: Fixed-size graph embedding

2. **SMILESEncoder** (`models.models.SMILESEncoder`)
   - Architecture: Pre-trained MoLFormer-XL transformer
   - Input: Tokenized SMILES strings (max length: 202)
   - Layers: Transformer encoder + 2-layer MLP with residual connection
   - Output: Fixed-size SMILES embedding

3. **ImageEncoder** (`models.models.ImageEncoder`)
   - Architecture: Pre-trained ResNet18 (ImageMol)
   - Input: 224×224 RGB images of 2D molecular structures
   - Layers: ResNet18 backbone + 2-layer MLP
   - Output: Fixed-size image embedding

4. **SpectrumEncoder** (`models.models.SpectrumEncoder`)
   - Architecture: Pre-trained CReSS NMR encoder
   - Input: List of PPM values from ¹³C NMR spectra
   - Layers: 1D CNN encoder + 2-layer MLP
   - Output: Fixed-size spectrum embedding

### Multi-Modal Fusion Models

All multi-modal models follow a unified architecture:
1. **Projection**: Project each modality embedding to common dimension
2. **Fusion**: Multi-head self-attention over modality tokens
3. **Pooling**: Mean pooling across modality dimension
4. **Classification**: MLP head for multi-task prediction

Available fusion models:
- `MultiModalGphSMI`: Graph + SMILES
- `MultiModalGphImg`: Graph + Image
- `MultiModalGphSpec`: Graph + Spectrum
- `MultiModalSMIImg`: SMILES + Image
- `MultiModalSMISpec`: SMILES + Spectrum
- `MultiModalSpecImg`: Spectrum + Image
- `MultiModalGphSMIImg`: Graph + SMILES + Image
- `MoltiTox`: Graph + SMILES + Image + Spectrum (Full)

### Loss Function

Binary cross-entropy with logits, masked for missing labels:
```python
mask = (labels >= 0).float()
targets = labels.clamp(min=0)
per_sample_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
loss = (per_sample_loss * mask).sum() / mask.sum()
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

### Setup

```bash
# Clone the repository
git clone <repository_url>
cd my_model

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
# - Place ImageMol checkpoint in: checkpoints/pretrained_models/ImageMol.pth.tar
# - CReSS model files in: experiments/spectrum/8.json and experiments/spectrum/8.pth
```

## Usage

### Running All Experiments

To run all experiments sequentially (single-modal + multi-modal):

```bash
python -u main.py 2>&1 | tee result.txt
```

This will:
1. Train and test all 4 single-modal models
2. Train and test all 10 multi-modal models
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

# Multi-modal examples
cd ../multimodal/gph_smi
python train.py  # Train Graph+SMILES fusion model
python test.py   # Test Graph+SMILES fusion model

cd ../smi_img
python train.py  # Train SMILES+Image fusion model
python test.py   # Test SMILES+Image fusion model
```

### Training Pipeline

Each training script follows this pipeline:

1. **Load Data**: Load train/valid splits and create datasets
2. **Load Encoders**: Load pre-trained encoders (for multi-modal) and freeze weights
3. **Hyperparameter Search**:
   - Optuna TPE sampler with 10-20 trials
   - Search space: embedding dim, hidden dim, learning rate, weight decay, dropout
   - Early stopping with patience (5-10 epochs)
   - Maximize validation AUC
4. **Save Best Parameters**: Save best hyperparameters to JSON
5. **Retrain on Full Data**:
   - Load train+valid encoders
   - Retrain with best hyperparameters for N epochs (from step 3)
   - Save final model checkpoint

### Testing Pipeline

Each testing script:
1. Loads test data
2. Loads best hyperparameters
3. Loads trained model checkpoint
4. Evaluates on test set
5. Reports per-task AUC and mean AUC

## Experiments

### Hyperparameter Search Spaces

**Single-Modal Models:**
```python
hidden_dim: [256, 512]
num_layers: [3, 4]
emb_dim: [256, 512]
lr: [1e-4, 3e-4, 1e-3]
weight_decay: [1e-5, 1e-4, 1e-3]
```

**Multi-Modal Models:**
```python
emb_dim: [128, 256]
hidden_dim: [512, 768]
num_layers: [1]
lr: [1e-4, 3e-4]
weight_decay: [1e-5, 1e-4]
dropout: [0.3, 0.4, 0.5]
```

### Training Configuration

- **Optimizer**: AdamW
- **Batch Size**: 32 (train), 64 (valid/test)
- **Max Epochs**: 30-50 (depending on model)
- **Early Stopping Patience**: 5-10 epochs
- **Metric**: ROC-AUC (averaged across tasks)
- **Device**: CUDA if available, otherwise CPU

## Results

The framework generates results for 12 Tox21 toxicity endpoints:

| Endpoint | Description |
|----------|-------------|
| NR-AR | Nuclear Receptor - Androgen Receptor |
| NR-AR-LBD | NR-AR Ligand Binding Domain |
| NR-AhR | Nuclear Receptor - Aryl Hydrocarbon Receptor |
| NR-Aromatase | Nuclear Receptor - Aromatase |
| NR-ER | Nuclear Receptor - Estrogen Receptor |
| NR-ER-LBD | NR-ER Ligand Binding Domain |
| NR-PPAR-gamma | NR - Peroxisome Proliferator-Activated Receptor Gamma |
| SR-ARE | Stress Response - Antioxidant Response Element |
| SR-ATAD5 | Stress Response - ATAD5 |
| SR-HSE | Stress Response - Heat Shock Element |
| SR-MMP | Stress Response - Mitochondrial Membrane Potential |
| SR-p53 | Stress Response - p53 |

Results are reported as ROC-AUC scores for each endpoint and mean AUC across all endpoints.

### Output Format

Test results are printed in this format:
```
Total testing time: X.XX seconds
NR-AR          : 0.XXXX
NR-AR-LBD      : 0.XXXX
...
SR-p53         : 0.XXXX
------------------------------
Mean AUC       : 0.XXXX
```

## File Descriptions

### Core Model Files

- **`models/models.py`**: Contains all encoder and classifier implementations
  - Single-modal: GraphEncoder, SMILESEncoder, ImageEncoder, SpectrumEncoder
  - Classifiers: GraphClassifier, SMILESClassifier, ImageClassifier, SpectrumClassifier
  - Multi-modal: All fusion model classes

- **`utils/dataset.py`**: Dataset classes for loading and preprocessing
  - Single-modal: GraphDataset, SMILESDataset, ImageDataset, SpectrumDataset
  - Multi-modal: GraphSMILESDataset, GraphImageDataset, etc.

### Checkpoints

Models and parameters are saved to:
- `checkpoints/encoder/train_only/`: Encoders trained on train set (for validation)
- `checkpoints/encoder/train_and_valid/`: Encoders trained on train+valid (for test)
- `checkpoints/model/`: Full model checkpoints (encoder + classifier)
- `checkpoints/parameters/`: Best hyperparameters in JSON format

## Key Design Decisions

1. **Two-Stage Training**: First optimize hyperparameters on train/valid split, then retrain on full train+valid with best hyperparameters for final test evaluation.

2. **Frozen Encoders**: For multi-modal models, pre-trained single-modal encoders are frozen to prevent overfitting and reduce training time.

3. **Self-Attention Fusion**: Multi-head attention allows the model to learn adaptive weights for each modality based on the input.

4. **Masked Loss**: Handles missing labels in the Tox21 dataset without introducing bias.

5. **Data Augmentation**: Image data uses standard augmentations (flip, rotation, grayscale) during training.

## References

### Pre-trained Models

1. **MoLFormer-XL**: Ross, J., et al. (2022). "Large-Scale Chemical Language Representations Capture Molecular Structure and Properties." *Nature Machine Intelligence*.

2. **ImageMol**: Zhu, J., et al. (2022). "Accurate Prediction of Molecular Properties and Drug Targets Using a Self-Supervised Image Representation Learning Framework." *Nature Machine Intelligence*.

3. **CReSS**: Kwon, Y., et al. (2021). "Cross-Modal Retrieval between ¹³C NMR Spectra and Structures for Compound Identification Using Deep Contrastive Learning." *Journal of Chemical Information and Modeling*.

### Architectures

4. **GIN**: Xu, K., et al. (2019). "How Powerful are Graph Neural Networks?" *ICLR*.

### Dataset

5. **Tox21**: Huang, R., et al. (2016). "Tox21Challenge to Build Predictive Models of Nuclear Receptor and Stress Response Pathways as Mediated by Exposure to Environmental Chemicals and Drugs." *Frontiers in Environmental Science*.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{multimodal_tox_framework,
  title={Multi-Modal Molecular Toxicity Prediction Framework},
  author={JunWoo},
  year={2025},
  url={https://github.com/...}
}
```

## License

[Add your license here]

## Contact

For questions or issues, please contact:
- Email: [your email]
- GitHub Issues: [repository issues page]

---

**Last Updated**: October 2025
