# DR Classification — Multi-Scale Deformable Attention & Hierarchical Feature Fusion

Diabetic Retinopathy (DR) grading using SwinV2-Base backbone with novel MSDA and HFF modules.

## Key Features

- **Backbone**: SwinV2-Base (pretrained on ImageNet-22k, fine-tuned on ImageNet-1k)
- **MSDA**: Multi-Scale Deformable Attention on deep feature stages
- **HFF**: Hierarchical Feature Fusion with learnable gating (Stage 2 → Stage 4)
- **Ordinal-aware loss**: Combined Focal Loss + Ordinal Regression Loss
- **Mixup/CutMix**: Strong regularization for small datasets
- **SWA**: Stochastic Weight Averaging for flatter minima
- **EMA**: Exponential Moving Average of model weights
- **K-Fold Ensemble**: 5-fold cross-validation with ensemble inference
- **Multi-scale TTA**: Test-time augmentation with flips, rotations, and multi-scale
- **CLAHE preprocessing**: Enhanced lesion contrast for microaneurysm visibility

## Quick Start

### 1. Environment Setup

```bash
conda create -n dr_iccit python=3.12 -y
conda activate dr_iccit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Dataset Setup

Place datasets in the following structure:
```
datasets/
├── APTOS_19/
│   ├── train_images/
│   ├── val_images/
│   ├── test_images/
│   ├── train_1.csv      # columns: id_code, diagnosis
│   ├── valid.csv
│   └── test.csv
└── Messidor_2/
    ├── images/
    └── messidor_data.csv  # columns: id_code, diagnosis, adjudicated_dme, adjudicated_gradable
```

### 3. Training

#### Full K-Fold Training (recommended for best accuracy)
```bash
python main.py --mode train --ablation proposed
```
This trains 5 models (one per fold) and saves each as `best_iccit_model_proposed_fold{0-4}.pth`.

#### Train a Single Fold
```bash
python main.py --mode train --ablation proposed --fold 0
```

#### Standard Train/Val Split (no K-fold)
Set `use_kfold: False` in `configs/config.yaml`, then:
```bash
python main.py --mode train --ablation proposed
```

### 4. Testing

```bash
python main.py --mode test --ablation proposed
```
Runs ensemble inference across all K-fold models with multi-scale TTA.

### 5. External Validation (Messidor-2)

```bash
python main.py --mode external_validation --ablation proposed
```

### 6. Explainable AI (Heatmaps)

```bash
python main.py --mode xai --ablation proposed
```

### 7. Architecture Tests

```bash
python test_model.py
```

## Ablation Configurations

| Config | MSDA | HFF | Command |
|--------|------|-----|---------|
| Baseline | ✗ | ✗ | `--ablation baseline` |
| MSDA Only | ✓ | ✗ | `--ablation msda_only` |
| HFF Only | ✗ | ✓ | `--ablation hff_only` |
| Proposed | ✓ | ✓ | `--ablation proposed` |

## Configuration

All hyperparameters are in `configs/config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_size` | 384 | Input image resolution |
| `batch_size` | 16 | Batch size (A100) |
| `accumulation_steps` | 2 | Gradient accumulation (effective BS = 32) |
| `lr` | 5e-5 | Base learning rate |
| `backbone_lr_mult` | 0.05 | Backbone LR multiplier |
| `epochs` | 50 | Max training epochs |
| `patience` | 15 | Early stopping patience |
| `use_kfold` | True | K-fold cross-validation |
| `n_folds` | 5 | Number of K-fold splits |
| `use_mixup` | True | Mixup/CutMix augmentation |
| `use_swa` | True | Stochastic Weight Averaging |
| `use_ordinal_loss` | True | Ordinal regression loss |
| `use_tta` | True | Test-time augmentation |

## Architecture Overview

```
Input (3, 384, 384)
    │
    ▼
SwinV2-Base Backbone (features_only)
    │
    ├── Stage 1 (128, 96, 96)   ── Low-level features
    ├── Stage 2 (256, 48, 48)   ── Fine-grained detail
    ├── Stage 3 (512, 24, 24)   ── Mid-level semantics
    └── Stage 4 (1024, 12, 12)  ── Deep semantics
         │                              │
         ▼                              ▼
       MSDA                            HFF
    (Stage 3 & 4)          (Stage 2 → Stage 4 fusion)
         │                              │
         └──────────┬───────────────────┘
                    ▼
          Multi-Scale Head
     (AvgPool + MaxPool → 2048-dim)
                    │
                    ├── Classification Head (5 classes)
                    └── Ordinal Head (4 cumulative logits)
```
