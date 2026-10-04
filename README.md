# RetiNA-Net: Multi-Scale Deformable Attention & Hierarchical Feature Fusion for Diabetic Retinopathy Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning framework for robust Diabetic Retinopathy (DR) grading, designed to achieve high internal accuracy on **APTOS 2019** (targeting 97.5% test accuracy with high QWK) and robust generalization across clinical distribution shifts on **Messidor-2** (targeting 90%+ external accuracy).

---

## Table of Contents
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Usage Guide](#usage-guide)
  - [1. Model Training](#1-model-training)
  - [2. Decoupled Classifier Retraining](#2-decoupled-classifier-retraining)
  - [3. Evaluation & Testing](#3-evaluation--testing)
  - [4. External Validation](#4-external-validation)
  - [5. Explainable AI (XAI)](#5-explainable-ai-xai)
  - [6. SSL & Pretraining](#6-ssl--pretraining)
  - [7. Diffusion Synthetic Augmentation Pipeline](#7-diffusion-synthetic-augmentation-pipeline)
  - [8. Automated Results & Paper Table Generation](#8-automated-results--paper-table-generation)
  - [9. Architecture Smoke Tests](#9-architecture-smoke-tests)
- [Ablation Studies](#ablation-studies)
- [Configuration Reference](#configuration-reference)
- [Architecture Overview](#architecture-overview)

---

## Key Features

1. **Vision Transformer Backbone with Partial Freezing**:
   - Powered by **SwinV2-Large** (or SwinV2-Base), capturing fine-grained multi-scale retinal fundus characteristics.
   - Configurable backbone stage freezing (`freeze_backbone_stages`) and layer-wise learning rate decay to prevent overfitting on small clinical cohorts.

2. **Novel Multi-Scale Deformable Attention (MSDA)**:
   - Dynamic, deformable receptive fields on deep feature stages (Stages 3 & 4) tailored to irregular lesion shapes (microaneurysms, hemorrhages, hard/soft exudates).

3. **Hierarchical Feature Fusion (HFF) with Structural Reparameterization**:
   - Learnable cross-stage gating bridging high-resolution spatial details (Stage 2) with high-level semantics (Stage 4).
   - **RepConv / RepVGG-style reparameterization** (`use_rep_proj: True`): multi-branch residual/identity convolutions during training that fuse losslessly into single $3\times3$ convolutions at test time for zero-latency overhead.

4. **Class-Imbalance Mitigations & Ordinal Regression**:
   - **Logit Adjustment** and **Deferred Re-Balancing (DRW)** with effective-number/inverse-frequency class weighting.
   - Dual-head formulation: Standard multi-class classification alongside an **Ordinal Regression Head** with cumulative logits to penalize distance errors across DR severity grades.
   - **Decoupled classifier retraining** (`--mode decoupled_retrain`): fixes backbone representations and retrains the classification head under balanced sampling.
   - Class-conditioned CutMix and adaptive confidence thresholding.

5. **Fourier Domain Adaptation (FDA)**:
   - Reduces domain shift between clinical datasets (e.g., APTOS-19 $\rightarrow$ Messidor-2) by swapping low-frequency Fourier amplitude spectrum components from an external style pool.

6. **Lesion-Guided Latent Diffusion Augmentation (P6)**:
   - Conditional Latent Diffusion pipeline (VAE + conditional U-Net) with 2D FFT spectral loss and classifier-based confidence/disagreement filtering to synthesize realistic minority-class lesions (Grades 3 & 4).

7. **Robust Optimization & Regularization**:
   - **Sharpness-Aware Minimization (SAM)** option for finding flatter loss minima.
   - **Stochastic Weight Averaging (SWA)** and **Exponential Moving Average (EMA)** for model stabilization.
   - Multi-scale **Test-Time Augmentation (TTA)** (5 scales, rotations, flips, multi-crop) and 5-fold cross-validation ensembling.

8. **Automated Results & Publication-Ready Tables**:
   - Single-command automated generation of complete LaTeX and Markdown tables, ROC/PR curves, confusion matrices, radar charts, and Grad-CAM lesion heatmaps.

---

## Repository Structure

```
DR/
├── configs/
│   ├── config.yaml                # Main training and evaluation configuration
│   ├── config_ssl.yaml            # Self-supervised pretraining configuration
│   └── config_diffusion.yaml      # Latent diffusion generation configuration
├── datasets/                      # Directory for APTOS-19, Messidor-2, EyePACS
├── instructions/                  # Detailed implementation plans (P1 - P7)
├── results/                       # Persisted .npz metrics, generated tables, and plots
├── checkpoints/                   # Saved model weights across folds and ablations
├── logs/                          # Execution logs with timestamps
├── src/
│   ├── data/                      # Data loaders, Albumentations pipelines, K-fold split
│   ├── models/                    # RetiNA-Net, SwinV2 backbone, MSDA, HFF, RepConv
│   ├── training/                  # DRTrainer, SAM, SWA, Loss functions (Focal, LDAM, Ordinal)
│   ├── evaluation/                # Metrics, TTA, Ensemble, Grad-CAM, Results Logger
│   ├── generation/                # VAE, Diffusion U-Net, Spectral Loss, Filtering
│   └── preprocessing/             # CLAHE, Vessel segmentation, Lesion detection
├── main.py                        # Unified entry point for all workflows
├── test_model.py                  # PyTorch architectural smoke tests
├── requirements.txt               # Python package dependencies
└── README.md
```

---

## Environment Setup

### 1. Conda Environment
```bash
conda create -n dr_retina python=3.12 -y
conda activate dr_retina
```

### 2. Install PyTorch & Dependencies
Install PyTorch matching your CUDA version (e.g., CUDA 12.4):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> **Note on Air-Gapped / Cluster Environments**: `main.py` defaults to `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Ensure pretrained backbone checkpoints are placed locally in `checkpoints/` or the torch cache if running offline.

---

## Dataset Preparation

Organize the datasets inside the `datasets/` directory:

```
datasets/
├── APTOS_19/
│   ├── train_images/
│   ├── val_images/
│   ├── test_images/
│   ├── train_1.csv                # columns: id_code, diagnosis
│   ├── valid.csv
│   └── test.csv
├── Messidor_2/
│   ├── images/
│   └── messidor_data.csv          # columns: id_code, diagnosis, adjudicated_dme, adjudicated_gradable
└── EyePACS/                       # (Optional, for SSL pretraining)
    ├── images/
    └── train.csv
```

---

## Usage Guide

All major workflows are accessible via `main.py` using the `--mode` flag.

### 1. Model Training

#### Full 5-Fold Cross-Validation (Recommended)
Trains models across all 5 folds with SWA, EMA, and regularizations:
```bash
python main.py --mode train --ablation proposed
```

#### Train a Single Fold
```bash
python main.py --mode train --ablation proposed --fold 0
```

#### Standard Train/Val Split (No K-Fold)
Set `use_kfold: False` in `configs/config.yaml`, then run:
```bash
python main.py --mode train --ablation proposed
```

### 2. Decoupled Classifier Retraining
After representation learning finishes, freeze the backbone and retrain only the linear classification head under balanced class sampling:
```bash
python main.py --mode decoupled_retrain --ablation proposed --fold 0
```

### 3. Evaluation & Testing
Run ensemble inference across all 5 trained fold checkpoints on the APTOS-19 test set using multi-scale Test-Time Augmentation (TTA):
```bash
python main.py --mode test --ablation proposed
```
*Results are automatically logged to `results/ablation_results.npz`.*

### 4. External Validation
Evaluate cross-domain generalization on the un-seen **Messidor-2** dataset:
```bash
python main.py --mode external_validation --ablation proposed
```
*Evaluates both 5-class DR classification and binary Referable DR (RDR, Grade $\ge$ 2) AUC/Sensitivity/Specificity.*

### 5. Explainable AI (XAI)
Generate Grad-CAM heatmaps to visualize the model's focus on microaneurysms, hemorrhages, and exudates:
```bash
python main.py --mode xai --ablation proposed
```

### 6. SSL & Pretraining
- **Lesion Detection Mask Extraction**:
  ```bash
  python main.py --mode detect_lesions
  ```
- **Self-Supervised Pretraining (EyePACS)**:
  ```bash
  python main.py --mode pretrain --config configs/config_ssl.yaml
  ```
- **Supervised EyePACS Pretraining**:
  ```bash
  python main.py --mode pretrain_eyepacs_supervised
  ```

### 7. Diffusion Synthetic Augmentation Pipeline
For minority class synthesis (Grades 3 & 4):
- **Train VAE**:
  ```bash
  python main.py --mode train_vae --config configs/config_diffusion.yaml
  ```
- **Train Latent Diffusion U-Net**:
  ```bash
  python main.py --mode train_diffusion_unet --config configs/config_diffusion.yaml
  ```
- **Generate Synthetic Samples**:
  ```bash
  python main.py --mode generate_synthetic --config configs/config_diffusion.yaml
  ```

### 8. Automated Results & Paper Table Generation
Generate all markdown and LaTeX ablation tables, SOTA comparison tables, and figures from persisted `results/*.npz` runs without manual bookkeeping:
```bash
python main.py --mode generate_results
```
Outputs are saved in `results/`:
- `architecture_ablation_table.md` & `.tex`
- `ssl_ablation_table.md` & `.tex`
- `per_class_metrics_table.md` & `.tex`
- `sota_comparison_table.md` & `.tex`
- `external_validation_table.md` & `.tex`

### 9. Architecture Smoke Tests
Verify tensor shapes, auxiliary logits, ordinal heads, and probability outputs across all ablation modes:
```bash
python test_model.py
```

---

## Ablation Studies

### Standard Ablation Modes
| Mode | MSDA | HFF | Description | CLI Flag |
|---|:---:|:---:|---|---|
| **Baseline** | ✗ | ✗ | SwinV2 backbone + classification head | `--ablation baseline` |
| **MSDA Only** | ✓ | ✗ | SwinV2 + Multi-Scale Deformable Attention | `--ablation msda_only` |
| **HFF Only** | ✗ | ✓ | SwinV2 + Hierarchical Feature Fusion | `--ablation hff_only` |
| **Proposed** | ✓ | ✓ | Full RetiNA-Net architecture | `--ablation proposed` |

### Fine-Grained Component Ablation Registry
For comprehensive paper tables, the extended architecture registry isolates individual components:

| Identifier | MSDA | HFF | Attention Pool | Aux Head | Ordinal Head |
|---|:---:|:---:|:---:|:---:|:---:|
| `arch_baseline` | ✗ | ✗ | ✗ | ✗ | ✗ |
| `arch_msda` | ✓ | ✗ | ✗ | ✗ | ✗ |
| `arch_hff` | ✗ | ✓ | ✗ | ✗ | ✗ |
| `arch_msda_hff` | ✓ | ✓ | ✗ | ✗ | ✗ |
| `arch_attnpool` | ✓ | ✓ | ✓ | ✗ | ✗ |
| `arch_auxhead` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `arch_full` | ✓ | ✓ | ✓ | ✓ | ✓ |

Use via `--ablation <identifier>`, e.g.:
```bash
python main.py --mode train --ablation arch_full
```

---

## Configuration Reference

Key settings in `configs/config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `img_size` | `512` | Input image resolution ($512 \times 512$) |
| `backbone` | `swinv2_large...` | Vision Transformer backbone architecture |
| `freeze_backbone_stages` | `2` | Number of initial Swin stages frozen to combat overfitting |
| `use_rep_proj` | `False` | Enable RepVGG structural reparameterization in HFF projections |
| `batch_size` | `8` | Per-GPU batch size |
| `accumulation_steps` | `4` | Gradient accumulation steps (Effective batch size = 32) |
| `lr` | `3.0e-4` | Peak learning rate |
| `backbone_lr_mult` | `0.2` | Differential learning rate multiplier for backbone |
| `loss_type` | `"focal"` | Primary classification loss: `"focal"` or `"ldam"` |
| `use_ordinal_loss` | `True` | Cumulative ordinal regression loss toggle |
| `use_logit_adjustment` | `False` | Logit Adjustment for long-tail class imbalance |
| `use_drw` | `False` | Deferred Re-Balancing schedule |
| `use_fda` | `False` | Fourier Domain Adaptation toggle for external transfer |
| `fda_beta` | `0.03` | Low-frequency spectral swap fraction |
| `use_sam` | `False` | Sharpness-Aware Minimization optimizer toggle |
| `use_swa` | `True` | Stochastic Weight Averaging enabled |
| `use_ema` | `True` | Exponential Moving Average of weights (decay = 0.999) |
| `use_tta` | `True` | Multi-scale Test-Time Augmentation |

---

## Architecture Overview

```
                        Input Fundus Image (3, 512, 512)
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │    SwinV2-Large Transformer Backbone   │
                   └───────────────────────────────────────┘
                                       │
       ├── Stage 1 (192, 128, 128)  ── Low-level retinal features
       ├── Stage 2 (384, 64, 64)    ── Fine-grained vascular/lesion details
       ├── Stage 3 (768, 32, 32)    ── Mid-level lesion context
       └── Stage 4 (1536, 16, 16)   ── Deep semantic context
            │               │                             │
            │               ▼                             ▼
            │        ┌──────────────┐              ┌──────────────┐
            │        │  Aux Head    │              │     HFF      │
            │        │  (Stage 3)   │              │ (Stage 2 → 4)│
            │        └──────────────┘              │ (RepConv 3x3)│
            │               │                      └──────────────┘
            │               ▼                             │
            │         [Aux Logits]                        │
            ▼                                             │
      ┌───────────┐                                       │
      │   MSDA    │                                       │
      │(Stage 3/4)│                                       │
      └───────────┘                                       │
            │                                             │
            └──────────────────────┬──────────────────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │   Spatial Attention Pooling   │
                   │      (AvgPool + MaxPool)      │
                   └───────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌────────────────────────┐    ┌────────────────────────┐
       │  Classification Head   │    │ Ordinal Regression Head│
       │   (5 Severity Grades)  │    │  (4 Cumulative Logits) │
       └────────────────────────┘    └────────────────────────┘
```
