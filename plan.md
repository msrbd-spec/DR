# RetiNA-Net: Complete A-Z Plan for Q1/A* Conference Standard

## Paper Title (Suggested)
> "RetiNA-Net: Lesion-Aware Self-Supervised Pretraining with Multi-Scale Deformable Attention for Diabetic Retinopathy Grading"

---

## PHASE 1: DATA PREPROCESSING

### 1.1 Dataset Structure
```
datasets/
├── APTOS_19/          (fine-tuning + internal test)
├── EyePACS/           (SSL pretraining, 88,702 images, no labels)
└── Messidor_2/        (external validation)
```

### 1.2 Preprocessing Pipeline (all datasets)
1. **Fundus circle cropping** — crop black borders by detecting the circular fundus region
2. **Ben Graham preprocessing** — Gaussian blur subtraction to enhance lesions
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — on green channel (best for vessel visibility)
4. **Resize** — 512×512 for RetiNA-Net training
5. **Normalization** — ImageNet mean/std (matching pretrained backbone)

### 1.3 Data Splits
| Dataset | Purpose | Split |
|---------|---------|-------|
| EyePACS (88,702) | SSL pretraining | No split needed (all images, no labels) |
| APTOS-2019 (3,662) | Fine-tuning + internal test | 80% train/val (5-fold CV), 20% held-out test |
| Messidor-2 (~1,200) | External validation | Entire set (no training) |

---

## PHASE 2: CLASSICAL LESION DETECTION (for SSL pseudo-labels)

### 2.1 Lesion Detection Module (`src/preprocessing/lesion_detection.py`)
| Lesion Type | Detection Method |
|-------------|-----------------|
| **Microaneurysms** | Green channel → morphological top-hat → circular Hough transform → radius filter (3-15px) |
| **Hemorrhages** | Green channel → adaptive thresholding → connected components → area filter (>20px) |
| **Exudates** | LAB color space → high L & B channel → morphological closing → area filter |
| **Drusen** (false positive filter) | Exclude small bright spots near macula (within 1 disc diameter) |

### 2.2 Output per Image
For each of the 88,702 EyePACS images, generate:
- **Lesion type mask** (3 binary masks: microaneurysm, hemorrhage, exudate)
- **Lesion count** (integer per type)
- **Lesion presence** (binary multi-label: [microaneurysm_present, hemorrhage_present, exudate_present])
- **Lesion location heatmap** (Gaussian-blurred density map)

### 2.3 Quality Control
- Sanity check: detect on 100 known-severe EyePACS images (from `trainLabels.csv` level 3-4)
- Verify lesion detection correlates with DR grade (should show positive correlation)
- This correlation analysis becomes a **figure in the paper** (Fig: "Lesion detection quality vs DR grade")

---

## PHASE 3: SSL PRETRAINING (Lesion-Aware Contrastive + Multi-Task)

> **Combined approach:** Lesion-Aware Contrastive Learning (Option 1) + Lesion-Guided Multi-Task SSL (Option 4)

### 3.1 Pretraining Architecture (`src/models/ssl_model.py`)
```
SwinV2-Large backbone (shared, pretrained=ImageNet)
├── Projection head (MLP 1536→512→128) → Contrastive loss (InfoNCE)        [Option 1]
├── Lesion type head (Multi-label BCE, 3 outputs) → from stage-3 features   [Option 4]
├── Lesion count head (MSE regression, 3 outputs) → from stage-4 features   [Option 4]
└── Lightweight decoder (ConvTranspose → 128×128) → Reconstruction loss     [Option 4]
```

### 3.2 Pretraining Loss
```
L_pretrain = 1.0 * L_contrastive (InfoNCE, temperature=0.07)           [Option 1 — Lesion-Aware Contrastive]
           + 0.5 * L_lesion_type (BCE multi-label)                      [Option 4 — Lesion-Guided Multi-Task]
           + 0.3 * L_lesion_count (MSE)                                 [Option 4]
           + 0.2 * L_reconstruction (L1 on lesion masks)                 [Option 4]
```

**How they combine:**
- **Lesion-Aware Contrastive (Option 1):** The contrastive branch learns invariant representations across images. Positive pairs are created from augmentations of the same image; the model learns what makes two fundus images similar in a DR-aware feature space.
- **Lesion-Guided Multi-Task (Option 4):** The multi-task branch uses auto-generated lesion pseudo-labels (from classical detection) as supervision. The model learns to predict lesion type, count, and location — all without human labels.
- **Together:** Contrastive provides generalizable features; multi-task provides DR-specific supervision. They complement each other.

### 3.3 Pretraining Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1.5e-4 (backbone 0.1×) |
| Weight decay | 0.05 |
| Scheduler | Cosine with 5-epoch warmup |
| Batch size | 64 (effective, with accumulation) |
| Epochs | 50 |
| Temperature (contrastive) | 0.07 |
| Augmentations | RandomResizedCrop, ColorJitter, GaussianBlur, HorizontalFlip |
| Mixed precision | AMP (fp16) |
| EMA | Yes, decay=0.996 |

### 3.4 Pretraining Data Augmentation
- Two augmented views per image (for contrastive)
- Augmentations: RandomResizedCrop(512), ColorJitter(brightness, contrast, saturation), GaussianBlur, HorizontalFlip
- **No vertical flip or rotation** (fundus has canonical vertical orientation)

### 3.5 Save Output
- Save pretrained backbone weights → `checkpoints/ssl_pretrained_backbone.pth`
- Save training curves → `results/ssl_pretrain_curves.png`

---

## PHASE 4: FINE-TUNING ON APTOS-2019

### 4.1 Model: RetiNA-Net (already implemented)
Load **SSL-pretrained backbone** instead of ImageNet backbone.

### 4.2 Architecture (unchanged from current)
| Component | Details |
|-----------|---------|
| Backbone | SwinV2-Large (SSL-pretrained) |
| MSDA | DeformConv2d on stages 3 & 4, deform_groups=4 |
| HFF | Learnable gated fusion of all 4 stages → 1536ch |
| Head | Spatial Attention Pooling → 2×1536 → 512 → 5 |
| Aux head | Stage-3 features → 256 → 5 (deep supervision) |
| Ordinal head | Stage-4 features → 256 → 4 cumulative logits |

### 4.3 Fine-Tuning Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 3e-4 (backbone 0.2×) |
| Weight decay | 0.01 (head: 0.05) |
| Scheduler | Cosine with 10-epoch warmup |
| Batch size | 8 (accumulation=4, effective=32) |
| Epochs | 80 |
| Loss | Focal(γ=1.5) + Ordinal(0.3) + Aux(0.2) |
| EMA | decay=0.999 |
| SWA | start=50, lr=1e-6, anneal=5 |
| Mixup | α=0.4, prob=0.1 |
| CutMix | α=1.0, prob=0.3 |
| Label smoothing | 0.0 |
| Grad clip | 1.0 |
| Gradient checkpointing | Enabled |

### 4.4 Training
- 5-fold Stratified K-Fold CV
- Each fold: 80 epochs with early stopping (patience=25 on QWK)
- Save best EMA/SWA model per fold

### 4.5 Inference
- 5-fold ensemble (probability averaging with 0.7/0.3 cls+ord weighting)
- Multi-scale TTA: 5 scales × flips × rotations × 2 center-crops

---

## PHASE 5: EXTERNAL VALIDATION ON MESSIDOR-2

### 5.1 Process
1. Load 5-fold ensemble models
2. Apply same preprocessing (fundus crop, Ben Graham, CLAHE, resize 512)
3. Run ensemble + TTA inference
4. Report all metrics

### 5.2 Metrics to Report
| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Precision (macro) | Macro-averaged precision |
| Recall (macro) | Macro-averaged recall |
| F1 (macro) | Macro-averaged F1 |
| F1 (weighted) | Weighted F1 (accounts for imbalance) |
| QWK | Quadratic Weighted Kappa |
| Cohen's Kappa | Linear kappa |
| AUC (macro OVR) | One-vs-Rest AUC |
| Sensitivity per class | Per-class recall |
| Specificity per class | Per-class specificity |

---

## PHASE 6: EXPLAINABLE AI (XAI)

### 6.1 Methods
- **EigenCAM** — fast, no gradient needed, works on SwinV2
- **GradCAM** — gradient-based, highlights discriminative regions

### 6.2 Target Layer
- If MSDA enabled: `msda4.deform_conv` (final deformable conv)
- If MSDA disabled: last SwinV2 stage block norm layer

### 6.3 Visualization
- Generate heatmaps for **5 sample images** (one per DR grade 0-4)
- Overlay heatmap on original fundus image
- Side-by-side: original | EigenCAM | GradCAM
- Save as `results/xai_grid_proposed.png`

---

## PHASE 7: ABLATION STUDIES

### 7.1 Architecture Ablation (Table 1 in paper)
| Experiment | MSDA | HFF | Attention Pool | Aux Head | Ordinal Head |
|------------|------|-----|----------------|----------|--------------|
| Baseline | ✗ | ✗ | ✗ | ✗ | ✗ |
| +MSDA | ✓ | ✗ | ✗ | ✗ | ✗ |
| +HFF | ✗ | ✓ | ✗ | ✗ | ✗ |
| +MSDA+HFF | ✓ | ✓ | ✗ | ✗ | ✗ |
| +AttnPool | ✓ | ✓ | ✓ | ✗ | ✗ |
| +AuxHead | ✓ | ✓ | ✓ | ✓ | ✗ |
| +Ordinal | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Full (Proposed)** | ✓ | ✓ | ✓ | ✓ | ✓ |

### 7.2 SSL Pretraining Ablation (Table 2 in paper)
| Experiment | Pretraining | Val Acc | Val QWK | Test Acc | Test QWK |
|------------|-------------|---------|---------|----------|----------|
| ImageNet pretrain | ImageNet-22k | | | | |
| EyePACS supervised | EyePACS (labeled) | | | | |
| Contrastive only | SSL (InfoNCE) — Option 1 alone | | | | |
| Multi-task only | SSL (lesion tasks) — Option 4 alone | | | | |
| **Full SSL (Proposed)** | SSL (Contrastive + Multi-task) — Combined | | | | |

### 7.3 Training Strategy Ablation (Table 3 in paper)
| Experiment | EMA | SWA | TTA | Ensemble | Test Acc |
|------------|-----|-----|-----|----------|----------|
| No tricks | ✗ | ✗ | ✗ | ✗ | |
| +EMA | ✓ | ✗ | ✗ | ✗ | |
| +EMA+SWA | ✓ | ✓ | ✗ | ✗ | |
| +TTA | ✓ | ✓ | ✓ | ✗ | |
| **Full (Proposed)** | ✓ | ✓ | ✓ | ✓ | |

---

## PHASE 8: TABLES & CHARTS GENERATION

### 8.1 Tables

**Table 1: Architecture Ablation** (auto-generated as CSV)
```
results/tables/ablation_architecture.csv
```
Columns: Model, Acc, Precision, Recall, F1, QWK, AUC, Params(M), FLOPs(G)

**Table 2: SSL Pretraining Ablation**
```
results/tables/ablation_ssl.csv
```

**Table 3: Training Strategy Ablation**
```
results/tables/ablation_training.csv
```

**Table 4: Comparison with SOTA**
```
results/tables/sota_comparison.csv
```
Columns: Method, Backbone, Year, Acc, QWK, AUC, Dataset

**Table 5: Per-Class Metrics**
```
results/tables/per_class_metrics.csv
```
Columns: Class, Sensitivity, Specificity, F1, Precision, Recall, AUC

**Table 6: External Validation Results**
```
results/tables/external_validation.csv
```

### 8.2 Charts

**Chart 1: Training Curves** (per fold + averaged)
```
results/figures/train_val_loss_curve.png     — Loss vs Epoch
results/figures/train_val_acc_curve.png       — Accuracy vs Epoch
results/figures/train_val_qwk_curve.png       — QWK vs Epoch
results/figures/lr_schedule.png               — Learning rate over epochs
```

**Chart 2: Confusion Matrices**
```
results/figures/cm_internal_test.png          — APTOS-2019 test set
results/figures/cm_external_test.png          — Messidor-2
results/figures/cm_normalized.png             — Normalized CM (percentages)
```

**Chart 3: ROC Curves**
```
results/figures/roc_multiclass.png            — OVR ROC for all 5 classes
results/figures/pr_curve.png                   — Precision-Recall curves
```

**Chart 4: Ablation Bar Charts**
```
results/figures/ablation_arch_bar.png         — Bar chart: accuracy across ablations
results/figures/ablation_ssl_bar.png          — Bar chart: SSL pretraining comparison
results/figures/ablation_training_bar.png     — Bar chart: training tricks comparison
```

**Chart 5: SSL Pretraining Curves**
```
results/figures/ssl_contrastive_loss.png      — Contrastive loss over epochs
results/figures/ssl_multitask_loss.png        — Multi-task loss over epochs
results/figures/ssl_lesion_correlation.png    — Lesion detection vs DR grade correlation
```

**Chart 6: XAI Heatmaps**
```
results/figures/xai_grid_5grades.png          — 5 DR grades × 3 columns (original, EigenCAM, GradCAM)
```

**Chart 7: Per-Class Performance Radar Chart**
```
results/figures/radar_per_class.png           — Radar chart: sensitivity/specificity/F1 per class
```

### 8.3 Metrics Formatting
All metrics reported to **2 decimal places**:
- Accuracy: 95.67%
- QWK: 0.96
- AUC: 0.98
- F1: 0.94

### 8.4 Classification Report
```
results/tables/classification_report.txt      — sklearn classification_report (per fold + averaged)
```

---

## PHASE 9: CODE STRUCTURE (New Files)

```
src/
├── preprocessing/
│   ├── __init__.py
│   ├── lesion_detection.py        # Classical lesion detection (microaneurysms, hemorrhages, exudates)
│   └── preprocess_pipeline.py     # Fundus crop, Ben Graham, CLAHE
│
├── models/
│   ├── ssl_model.py               # SSL pretraining model (contrastive + multi-task)
│   ├── dr_model.py                # RetiNA-Net (already exists, modified to load SSL backbone)
│   └── components.py               # (already exists)
│
├── data/
│   ├── eyepacs_dataset.py          # EyePACS dataset for SSL pretraining
│   └── datamodule.py               # (already exists)
│
├── training/
│   ├── ssl_trainer.py              # SSL pretraining trainer
│   ├── trainer.py                  # (already exists)
│   └── mixup.py                    # (already exists)
│
├── evaluation/
│   ├── visualizer.py               # (already exists, add ablation charts, radar chart)
│   ├── generate_tables.py          # Auto-generate all tables as CSV
│   ├── generate_charts.py          # Auto-generate all charts
│   └── ...                         # (rest already exists)
│
main.py                             # Add --mode pretrain, --mode generate_results
configs/
├── config.yaml                     # (already exists)
├── config_ssl.yaml                 # SSL pretraining config
└── config_ablation.yaml            # Ablation experiment configs
```

---

## PHASE 10: EXECUTION ORDER

```
1. Download & set up EyePACS dataset
2. Run classical lesion detection on 88,702 images
   → phd run ... -- python main.py --mode detect_lesions
3. Run SSL pretraining (50 epochs)
   → phd run ... -- python main.py --mode pretrain
4. Fine-tune RetiNA-Net on APTOS-2019 (5-fold × 80 epochs, all 4 ablations)
   → phd run ... -- python main.py --mode train --ablation proposed
   → phd run ... -- python main.py --mode train --ablation baseline
   → phd run ... -- python main.py --mode train --ablation msda_only
   → phd run ... -- python main.py --mode train --ablation hff_only
5. Test with ensemble + TTA (all ablations)
   → phd run ... -- python main.py --mode test --ablation proposed
   → phd run ... -- python main.py --mode test --ablation baseline
   → ... (all 4 ablations)
6. External validation on Messidor-2
   → phd run ... -- python main.py --mode external_validation --ablation proposed
7. XAI heatmaps
   → phd run ... -- python main.py --mode xai --ablation proposed
8. Generate all tables and charts
   → python main.py --mode generate_results
```

---

## SUMMARY

| Phase | New Files | Time to Implement | Time to Run |
|-------|-----------|-------------------|-------------|
| Lesion detection | 1 | 2 hours | 1 hour |
| SSL pretraining | 3 | 4 hours | 12 hours |
| Fine-tuning | 0 (exists) | 0 | 15 hours × 4 ablations |
| Testing | 0 (exists) | 0 | 2 hours |
| External val | 0 (exists) | 0 | 1 hour |
| XAI | 0 (exists) | 0 | 30 min |
| Tables & charts | 2 | 3 hours | 10 min |
| **Total** | **6 new files** | **~9 hours** | **~80 hours** |

---

## NOVELTY SUMMARY

1. **Architecture:** RetiNA-Net (MSDA + HFF + Attention Pooling + Ordinal Head + Deep Supervision)
2. **Pretraining:** Lesion-Aware Self-Supervised Learning combining:
   - Lesion-Aware Contrastive Learning (Option 1) — learns invariant DR-aware representations
   - Lesion-Guided Multi-Task SSL (Option 4) — learns lesion type, count, and location
3. **Training:** Triple loss (Focal + Ordinal + Auxiliary) with EMA + SWA
4. **Inference:** 5-fold ensemble + multi-scale TTA with ordinal fusion

### Target Venues
| Venue | Tier | Fit |
|-------|------|-----|
| IEEE TMI (Trans. Medical Imaging) | Q1 | ⭐⭐⭐ Perfect fit |
| Medical Image Analysis (MedIA) | Q1 | ⭐⭐⭐ Perfect fit |
| MICCAI | A* | ⭐⭐⭐ Top medical imaging conference |
| IEEE JBHI | Q1 | ⭐⭐ Good fit |
| Nature Machine Intelligence | Q1 | ⭐ Stretch (need exceptional results) |

### Expected Performance
| Method | Expected Accuracy | Expected QWK |
|--------|-------------------|-------------|
| ImageNet pretrain (current) | ~90-93% | ~0.93 |
| EyePACS supervised pretrain | ~93-95% | ~0.95 |
| **Lesion-Aware SSL (Proposed)** | **~95-97%** | **~0.96-0.97** |
