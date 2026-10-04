# RetiNA-Net Implementation Checklist

## P1 — Bug Fixes + Overfitting Reduction
- [x] P1.0: Fix `val_loader`/`test_loader` dropping samples & train/val curve colors
- [x] P1.1: Partial backbone freezing
- [x] P1.2: Stronger Mixup/CutMix + bounded CoarseDropout
- [x] P1.3: SAM (Sharpness-Aware Minimization) optimizer

## P2 — Class-Imbalance Rebalancing (Loss + Sampling Level)
- [x] P2.1: Logit Adjustment
- [x] P2.2: DRW (Deferred Re-Weighting)
- [x] P2.3: Class-Balanced Loss
- [x] P2.4: LDAM (Label-Distribution-Aware Margin)
- [x] P2.5: Decoupled Classifier Re-training

## P3 — Ablation Harness Extension
- [x] P3.A: Architecture ablation (full 5-flag combo)
- [x] P3.B: SSL ablation: missing `EyePACS supervised` row

## P4 — Fourier Domain Adaptation (External Generalization)
- [x] P4: FDA core, style pool, and dataloader integration

## P5 — Structural Reparameterization for HFF Projections
- [x] P5: RepConv-style `RepProjConv`, `HFFBlock` updates, and fusion method

## P6 — Lesion-Guided, Classifier-Filtered Diffusion Synthetic Augmentation
- [x] P6.0: DR-LDS replication
- [x] P6.1: Lesion-mask spatial conditioning
- [x] P6.2: Classifier-in-the-loop filtering
- [x] P6.3: Ordinal-aware conditioning
- [x] P6.4: Active error-driven synthetic sampling loop
- [x] P6.5: Spectral fidelity loss
- [x] P6.6: Lesion-weighted VAE reconstruction loss

## P7 — Final Results-Generation Automation
- [x] P7: `results_logger.py`, wiring into `main.py`, and `generate_tables.py`
