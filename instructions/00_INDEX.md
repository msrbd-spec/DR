# RetiNA-Net Implementation Plan — Index

Target: 97.5% internal (APTOS-19) test accuracy, 90%+ external (Messidor-2)
validation accuracy. Beat DR-LDS (Springer Nature 2026, verified peak 95.60%
val-acc via diffusion-synthetic augmentation, no novel classifier
architecture, no external validation) by ≥1% on an honest held-out test set.

Reality check (keep visible throughout implementation): APTOS-19 inter-rater
agreement between ophthalmologists is ~84.8%. Raw 97.5% 5-class test
accuracy sits above that label-noise ceiling. Push every phase below as far
as legitimately possible; report QWK and adjacent-grade accuracy as
co-primary metrics alongside raw accuracy, not as a fallback excuse.

## Execution order

| Phase | File | What | Cost | Priority |
|---|---|---|---|---|
| P1 | `01_overfitting_fixes.md` | Bug fixes + regularization (backbone freezing, SAM, augmentation) | Low | Do first — everything else depends on a correctly regularized base |
| P2 | `02_class_imbalance.md` | Loss/sampling-level rebalancing (Logit Adjustment, DRW, CB-Loss, LDAM, decoupled re-training) | Low–Medium | Do second — cheapest, highest-certainty accuracy gain |
| P3 | `03_ablation_harness.md` | Extend architecture + SSL ablation systems to match the planned tables | Medium | After P1/P2 land, before running full ablation sweeps |
| P4 | `04_generalization_fda.md` | Fourier Domain Adaptation for external (Messidor-2) robustness | Low | Parallel with P2/P3 — required for the 90%+ external target |
| P5 | `05_repconv_reparam.md` | RepVGG/GCNet-style structural reparameterization (HFF projections) | Low | Secondary — efficiency novelty, not an accuracy lever |
| P6 | `06_diffusion_pipeline.md` | Lesion-guided, classifier-filtered diffusion synthetic augmentation | High | Stretch/Phase-2 — only after P1–P4 show where the remaining gap actually is |
| P7 | `07_results_automation.md` | Auto-populate every results table/chart from saved run outputs | Low | Can be done anytime after P3; needed before final paper assembly |

## Cross-cutting rules for whoever implements this (Claude Code / Antigravity)

1. Never push to GitHub automatically — leave commits to the user.
2. Every new ablation/training variant must get a distinct checkpoint name
   (`checkpoints/best_model_{ablation}_fold{f}.pth` pattern already exists —
   extend the `{ablation}` tag, don't overwrite).
3. Default config values must preserve current behavior when a new flag is
   off (e.g. `use_sam: False`, `freeze_backbone_stages: 0`) — these are
   additive changes, not replacements, until explicitly enabled.
4. Run `test_model.py` after any change to `src/models/` before launching a
   full training run.
5. All line-number anchors below were verified against the uploaded
   `DR-main.zip` repo state — re-verify before editing if the repo has
   since changed.
