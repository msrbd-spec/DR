## Phase 1 — Fix the foundation (highest expected yield, do this first)

**1.1 Verify SSL pretraining is actually wired in.** Still unconfirmed from your logs. Add a checksum/weight-diff assertion at load time. This is your single highest-leverage unknown.

**1.2 Upgrade the pretraining source.** EyePACS-only contrastive+multi-task SSL (88K images) is good, but there are now public retina-domain foundation models trained on much larger corpora:
- **RETFound** (Zhou et al., *Nature* 2023) — ViT pretrained via masked autoencoding on 1.6M retinal images. In a recent calibration benchmark I found, RETFound scored QWK 0.884 on APTOS with just a linear/MLP head — a strong prior with zero task-specific engineering.
- **FLAIR** (Silva-Rodríguez et al., *Medical Image Analysis* 2024) — vision-language pretraining across a large multi-dataset retinal corpus, encodes expert textual knowledge as supervision signal.

Initializing from one of these (or ensembling a RETFound/FLAIR branch with your SwinV2 branch) gives you pretraining signal from ~20x more images than EyePACS alone provides. This is likely worth more than any architectural tweak.

**1.3 Clean the labels, don't just regularize around them.** Run **confident learning** (Northcutt et al., cleanlab) — cross-validated out-of-fold predictions flag likely-mislabeled training images. On a single-grader dataset like APTOS, this reliably surfaces 3-8% probable label errors. Either drop or re-weight these in training. This directly attacks the ceiling problem above rather than fighting it.

**1.4 Soft ordinal targets from real grader-confusion data.** Instead of one-hot labels, use label smoothing shaped by Krause et al.'s empirically measured confusion structure — adjacent grades (e.g., Moderate↔Severe) get more smoothing mass than distant ones (No-DR↔Proliferative), since that mirrors where actual human disagreement concentrates. This is more defensible than flat label smoothing and should reduce your model being punished for "reasonable" errors.

## Phase 2 — Regularization discipline (match, then exceed, the 94.64% baseline)

- Head dropout **0.5/0.3** (yours is 0.1) — directly matches the ResNet-50 paper that's currently ahead of you.
- `drop_path_rate` 0.1 → 0.25–0.3.
- Label smoothing 0.0 → 0.05–0.1 (or the grader-confusion-weighted version above).
- **Sharpness-Aware Minimization** (Foret et al., ICLR 2021) as the optimizer wrapper — SAM explicitly seeks flat minima and has repeatedly shown outsized gains specifically in small-data, large-model regimes like yours.
- Tighter early stopping: patience 25 → ~12 on QWK. Your own logs show best val QWK per fold lands by epoch 24–37; everything after is overfitting the SWA average.
- Progressive backbone unfreezing (freeze stages 1-2 for first 15 epochs) instead of uniform 0.2× LR from epoch 1.

## Phase 3 — Architecture: prove novelty earns its keep, don't assume it

- **Run the ablation you already have flagged** (`ablation: proposed` vs. baseline). If MSDA/HFF/aux-head/ordinal-head don't beat a simpler backbone + Phase 1/2 changes on val QWK, that's real evidence to simplify — a smaller model with less to overfit may outperform here, exactly like the ResNet-50 case.
- If ordinal head stays, swap to a true **CORAL** formulation (Cao et al., 2020) — guarantees rank-monotonic thresholds rather than an approximate ordinal loss, which is a real, citable, defensible upgrade over ad hoc cumulative logits.
- Consider **architecturally diverse ensembling** (SwinV2 + ConvNeXt + EfficientNet-B4, not 5× the same architecture across folds) — diversity across inductive biases typically buys more ensemble gain than fold diversity alone.

## Phase 4 — Inference-time gains (cheap, real, stackable)

- Multi-scale + flip + rotation TTA (already planned) — implement it, it's usually 1-2 QWK points free.
- **QWK-optimized threshold search** on the ordinal/classification blend, fit on validation, applied at test time — routinely worth 1-3 points versus naive argmax, and is standard practice in the literature you're competing against.
- Diverse-architecture ensemble stacking with a small meta-learner instead of fixed 0.7/0.3 weighting.

## What this plan realistically buys you

Stacking all of the above, honestly: closing the gap from your current ~85% toward **~90-93% accuracy, QWK ~0.94-0.96** is a defensible, achievable, publishable target — genuinely competitive with the best legitimate Q1 work I found. That's a real, substantial improvement over where you are now.

## If you need a legitimate "95%+" headline number

There's one honest way to get it: report **±1-grade (adjacent) accuracy** alongside exact accuracy — standard practice in ordinal DR grading papers, since QWK itself is built on the idea that adjacent-grade errors matter far less than distant ones. Models sitting at 85-90% exact accuracy routinely hit **96-99% within-one-grade accuracy**, because nearly all errors are between neighboring severity levels, not wild misses. This is a real, literature-standard metric — not a trick — as long as you clearly label it as adjacent accuracy and report exact accuracy alongside it, not instead of it.

**Bottom line, honestly:** I've given you everything I can find that's real, citable, and stacks. It should meaningfully beat 94.64%. It will not, and should not, hit 95%+ exact-match accuracy on unmodified 5-class APTOS grading — and treating that as the goalpost risks optimizing for label noise instead of diagnostic signal, which is the opposite of what you want in a paper that will get reviewed by people who know this literature.