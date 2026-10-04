# P6 — Lesion-Guided, Classifier-Filtered Diffusion Synthetic Augmentation

## Objective
Stretch/Phase-2 item. Only start this after P1–P4 are implemented, run,
and measured — if the remaining gap to 97.5%/90% is closing from
regularization + rebalancing + generalization alone, this phase may not be
needed at all. If a real gap remains (not just label-noise ceiling), this
is where the biggest additional lever is. Not a DR-LDS clone — six honestly
distinguished novelty pieces on top of a fair baseline replication of their
method (so we can prove we match them before claiming to beat them).

## Priority / Cost / Risk
**High cost — realistically weeks, not days.** Core 3 items are the
must-have minimum if this phase is attempted at all; the other 4 are
explicitly "if time permits."

---

## Directory structure (new top-level package, mirrors DR-LDS's 4-folder
structure conceptually, adapted to our codebase conventions)
```
src/generation/
  __init__.py
  vae.py              # KL-regularized VAE (encoder/decoder), lesion-weighted loss option (item 6)
  unet.py             # conditional U-Net: AdaGN time+class conditioning, bottleneck
                       # cross-attention, lesion-mask spatial conditioning (item 1),
                       # ordinal-aware conditioning option (item 3), optional
                       # RepConv-reparameterized ResBlocks (reuses P5's RepProjConv design)
  diffusion.py         # forward process (q_sample), DDIM reverse sampling
  spectral_loss.py     # FFT-domain auxiliary loss (item 5)
  dataset.py            # APTOS training-split images + LesionDetector masks on the fly
  train_vae.py
  train_unet.py
  generate.py           # sampling + classifier-in-the-loop filtering (item 2)
  active_loop.py        # closed-loop error-driven sampling (item 4)
  leakage_check.py       # SSIM non-memorization check vs. held-out val/test
configs/config_diffusion.yaml
```

---

## Fair baseline first: DR-LDS replication
Before any of our 6 novelty pieces, implement the **plain** version: VAE
(kl-f8 style, fine-tuned on APTOS train split only — 3,296 images, matching
their protocol) + conditional U-Net with flat `nn.Embedding(5, 512)` class
conditioning only, AdaGN injection, bottleneck cross-attention at 16×16.
Train to convergence, report FID/KID against their published numbers
(target: come close to their FID 8.05 — this is the "Row 1: DR-LDS-style
replication" ablation row, and proves good-faith fair comparison before
claiming improvement).

---

## Core novelty (must-have if this phase runs at all)

### Item 1 — Lesion-mask spatial conditioning
**Mechanism (scoped honestly as channel-concatenation conditioning, not a
full duplicate-encoder ControlNet branch — cheaper, still spatial, still
genuinely new vs. DR-LDS's class-only conditioning):**

In `src/generation/unet.py`, concatenate the (downsampled-to-latent-
resolution) `LesionDetector` mask (3 channels: microaneurysm/hemorrhage/
exudate) as extra input channels to the noisy latent at the U-Net's first
conv:
```python
# Instead of: self.input_conv = nn.Conv2d(4, base_ch, 3, padding=1)  # 4 = latent channels
self.input_conv = nn.Conv2d(4 + 3, base_ch, 3, padding=1)  # +3 for the lesion mask
```
`src/generation/dataset.py` produces, per training sample: the VAE-encoded
latent of the real APTOS image, its DR grade, and its `LesionDetector`
mask (already computed by the existing `src/preprocessing/lesion_detection.py`
— reuse `LesionDetector.detect_all()` directly, resized via
`cv2.resize(..., interpolation=cv2.INTER_NEAREST)` to the latent spatial
resolution (64×64 at 512px input / 8× VAE compression)).

For **generation** (not reconstruction), the lesion mask must be
*specified*, not extracted from a real image — sample a mask template from
real lesion masks of the SAME target class (nearest-neighbor retrieval
from the training set's precomputed masks, or simple augmentation
—jitter/flip/shift— of a retrieved template) rather than inventing one from
scratch. Document this retrieval step clearly as part of the generation
pipeline (`generate.py`).

### Item 2 — Classifier-in-the-loop filtering
In `src/generation/generate.py`, after sampling a batch of synthetic
images: run them through the trained RetiNA-Net K-fold ensemble (reuse
`src/evaluation/ensemble.py`'s `EnsembleModel`/`ensemble_predict`
directly). Keep a synthetic sample only if:
1. The ensemble's predicted class matches the conditioning target class, AND
2. Ensemble confidence (max softmax prob) exceeds a threshold
   (`classifier_filter_min_confidence`, e.g. 0.6), AND
3. (optional, pairs with item 4) ensemble *disagreement* (variance across
   fold models' predictions) is below a ceiling — rejects synthetic images
   the current model ensemble is wildly inconsistent about, which likely
   indicates an anatomically implausible sample rather than a genuinely
   hard/informative one.
```python
def filter_synthetic_batch(images, target_class, ensemble, device,
                            min_confidence=0.6, max_disagreement=0.3):
    probs = ensemble.predict(images.to(device))  # (B, 5)
    preds = probs.argmax(dim=1)
    confidence = probs.max(dim=1).values
    keep = (preds == target_class) & (confidence >= min_confidence)
    return images[keep], probs[keep]
```

---

## Extended novelty (if runway remains after core 3 + loss-rebalancing + generalization)

### Item 3 — Ordinal-aware conditioning
Replace the flat `nn.Embedding(5, 512)` class lookup with a monotonic
severity embedding: a 1-D learnable scalar-to-vector mapping (e.g. a small
MLP on the normalized grade `y/4 ∈ [0,1]`, or an embedding table with an
explicit smoothness regularizer penalizing large differences between
adjacent-grade embeddings). Ties narratively to the existing
`OrdinalRegressionHead` design philosophy. Enables sampling at interpolated
severities, useful for smoothing the Mild↔Moderate / Moderate↔Severe
decision boundary where confusion matrices show the most errors.

### Item 4 — Active error-driven synthetic sampling loop
`src/generation/active_loop.py`: after each RetiNA-Net training round,
compute K-fold ensemble disagreement per training/val sample (variance of
per-fold predicted probabilities). Identify the **lesion-count/type
profile** (from `LesionDetector`) of the highest-disagreement samples,
condition the diffusion generator to produce more synthetic samples
matching that profile (via item 1's lesion-mask retrieval, biased toward
templates matching the identified profile), regenerate, retrain, repeat.
This is the single highest-payoff item here — closed-loop, driven by the
model's actual current weaknesses rather than a static per-class quota —
but also the most expensive (multiple full train→generate→retrain cycles).

### Item 5 — Spectral fidelity loss
`src/generation/spectral_loss.py`:
```python
def spectral_loss(pred_noise, target_noise, high_freq_weight=2.0):
    """FFT-domain auxiliary loss — penalizes high-frequency-band error
    more heavily than the standard pixel/latent-space epsilon-MSE, since
    diffusion models are known to underfit high-frequency detail (exactly
    where microaneurysms live)."""
    pred_fft = torch.fft.fft2(pred_noise, dim=(-2, -1))
    target_fft = torch.fft.fft2(target_noise, dim=(-2, -1))
    diff = torch.abs(pred_fft - target_fft) ** 2

    h, w = pred_noise.shape[-2:]
    cy, cx = h // 2, w // 2
    freq_weight = torch.ones_like(diff)
    low_freq_radius = min(h, w) // 8
    yy, xx = torch.meshgrid(torch.arange(h) - cy, torch.arange(w) - cx, indexing='ij')
    high_freq_mask = (yy**2 + xx**2) > low_freq_radius**2
    freq_weight[..., high_freq_mask] = high_freq_weight

    return (diff * freq_weight.to(diff.device)).mean()
```
Add to the standard ε-prediction MSE in `train_unet.py`'s loss with a
small weight (e.g. `total_loss = mse_loss + 0.1 * spectral_loss(...)`).
Reuses the same FFT toolkit as P4's FDA — a deliberate cross-task
consistency choice worth stating in the paper.

### Item 6 — Lesion-weighted VAE reconstruction loss
In `src/generation/vae.py`'s training loss, weight the L1/LPIPS
reconstruction term by the `LesionDetector` mask (upsampled to pixel
resolution), so lesion-containing pixels are penalized more than
background retina during VAE fine-tuning — directly addresses DR-LDS's own
admitted "washing out" limitation using our own detector, not a generic
re-tune:
```python
def lesion_weighted_recon_loss(recon, target, lesion_mask, l1_weight=1.0, lesion_boost=2.0):
    pixel_weight = 1.0 + lesion_boost * lesion_mask.amax(dim=1, keepdim=True).clamp(0, 1)
    l1 = (recon - target).abs() * pixel_weight
    return l1_weight * l1.mean()
```

---

## `configs/config_diffusion.yaml` (new file)
```yaml
# VAE
vae_backbone: "kl-f8"          # or whichever base checkpoint is used
vae_epochs: 50
vae_use_lesion_weighted_loss: False   # item 6

# U-Net
unet_base_channels: 128
unet_channel_mult: [1, 2, 4, 4]
unet_use_lesion_conditioning: False   # item 1
unet_use_ordinal_conditioning: False  # item 3
unet_use_rep_blocks: False            # RepConv-reparam ResBlocks, reuses P5 design
unet_epochs: 150

# Loss
use_spectral_loss: False              # item 5
spectral_loss_weight: 0.1

# Sampling / filtering
ddim_steps: 50
guidance_scale: 3.0
classifier_filter_min_confidence: 0.6
classifier_filter_max_disagreement: 0.3

# Active loop (item 4)
use_active_loop: False
active_loop_rounds: 3

# Target
target_per_class: 2000   # matches DR-LDS's own target, for fair comparison
```

## `main.py` — new modes
`train_vae`, `train_diffusion_unet`, `generate_synthetic`,
`generation_ablation` (runs the 8-row table below end-to-end). Each
follows the same `--config`/`--ablation`-style CLI pattern already used
elsewhere.

---

## Ablation table (own table, separate from architecture/SSL ablation)
| Row | Configuration |
|---|---|
| 0 | No synthetic augmentation (current pipeline) |
| 1 | DR-LDS-style replication (flat class-conditioning only) |
| 2 | + Lesion-mask spatial conditioning (item 1) |
| 3 | + Classifier-in-the-loop filtering (item 2) |
| 4 | + Lesion-weighted VAE loss (item 6) |
| 5 | + Ordinal-aware conditioning (item 3) |
| 6 | + Spectral fidelity loss (item 5) |
| 7 | **Full (+ Active error-driven loop, item 4)** |

Metrics per row: FID/KID, per-class F1/recall (Severe/Proliferative
especially), SSIM-leakage check, generation time/step.

## Leakage / validation checklist (match DR-LDS's own rigor)
1. VAE and U-Net trained **only** on the APTOS train split — never val/test.
2. SSIM leakage check (`src/generation/leakage_check.py`): pairwise SSIM
   between every generated image and the held-out val/test set; flag any
   max SSIM ≥ 0.95 as a likely memorization/duplication (same threshold
   DR-LDS uses).
3. Report synthetic-to-real ratio per class in the final augmented
   training set, same transparency DR-LDS provides in their Table 9.

## Expected outcome
If pursued: closes remaining Severe/Proliferative gap with a defensible,
multi-part novelty story; own FID/KID numbers competitive with or better
than DR-LDS's 8.05, backed by an honest replication baseline proving the
comparison is fair.
