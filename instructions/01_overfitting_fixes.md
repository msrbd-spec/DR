# P1 — Bug Fixes + Overfitting Reduction

## Objective
Fix two silent bugs, then reduce the train/val gap visible in every logged
run (train acc 90–96% vs val acc plateauing 83–85%, val loss rising after
~epoch 20–30 while train loss keeps falling). Root cause: SwinV2-Large
(~197M params) fully unfrozen and fine-tuned against ~3.3k images/fold,
combined with a deliberately softened augmentation pipeline (see
`src/data/transforms.py` docstring — GridDistortion/OpticalDistortion/
CoarseDropout were removed, RandomResizedCrop range narrowed, to protect
lesion geometry, at the cost of regularization strength).

## Priority / Cost / Risk
Low cost, no downside risk (every change is additive / off-by-default where
it changes training dynamics). Do this first — every later phase assumes a
correctly regularized base model.

---

## P1.0 — Bug fixes

### Fix 1: `val_loader`/`test_loader` dropping samples
**File:** `src/data/datamodule.py`
**Lines:** 104, 109 (val_loader and the K-fold `train_loader`'s sibling
val_loader both have `drop_last=True`; line 114 is the test_loader, also
`drop_last=True`).

Change `drop_last=True` → `drop_last=False` on **lines 104 and 109 only**
(the val loaders). Line 114 (test_loader) should also become `False` for
the same reason — never silently discard evaluation samples on an already
small dataset.

```python
# line 104 (val_loader inside K-fold branch) and line 109 (val_loader
# inside standard-mode branch): change
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False   # was True
    )

# line 114 (test_loader):
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False   # was True
    )
```
Leave `train_loader`'s `drop_last=True` (lines near 95–101) unchanged —
dropping a partial final training batch is standard practice and harmless.

### Fix 2: train/val curve colors
**Files:** `src/evaluation/visualizer.py` (lines 10–11, 20–21),
`src/evaluation/generate_charts.py` (lines 50/52, 67/69, COLORS list line 33)

Rule: **train = blue, val = red**, consistently, everywhere a train/val pair
is plotted.

`src/evaluation/visualizer.py`:
```python
# line 10-11, inside plot_training_curves — change:
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Val Loss', color='red')

# line 20-21, accuracy subplot — change:
    plt.plot(train_acc, label='Train Acc', color='blue')
    plt.plot(val_acc, label='Val Acc', color='red')
```

`src/evaluation/generate_charts.py`:
```python
# line 33 — COLORS list stays as-is for the 5-class palette (used
# elsewhere for ROC/PR/radar per-class plots, unrelated to train/val).
# Instead, in plot_training_curves (around lines 50-52 and 67-69),
# hardcode train/val colors directly instead of indexing COLORS:
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)   # was COLORS[0]
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='red', linewidth=2)        # was COLORS[3]
    ...
    ax.plot(epochs, [a * 100 for a in history['train_acc']], label='Train Acc', color='blue', linewidth=2)  # was COLORS[0]
    ax.plot(epochs, [a * 100 for a in history['val_acc']], label='Val Acc', color='red', linewidth=2)       # was COLORS[1]
```

---

## P1.1 — Partial backbone freezing

Freeze the patch embedding + earliest SwinV2 stages; keep the deepest
stage(s) and all custom heads (MSDA/HFF/classification/aux/ordinal)
trainable. Config-driven, default off (0 = current behavior, fully
unfrozen).

**File:** `configs/config.yaml` — insert after line 8 (`dropout: 0.2 ...`),
before line 10 (`num_classes: 5`):
```yaml
freeze_backbone_stages: 2   # P1 fix: freeze SwinV2 stages 0-1 (of 0-3) + patch_embed.
                             # 0 = fully unfrozen (old behavior). Cuts trainable
                             # backbone capacity on a ~197M-param model vs ~3.3k
                             # train images/fold.
```

**File:** `src/models/dr_model.py`
**Constructor signature, line 36** — add a new parameter:
```python
    def __init__(self, use_msda: bool = True, use_hff: bool = True,
                 num_classes: int = 5, drop_path_rate: float = 0.1,
                 dropout: float = 0.1, use_ordinal: bool = True,
                 use_aux_head: bool = True, use_attention_pool: bool = True,
                 backbone_name: str = 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
                 stage_channels=(192, 384, 768, 1536),
                 ssl_pretrained_path: str = None,
                 freeze_backbone_stages: int = 0):   # <-- NEW
```

**Insert between line 82's block end and line 86** (i.e. right after the
`elif ssl_pretrained_path is not None:` warning block, before
`if self.use_msda:`):
```python
        # P1 fix: partial backbone freezing — SwinV2-Large (~197M params) is
        # heavily overparameterized relative to APTOS-19's ~3.3k train images
        # per fold; freezing earlier (more generic) stages cuts trainable
        # backbone capacity and reduces the train/val gap.
        self.freeze_backbone_stages = freeze_backbone_stages
        if freeze_backbone_stages > 0:
            self._freeze_backbone_stages(freeze_backbone_stages)
```

**Add new method** anywhere in the class body (e.g. right after `__init__`,
before `forward`):
```python
    def _freeze_backbone_stages(self, num_stages: int):
        """
        Freeze the patch embedding and the first `num_stages` SwinV2 stages
        (0-indexed: layers.0 .. layers.3). Later stages and all custom
        heads stay trainable. Confirmed module path 'backbone.layers.{i}'
        from src/evaluation/xai.py's get_target_layer().
        """
        num_stages = min(num_stages, len(self.backbone.layers))

        if hasattr(self.backbone, 'patch_embed'):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad_(False)

        for i in range(num_stages):
            for p in self.backbone.layers[i].parameters():
                p.requires_grad_(False)

        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"Backbone freeze: stages 0-{num_stages-1} frozen "
              f"({frozen:,}/{total:,} backbone params, {100*frozen/total:.1f}%)")
```

**File:** `main.py`, function `create_model` (starts line 72) — after line
85 (`ssl_pretrained_path = config.get(...)`), add:
```python
    freeze_backbone_stages = config.get("freeze_backbone_stages", 0)
```
Then in the `RetiNA_Net(...)` call (lines 87+), add the kwarg:
```python
        ssl_pretrained_path=ssl_pretrained_path,
        freeze_backbone_stages=freeze_backbone_stages   # <-- NEW
    ).to(device)
```

**File:** `src/training/trainer.py`, `__init__` (starts line 148) — line
165 currently builds `backbone_params`/`head_params` from ALL named
parameters regardless of `requires_grad`. Change to filter frozen params
out of the optimizer entirely:
```python
# line 165-166 — change:
        backbone_params = [p for n, p in self.model.named_parameters() if n.startswith("backbone") and p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith("backbone") and p.requires_grad]
```

---

## P1.2 — Stronger Mixup/CutMix + bounded CoarseDropout

**File:** `configs/config.yaml`, line 62:
```yaml
mix_prob: 0.4                # P1 fix: increased from 0.2 — stronger regularization
```
(`cutmix_prob: 0.5` on line 63 stays unchanged.)

**File:** `src/data/transforms.py`
Update the module docstring (lines ~9-13) to stop claiming CoarseDropout is
fully removed:
```python
    Key changes from previous pipeline:
      - Removed: GridDistortion, OpticalDistortion (destroy fine lesion geometry)
      - Reduced: ColorJitter hue 0.1→0.02, RandomResizedCrop scale (0.8,1.0)→(0.9,1.0)
      - Kept: RandomRotate90, flips, CLAHE, Sharpen, RandomBrightnessContrast, Affine
      - P1 fix: small-scale CoarseDropout re-added (holes ≤16px, well under
        lesion scale) — counteracts the overfitting the full removal caused
```

Same edit inside `get_train_transforms` body (around line 35, the inline
comment `# Removed: GridDistortion, OpticalDistortion, CoarseDropout`):
replace that comment line and insert a new augmentation step right after
the `A.Affine(...)` call and before `A.Normalize(...)`:
```python
        # Affine — mild geometric augmentation (kept, was already reasonable)
        A.Affine(translate_percent=0.1, scale=0.9, rotate=45, p=0.3),
        # P1 fix: small-scale CoarseDropout re-added — holes capped at 16px,
        # well below microaneurysm/hemorrhage scale, so lesion geometry is
        # preserved while still providing a regularizing cutout signal.
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            fill=0,
            p=0.15
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
```
**Note:** `num_holes_range`/`hole_height_range`/`hole_width_range`/`fill`
match the albumentations ≥1.4 API. Verify the pinned albumentations
version in the environment; if it's an older release using
`max_holes`/`min_holes`/`max_height`/`max_width`/`fill_value`, translate
param names accordingly (same semantics, 1-3 holes of 8-16px, p=0.15).

Re-test `label_smoothing: 0.05` (config.yaml, currently 0.0) as a follow-up
ablation once the above is stable — the existing code comment claiming
"focal loss already provides soft targets" is not technically accurate
(focal loss reweights by difficulty, it does not smooth target
distributions); don't assume this is redundant without testing it.

---

## P1.3 — SAM (Sharpness-Aware Minimization) optimizer

### New file: `src/training/sam.py`
```python
"""
Sharpness-Aware Minimization (SAM) optimizer wrapper.

Wraps a base optimizer (e.g. AdamW) and performs a two-step update per
batch: an ascent step that perturbs parameters toward the worst-case
point within an L2 ball of radius `rho` (the "sharpness" direction),
followed by a descent step of the base optimizer computed at that
perturbed point. This biases training toward flatter minima, which
generalizes better on small, imbalanced datasets like ours.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization", ICLR 2021.

Usage (see trainer.py): requires TWO forward+backward passes per
optimizer step, so when `use_sam=True` gradient accumulation is
bypassed (one SAM step == one full-batch step).
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    SAM optimizer wrapper around any base torch.optim.Optimizer class.

    Args:
        params: model parameters or param groups (as passed to any optimizer)
        base_optimizer: an UNINSTANTIATED optimizer class, e.g. torch.optim.AdamW
        rho: neighborhood size for the sharpness-aware ascent step
        **base_optimizer_kwargs: forwarded to base_optimizer(...)
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, **base_optimizer_kwargs):
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, **base_optimizer_kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Ascent step: perturb params toward the local worst-case direction."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                perturbation = p.grad * scale.to(p.device)
                p.add_(perturbation)
                self.state[p]["perturbation"] = perturbation

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Descent step: undo the perturbation, then apply the base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "perturbation" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["perturbation"])
                del self.state[p]["perturbation"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)

    def step(self, closure=None):
        raise RuntimeError(
            "SAM requires two explicit passes — call first_step() after the "
            "first backward(), then second_step() after the second backward(). "
            "Do not call step() directly."
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
```

### `configs/config.yaml` — add near the training hyperparameters (after
`grad_clip_norm: 1.0`):
```yaml
# SAM (Sharpness-Aware Minimization) — optional, P1 overfitting fix
use_sam: False                # set True to enable; bypasses grad accumulation (see trainer.py)
sam_rho: 0.05                 # neighborhood radius for the ascent step
```

### `src/training/trainer.py` changes

**Import (near top, after `from .mixup import ...`):**
```python
from .sam import SAM
```

**`__init__`, replace lines 165-172** (the backbone_params/head_params
block through the end of `self.optimizer = AdamW([...])`):
```python
        backbone_params = [p for n, p in self.model.named_parameters() if n.startswith("backbone") and p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith("backbone") and p.requires_grad]

        self.use_sam = config.get("use_sam", False)
        self.sam_rho = config.get("sam_rho", 0.05)
        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": head_weight_decay},
        ]
        if self.use_sam:
            self.optimizer = SAM(param_groups, AdamW, rho=self.sam_rho)
            logger.info(f"Using SAM optimizer (rho={self.sam_rho}); gradient accumulation is bypassed while SAM is enabled.")
        else:
            self.optimizer = AdamW(param_groups)
```

**`train_epoch` (starts line 243) — full method replacement:**
```python
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        self.optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
            file=sys.stdout,
            mininterval=30,
            ncols=100
        )):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Apply Mixup/CutMix (same mixed batch reused across both SAM passes)
            if self.use_mixup:
                mixed_inputs, y_a, y_b, lam = apply_mixup_or_cutmix(
                    inputs, targets,
                    mixup_alpha=self.mixup_alpha,
                    cutmix_alpha=self.cutmix_alpha,
                    mix_prob=self.mix_prob,
                    cutmix_prob=self.cutmix_prob
                )
            else:
                mixed_inputs, y_a, y_b, lam = inputs, targets, targets, 1.0

            def _compute_loss():
                outputs = self.model(mixed_inputs)
                if self.use_mixup and lam < 1.0:
                    batch_loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                else:
                    batch_loss = self.criterion(outputs, targets)
                return outputs, batch_loss

            if self.use_sam:
                # --- SAM two-pass update (bypasses grad accumulation) ---
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs, loss = _compute_loss()
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, loss2 = _compute_loss()
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.second_step(zero_grad=True)

                if self.scheduler_step_per_batch:
                    self.scheduler.step()
                if self.use_ema:
                    self.ema.update(self.model)

                running_loss += loss.item() * inputs.size(0)
            else:
                # --- Standard AMP + gradient-accumulation update (unchanged) ---
                with torch.amp.autocast('cuda'):
                    outputs, loss = _compute_loss()
                    loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()

                if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    new_scale = self.scaler.get_scale()

                    self.optimizer.zero_grad()

                    if new_scale >= old_scale and self.scheduler_step_per_batch:
                        self.scheduler.step()

                    if self.use_ema:
                        self.ema.update(self.model)

                running_loss += loss.item() * self.accumulation_steps * inputs.size(0)

            # For metrics, use original (unmixed) targets and classification logits
            with torch.no_grad():
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        train_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_metrics(all_targets, all_preds)
        return train_loss, metrics
```

Note: `self.scaler = torch.amp.GradScaler('cuda')` on line 212 stays — it's
simply unused when `use_sam=True` (the SAM branch uses bf16 autocast with
no GradScaler, since bf16 doesn't need loss scaling).

---

## Validation steps
1. `python test_model.py` — confirm architecture still instantiates for all
   4 ablations with `freeze_backbone_stages` left at config default.
2. Run one short fold (few epochs) with `use_sam: False,
   freeze_backbone_stages: 0` — confirm identical behavior to pre-P1 logs
   (regression check).
3. Run one short fold with `freeze_backbone_stages: 2` only — confirm
   reduced trainable-param count printout, confirm training still
   converges.
4. Run one short fold with `use_sam: True` — confirm no crash across both
   SAM passes, confirm `second_step()` actually updates weights (loss
   should decrease over a few epochs).
5. Full fold-0 run with all P1 changes enabled — compare train/val gap and
   val QWK against the original `logs/train_baseline_20260909_122449.log`
   and `logs/train_proposed_20260910_145854.log`.

## Expected outcome
Narrower train/val gap, more stable val loss across epochs, and a higher
or at-least-equal best val QWK versus the current baseline/proposed logs.
This is the foundation every later phase (P2–P7) builds on.
