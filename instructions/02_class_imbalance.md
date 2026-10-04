# P2 — Class-Imbalance Rebalancing (Loss + Sampling Level)

## Objective
Current `CombinedLoss` only does inverse-frequency class weights (via
`compute_class_weight('balanced', ...)`, `src/data/datamodule.py` line 148)
+ focal loss + ordinal loss. This is the weakest form of rebalancing in the
literature. Add, in priority order: Logit Adjustment → DRW schedule →
Class-Balanced Loss → LDAM → Decoupled classifier re-training. Target:
close the Severe/Proliferative gap visible in every confusion matrix
(Severe F1 0.37–0.43 across all current ablations).

## Priority / Cost / Risk
All five items are low-to-moderate cost, well-cited, proven techniques.
Implement in the order below — each is independently toggleable via config
so they can be ablated separately (feeds P3's architecture-adjacent
ablation work, though these are training-strategy ablations, not
architecture ablations).

---

## Item 1 — Logit Adjustment (Menon et al., ICLR 2021)

**Formula:** shift each class's logit by `τ · log(π_y)` before the
softmax/cross-entropy, where `π_y` is the class's prior probability
(training-set frequency) and `τ` is a temperature (default 1.0).
```
adjusted_logits[:, y] = logits[:, y] - τ · log(π_y)
```
(Subtracting `τ·log(π_y)` for small `π_y` — i.e. rare classes — makes
`log(π_y)` very negative, so the subtraction *adds* a positive boost to
rare-class logits, pushing the decision boundary to favor them.)

### `src/data/datamodule.py`
Line 148 currently computes only `compute_class_weight('balanced', ...)`.
Add, right after line 149 (`class_weights_tensor = torch.FloatTensor(weights)`),
before the `return` on line 151:
```python
    class_counts_for_prior = np.bincount(y_train, minlength=len(classes))
    class_priors = class_counts_for_prior / class_counts_for_prior.sum()
    class_priors_tensor = torch.FloatTensor(class_priors)
```
Change the function's `return` (line 151) to also return `class_priors_tensor`,
and update `get_dataloaders`'s docstring/signature accordingly. Every call
site (`main.py`'s `train_single_fold`, `run_test`, `run_external_validation`)
unpacks `train_loader, val_loader, test_loader, ext_loader, class_weights`
— update all of these to unpack the 6th return value too.

### `src/training/loss.py` — `CombinedLoss`
**`__init__` (line 77-96):** add `class_priors=None, use_logit_adjustment=False,
logit_adjustment_tau=1.0` to the signature; store:
```python
        self.use_logit_adjustment = use_logit_adjustment
        self.logit_adjustment_tau = logit_adjustment_tau
        if use_logit_adjustment and class_priors is not None:
            log_priors = torch.log(class_priors.clamp(min=1e-12)).to(device)
            self.register_buffer_fallback = log_priors  # see note below
            self.log_priors = log_priors
        else:
            self.log_priors = None
```
(`CombinedLoss` is an `nn.Module`; prefer `self.register_buffer('log_priors', log_priors)`
inside an `if` guard so it moves with `.to(device)` calls automatically —
register a zero-tensor buffer when disabled and overwrite it when enabled.)

**`forward` (line 98-133):** right after line 110
(`logits = pred['logits']`) and its `else` counterpart, add the adjustment
before computing `cls_loss`:
```python
        if self.use_logit_adjustment and self.log_priors is not None:
            logits = logits - self.logit_adjustment_tau * self.log_priors.unsqueeze(0)
```
Insert this line after the `if isinstance(pred, dict): ... else: ...` block
(after line 116) and before line 118's `cls_loss = self.focal_loss(...)`
call — this way the adjustment also flows into the aux-head loss at line
125 if desired (optional: apply only to the main head, your call — simplest
is to apply to both since `logits`/`aux_logits` share the same class prior).

### `configs/config.yaml`
```yaml
use_logit_adjustment: False
logit_adjustment_tau: 1.0
```

### `main.py` `create_criterion` (line 104-125)
Thread the new config values and `class_priors` tensor through:
```python
    use_logit_adjustment = config.get("use_logit_adjustment", False)
    logit_adjustment_tau = config.get("logit_adjustment_tau", 1.0)
    ...
    criterion = CombinedLoss(
        ...,
        class_priors=class_priors,              # new arg, passed in from caller
        use_logit_adjustment=use_logit_adjustment,
        logit_adjustment_tau=logit_adjustment_tau
    )
```
`create_criterion`'s signature needs a new `class_priors` parameter;
update its 2 call sites (`train_single_fold`, and wherever test-time
criterion is built, if any) to pass the new 6th return value from
`get_dataloaders`.

---

## Item 2 — DRW (Deferred Re-Weighting)

**Design (Cao et al., NeurIPS 2019 — "LDAM-DRW"):** train the first
`drw_start_frac` fraction of epochs with **no class reweighting** (uniform
weights), then switch to the class-balanced weights (from Item 3 below, or
the existing inverse-frequency weights if Item 3 isn't enabled yet) for the
remainder. Avoids early noisy gradients on 17–33-sample classes dominating
updates before the model has learned basic features.

### `src/training/loss.py` — `CombinedLoss`
Add a method to toggle which weight vector is active:
```python
    def set_reweighting(self, enabled: bool):
        """DRW hook: call from the trainer at the epoch boundary. When
        disabled, the focal loss uses uniform (None) class weights."""
        self._reweighting_enabled = enabled

    def _active_class_weights(self):
        return self.class_weights if getattr(self, '_reweighting_enabled', True) else None
```
In `forward`, line 119, change:
```python
        cls_loss = self.focal_loss(logits, targets, weights=self._active_class_weights())
```
(and the aux-head call at line 125 similarly). Default `_reweighting_enabled = True`
in `__init__` so behavior is unchanged unless the trainer explicitly calls
`set_reweighting(False)`.

### `configs/config.yaml`
```yaml
use_drw: False
drw_start_frac: 0.6   # switch to class-balanced weighting after 60% of epochs
```

### `src/training/trainer.py` — `train()` (line 369+)
At the top of `__init__`, read:
```python
        self.use_drw = config.get("use_drw", False)
        self.drw_start_epoch = int(config.get("drw_start_frac", 0.6) * self.epochs)
```
Inside the `for epoch in range(self.epochs):` loop (line 377), **before**
calling `self.train_epoch(epoch)`, add:
```python
            if self.use_drw:
                self.criterion.set_reweighting(epoch >= self.drw_start_epoch)
```

---

## Item 3 — Class-Balanced Loss (effective number of samples, Cui et al., CVPR 2019)

**Formula:**
```
effective_num(y) = (1 - β^{n_y}) / (1 - β)
weight(y) = 1 / effective_num(y)
weight = weight * num_classes / sum(weight)     # normalize
```
with `β ∈ {0.9, 0.99, 0.999, 0.9999}` (use `0.9999` given class counts as
low as 17–33).

### `src/data/datamodule.py`
Replace the plain inverse-frequency computation at line 148 — add a new
helper and a config-gated branch:
```python
def compute_cb_weights(y_train, beta=0.9999):
    classes = np.unique(y_train)
    counts = np.bincount(y_train, minlength=len(classes))
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.sum() * len(classes)
    return weights
```
In `get_dataloaders`, replace line 148:
```python
    class_weight_strategy = config.get("class_weight_strategy", "inverse")  # "inverse" | "effective_num"
    if class_weight_strategy == "effective_num":
        cb_beta = config.get("cb_beta", 0.9999)
        weights = compute_cb_weights(y_train, beta=cb_beta)
    else:
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.FloatTensor(weights)
```

### `configs/config.yaml`
```yaml
class_weight_strategy: "inverse"   # "inverse" | "effective_num"
cb_beta: 0.9999
```

---

## Item 4 — LDAM (Label-Distribution-Aware Margin, Cao et al., NeurIPS 2019)

**Formula:** per-class margin `Δ_y = C / n_y^{1/4}`, rescaled so
`max(Δ_y) = max_margin` (commonly 0.5). At the true-class logit only,
subtract the margin before scaling and computing cross-entropy:
```
adjusted_logit[y_true] = logit[y_true] - Δ_{y_true}
loss = CrossEntropy(scale · adjusted_logits, y_true)
```
with `scale` commonly 30 (keeps gradients well-scaled after the margin
subtraction).

### `src/training/loss.py` — new class, add after `FocalLoss` (after line 30)
```python
class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin loss (Cao et al., NeurIPS 2019).
    Enforces a larger decision margin for minority classes. Intended to be
    used in place of FocalLoss (select via config `loss_type: "ldam"`),
    typically paired with the DRW schedule (Item 2).
    """

    def __init__(self, cls_num_list, max_margin=0.5, scale=30.0, label_smoothing=0.0):
        super().__init__()
        cls_num = torch.tensor(cls_num_list, dtype=torch.float32)
        margins = 1.0 / torch.sqrt(torch.sqrt(cls_num))
        margins = margins * (max_margin / margins.max())
        self.register_buffer('margins', margins)
        self.scale = scale
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, weights=None):
        index = torch.zeros_like(inputs, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        batch_margins = self.margins[targets].unsqueeze(1).to(inputs.device)
        adjusted = inputs - index.float() * batch_margins
        return F.cross_entropy(
            self.scale * adjusted, targets, weight=weights,
            label_smoothing=self.label_smoothing
        )
```

### `src/training/loss.py` — `CombinedLoss.__init__` (line 77-96)
Add `loss_type="focal"`, `cls_num_list=None`, `ldam_max_margin=0.5`,
`ldam_scale=30.0` params. Replace line 87
(`self.focal_loss = FocalLoss(...)`) with:
```python
        self.loss_type = loss_type
        if loss_type == "ldam":
            assert cls_num_list is not None, "LDAM requires cls_num_list (per-class train counts)"
            self.focal_loss = LDAMLoss(cls_num_list, max_margin=ldam_max_margin,
                                        scale=ldam_scale, label_smoothing=label_smoothing)
        else:
            self.focal_loss = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
```
(The attribute stays named `self.focal_loss` so `forward()` — lines 119,
125 — needs no further changes; it's just whichever loss object was
constructed.)

### `configs/config.yaml`
```yaml
loss_type: "focal"      # "focal" | "ldam"
ldam_max_margin: 0.5
ldam_scale: 30.0
```

### `main.py` `create_criterion`
Needs `cls_num_list` — the raw per-class training counts (`np.bincount(y_train, ...)`).
Return this as a 7th value from `get_dataloaders` (alongside `class_priors`
from Item 1), thread through the same way.

---

## Item 5 — Decoupled Classifier Re-training (cRT / LWS, Kang et al., ICLR 2020)

**Design:** after a normal training run completes (instance-balanced
sampling, as today), freeze the backbone **and** MSDA/HFF, then re-train
only the classification/aux/ordinal heads for a short schedule
(10–15 epochs, lower LR) with **class-balanced sampling** (the existing
`WeightedRandomSampler` already in `src/data/datamodule.py`, gated by
`use_weighted_sampling` — already available, just needs forcing to `True`
for this phase regardless of the main config value).

### `src/models/dr_model.py`
Add a second freezing method alongside `_freeze_backbone_stages`
(from P1):
```python
    def freeze_all_except_heads(self):
        """Freeze backbone + MSDA/HFF; leave classification/aux/ordinal heads trainable.
        Used for decoupled classifier re-training (cRT/LWS)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if self.use_msda:
            for p in self.msda3.parameters(): p.requires_grad_(False)
            for p in self.msda4.parameters(): p.requires_grad_(False)
        if self.use_hff:
            for p in self.hff.parameters(): p.requires_grad_(False)
        # self.head, self.aux_head, self.ordinal_head stay trainable
```

### `main.py` — new mode
Add `'decoupled_retrain'` to the `--mode` choices (line with
`choices=['train', 'test', ...]`). New function:
```python
def run_decoupled_retrain(config, ablation, device, logger, fold_idx=None):
    """Phase 2: freeze backbone+MSDA+HFF, re-train heads with class-balanced
    sampling for a short schedule. Loads the existing best checkpoint,
    saves to a distinct suffix so the original is never overwritten."""
    train_loader, val_loader, _, _, class_weights = get_dataloaders(
        {**config, "use_weighted_sampling": True}, fold_idx=fold_idx
    )
    model = create_model(config, ablation, device)
    suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    model.load_state_dict(torch.load(f'checkpoints/best_model_{ablation}{suffix}.pth', map_location=device))
    model.freeze_all_except_heads()

    decoupled_config = {
        **config,
        "epochs": config.get("decoupled_epochs", 12),
        "lr": config.get("decoupled_lr", 3e-5),
        "warmup_epochs": 1,
        "use_swa": False,          # short phase, SWA not meaningful here
        "patience": config.get("decoupled_epochs", 12),  # no early stop
    }
    criterion = create_criterion(decoupled_config, class_weights, device)
    trainer = DRTrainer(model, train_loader, val_loader, criterion, device,
                         decoupled_config, ablation=f"{ablation}_decoupled", fold_idx=fold_idx)
    trainer.train()
```
Checkpoint naturally saves as `checkpoints/best_model_{ablation}_decoupled{suffix}.pth`
via the existing `ablation` naming convention in `trainer.py`'s `train()` —
no change needed there since `ablation` is just a string used for the
filename.

### `configs/config.yaml`
```yaml
decoupled_epochs: 12
decoupled_lr: 3.0e-5
```

---

## Validation / ablation design
Each item is independently toggleable — run a short (single-fold) sweep:
`baseline (none of P2) → +LogitAdj → +DRW → +CB-Loss → +LDAM(+DRW) → +decoupled-retrain`,
tracking **per-class F1 for Severe/Proliferative specifically**, plus
overall accuracy/QWK. This becomes a `training-strategy ablation` table
alongside the architecture and SSL ablation tables (P3).

## Expected outcome
Primary payoff: Severe/Proliferative F1 improvement without a full-dataset
accuracy regression. Secondary: a legitimate, well-cited "training strategy
ablation" table for the paper.
