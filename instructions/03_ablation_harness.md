# P3 — Ablation Harness Extension

## ⚠️ SCOPE NOTE — read before implementing
**Part A and Part B below are ALREADY IMPLEMENTED and verified correct in
the current codebase** (ARCHITECTURE_ABLATIONS registry, resolve_ablation_flags,
all 3 call sites, argparse choices, EyePACSSupervisedModel, the dataset
builder, and the pretrain_eyepacs_supervised mode — all confirmed present
and working). **Do NOT re-implement Part A/B — only verify they match this
spec if asked, but the actual new work is the two "Addendum" sections at
the bottom** (folder-structured checkpoints/logs, and the mix_prob
diagnostic note). Line numbers cited in Part A (e.g. "line 72-74", "line
~197", "line ~970") are from an EARLIER repo state and will not match the
current file — if you do need to locate code, match by the quoted code
snippets' content, not the line numbers.

## Objective
Today, `get_ablation_flags()` (`main.py` line 59) only toggles
`use_msda`/`use_hff`; `use_attention_pool`, `use_aux_head`,
`use_ordinal_loss` come from `config.yaml` and apply identically to every
`--ablation` run. So the logged "Baseline" (0.8388 test acc) is NOT the
planned architecture-ablation table's true all-off row — it already has
AttnPool+AuxHead+Ordinal on. This phase makes the full 8-row architecture
table obtainable, and adds the missing `EyePACS supervised` row for the
SSL ablation table.

## Priority / Cost / Risk
Medium cost, additive-only changes (old 4 ablation names keep working
unchanged — zero risk of invalidating already-logged baseline/msda_only/
hff_only/proposed results).

---

## Part A — Architecture ablation (full 5-flag combo)

### Design clarification (REVISED — all 7 rows now run 5-fold)
Originally this table used a mixed protocol (single-fold for rows 1-6,
full 5-fold ensemble reused from `proposed` for row 7/8) to save compute
under a tight deadline. **Revised: with more runway available, every row
runs the full 5-fold ensemble protocol** — this removes the confusing
"same config, different number, needs a footnote" situation entirely, and
turns the `arch_full` vs `arch_baseline` single-fold anomaly (observed in
practice — arch_full scored worst of 7, arch_baseline scored best) into
either a resolved non-issue (if 5-fold averaging closes the gap) or a
genuine, now-trustworthy finding worth reporting directly, instead of a
noise-vs-signal question hanging over a single-fold number.

Only **7 distinct configurations** are defined (see `ARCHITECTURE_ABLATIONS`
below); `arch_full`'s flags are identical to `proposed`'s, so its 5-fold
result should match `proposed`'s 5-fold result directly — if it doesn't,
that's a pipeline bug worth catching, not a protocol-difference footnote.

### `main.py` — new ablation registry
Add near `get_ablation_flags` (after line 69, before `create_model`):
```python
# Full 5-flag architecture ablation registry (P3). Distinct from the
# original 4-name get_ablation_flags(), which stays unchanged for
# backward compatibility with already-logged baseline/msda_only/
# hff_only/proposed results.
ARCHITECTURE_ABLATIONS = {
    'arch_baseline':  dict(use_msda=False, use_hff=False, use_attention_pool=False, use_aux_head=False, use_ordinal=False),
    'arch_msda':      dict(use_msda=True,  use_hff=False, use_attention_pool=False, use_aux_head=False, use_ordinal=False),
    'arch_hff':       dict(use_msda=False, use_hff=True,  use_attention_pool=False, use_aux_head=False, use_ordinal=False),
    'arch_msda_hff':  dict(use_msda=True,  use_hff=True,  use_attention_pool=False, use_aux_head=False, use_ordinal=False),
    'arch_attnpool':  dict(use_msda=True,  use_hff=True,  use_attention_pool=True,  use_aux_head=False, use_ordinal=False),
    'arch_auxhead':   dict(use_msda=True,  use_hff=True,  use_attention_pool=True,  use_aux_head=True,  use_ordinal=False),
    'arch_full':      dict(use_msda=True,  use_hff=True,  use_attention_pool=True,  use_aux_head=True,  use_ordinal=True),
}


def resolve_ablation_flags(ablation, config):
    """Single source of truth for all 3 call sites (create_model, run_test,
    run_external_validation). Returns (use_msda, use_hff, use_attention_pool,
    use_aux_head, use_ordinal)."""
    if ablation in ARCHITECTURE_ABLATIONS:
        f = ARCHITECTURE_ABLATIONS[ablation]
        return f['use_msda'], f['use_hff'], f['use_attention_pool'], f['use_aux_head'], f['use_ordinal']
    use_msda, use_hff = get_ablation_flags(ablation)
    return (use_msda, use_hff,
            config.get("use_attention_pool", True),
            config.get("use_aux_head", True),
            config.get("use_ordinal_loss", True))
```

### Replace all 3 call sites
**`create_model` (line 72-74):** replace
```python
    use_msda, use_hff = get_ablation_flags(ablation)
    use_ordinal = config.get("use_ordinal_loss", True)
    ...
    use_aux_head = config.get("use_aux_head", True)
    use_attention_pool = config.get("use_attention_pool", True)
```
with:
```python
    use_msda, use_hff, use_attention_pool, use_aux_head, use_ordinal = resolve_ablation_flags(ablation, config)
```
(keep the rest of the function — `drop_path_rate`, `num_classes`,
`backbone_name`, `stage_channels`, `ssl_pretrained_path`,
`freeze_backbone_stages` from P1 — unchanged.)

**`run_test` (line ~197)** and **`run_external_validation` (line ~314)**:
same replacement — both currently do
`use_msda, use_hff = get_ablation_flags(ablation)` followed by separate
`config.get(...)` calls for the other 3 flags; replace with the single
`resolve_ablation_flags(ablation, config)` call.

### Argparse choices (line ~970)
```python
    parser.add_argument('--ablation', type=str, default='proposed',
                        choices=['baseline', 'msda_only', 'hff_only', 'proposed',
                                 'arch_baseline', 'arch_msda', 'arch_hff', 'arch_msda_hff',
                                 'arch_attnpool', 'arch_auxhead', 'arch_full'],
                        help="Ablation configuration for the model.")
```

### Execution protocol (REVISED — full 5-fold for all 7 rows)
Per the full-rerun decision, train all 5 folds fresh for each config
(matches `commands.md` Phase 2 exactly — do not mix with any older
fold-0-only checkpoint from a prior partial run):
```bash
for ablation in arch_baseline arch_msda arch_hff arch_msda_hff arch_attnpool arch_auxhead arch_full; do
  for fold in 0 1 2 3 4; do
    python main.py --mode train --ablation $ablation --fold $fold
  done
done
```
Then test normally (standard 5-fold ensemble path, no `ensemble_folds`
override needed):
```bash
for ablation in arch_baseline arch_msda arch_hff arch_msda_hff arch_attnpool arch_auxhead arch_full; do
  python main.py --mode test --ablation $ablation
done
```
`arch_full`'s result should now match `proposed`'s 5-fold result closely
(identical config, identical protocol) — treat any persistent mismatch as
a pipeline bug to investigate, not something to footnote away.

---

## Part B — SSL ablation: missing `EyePACS supervised` row

### New file: `src/models/eyepacs_supervised_model.py`
Lightweight supervised classifier (backbone + simple head, NOT the full
RetiNA-Net apparatus — this is purely for producing a supervised-pretrained
backbone checkpoint comparable to the SSL backbone's output format).
```python
import torch
import torch.nn as nn
import timm


class EyePACSSupervisedModel(nn.Module):
    """Plain SwinV2 backbone + linear head, trained with standard
    supervised CE on EyePACS's own DR labels (trainLabels.csv). Produces
    the 'EyePACS supervised' row for the SSL pretraining ablation table —
    the counterfactual to our self-supervised (contrastive+multitask) SSL."""

    def __init__(self, backbone_name, num_classes=5, drop_path_rate=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True,
            dynamic_img_size=True, img_size=512, drop_path_rate=drop_path_rate
        )
        stage4_channels = self.backbone.feature_info[-1]['num_chs']
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage4_channels, num_classes)

    def forward(self, x):
        stage4 = self.backbone(x)[-1]
        if stage4.shape[-1] == self.fc.in_features:
            stage4 = stage4.permute(0, 3, 1, 2).contiguous()
        pooled = self.pool(stage4).flatten(1)
        return self.fc(pooled)

    def save_backbone(self, path):
        """Save ONLY backbone weights — same format SSLModel.save_backbone()
        produces, so it's drop-in compatible with config's ssl_pretrained_path."""
        torch.save(self.backbone.state_dict(), path)
```

### New file: `src/data/eyepacs_supervised_dataset.py`
Reuses the same preprocessing as `src/data/dataset.py`'s `DRDataset`
(fundus crop, CLAHE, Ben Graham) but reads `(image, level)` pairs from
`ssl_train_labels_csv` (`configs/config_ssl.yaml`'s
`ssl_train_labels_csv: "datasets/EyePACS/trainLabels.csv/trainLabels.csv"`,
columns `image`/`level` — confirmed from `generate_charts`/`main.py`'s
lesion-correlation code which already reads this exact CSV format). Easiest
implementation: subclass or directly reuse `DRDataset` with
`image_dir=ssl_image_dirs[0]`, `labels_df` built from the CSV renaming
`image→id_code`, `level→diagnosis` to match `DRDataset`'s expected column
names — avoids duplicating the preprocessing pipeline entirely.
```python
import pandas as pd
from .dataset import DRDataset


def build_eyepacs_supervised_dataset(ssl_config, transform, img_size=512):
    df = pd.read_csv(ssl_config['ssl_train_labels_csv'])
    df = df.rename(columns={'image': 'id_code', 'level': 'diagnosis'})
    return DRDataset(
        image_dir=ssl_config['ssl_image_dirs'][0],
        labels_df=df, transform=transform, img_size=img_size
    )
```

### `main.py` — new mode `pretrain_eyepacs_supervised`
```python
def run_pretrain_eyepacs_supervised(config, device, logger, timestamp):
    """EyePACS-supervised pretraining — the missing SSL-ablation row."""
    from src.data.eyepacs_supervised_dataset import build_eyepacs_supervised_dataset
    from src.models.eyepacs_supervised_model import EyePACSSupervisedModel
    from src.data.transforms import get_train_transforms, get_val_test_transforms
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np, torch.nn as nn

    ssl_config = load_config(config.get('ssl_config', 'configs/config_ssl.yaml'))
    img_size = ssl_config.get('ssl_img_size', 512)
    dataset = build_eyepacs_supervised_dataset(ssl_config, get_train_transforms(img_size), img_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=ssl_config.get('ssl_batch_size', 24), shuffle=True,
        num_workers=ssl_config.get('ssl_num_workers', 8), pin_memory=True, drop_last=True
    )

    backbone_name = ssl_config.get('ssl_backbone', config.get('backbone'))
    model = EyePACSSupervisedModel(backbone_name).to(device)

    y_all = dataset.labels_df['diagnosis'].values
    class_weights = torch.FloatTensor(
        compute_class_weight('balanced', classes=np.unique(y_all), y=y_all)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ssl_config.get('ssl_epochs', 50))

    epochs = ssl_config.get('ssl_epochs', 50)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        logger.info(f"[EyePACS-supervised] Epoch {epoch+1}/{epochs} Loss: {running_loss/len(loader):.4f}")

    os.makedirs('checkpoints', exist_ok=True)
    model.save_backbone('checkpoints/eyepacs_supervised_backbone.pth')
    logger.info("Saved checkpoints/eyepacs_supervised_backbone.pth")
```
Register `'pretrain_eyepacs_supervised'` in the `--mode` choices and wire
it in `main()`'s if/elif chain.

### Using the new backbone
Set `ssl_pretrained_path: "checkpoints/eyepacs_supervised_backbone.pth"` in
`config.yaml`, run `train`/`test` as normal (`ablation: proposed`) — this
produces the "EyePACS supervised" row's Val/Test Acc/QWK for the SSL
ablation table, same way the 3 already-obtainable rows work.

**Note on auto-populating `results/ssl_ablation_results.npz`:** the general
mechanism (every `test` run auto-appends its metrics to the relevant
results file instead of requiring manual npz edits) is specified in P7 —
this phase only needs to make the 5th row's data *obtainable*; P7 makes it
*automatic*.

---

## Addendum — Folder-structured checkpoints/logs (repo-wide, added per user request)
**Note: this affects P1 through P7, not just architecture ablation — it's
documented here because this is where the checkpoint-naming complexity
became unmanageable (11+ configs × 5 folds = 55+ flat files), but the
fix applies everywhere `checkpoints/` or `logs/` paths are constructed.**

### New file: `src/utils/paths.py`
```python
import os


def get_checkpoint_path(ablation: str, fold: int = None, create_dir: bool = True) -> str:
    """
    checkpoints/{ablation}/fold{N}.pth  (or model.pth if fold is None).
    Replaces the old flat checkpoints/best_model_{ablation}_fold{N}.pth
    naming — same info, folder-per-config instead of one giant flat dir.
    """
    folder = os.path.join('checkpoints', ablation)
    if create_dir:
        os.makedirs(folder, exist_ok=True)
    filename = f'fold{fold}.pth' if fold is not None else 'model.pth'
    return os.path.join(folder, filename)


def get_log_path(mode: str, ablation: str, timestamp: str, create_dir: bool = True) -> str:
    """logs/{ablation}/{mode}_{timestamp}.log"""
    folder = os.path.join('logs', ablation)
    if create_dir:
        os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{mode}_{timestamp}.log')
```

### Wire into existing save/load points
- `src/training/trainer.py` — every `torch.save(..., f'checkpoints/best_model_{self.ablation}{suffix}.pth')`
  (both the EMA-best save and the SWA-override save in `train()`) →
  `torch.save(..., get_checkpoint_path(self.ablation, fold=self.fold_idx))`
- `main.py` `run_test` / `run_external_validation` — `model_paths = [f'checkpoints/best_model_{ablation}_fold{f}.pth' for f in range(n_folds)]`
  → `model_paths = [get_checkpoint_path(ablation, fold=f, create_dir=False) for f in range(n_folds)]`
  (filter to existing paths the same way as before)
- `main.py` `run_xai` — same substitution for its single `model_path`
- `main.py` `run_decoupled_retrain` (P2) — load path uses
  `get_checkpoint_path(ablation, fold=fold_idx, create_dir=False)`;
  save naturally goes to `checkpoints/{ablation}_decoupled/fold{N}.pth`
  via the existing `ablation=f"{ablation}_decoupled"` naming passed to `DRTrainer`
- `main.py` `main()` — `log_file_path = os.path.join('logs', f'{args.mode}_{args.ablation}_{timestamp}.log')`
  → `log_file_path = get_log_path(args.mode, args.ablation, timestamp)`

### Note on existing checkpoints
If re-running everything from scratch (as planned), no migration needed —
the new structure populates naturally on the fresh runs. If any existing
flat-structure checkpoint needs preserving, migrate with:
```bash
cd checkpoints
for f in best_model_*_fold*.pth; do
  base="${f%.pth}"; base="${base#best_model_}"
  fold="${base##*_fold}"; ablation="${base%_fold*}"
  mkdir -p "$ablation"
  mv "$f" "$ablation/fold${fold}.pth"
done
```

---

## Addendum — Diagnostic for the `proposed` post-P1 regression (85.25%→84.97%)
**Observed in practice, not yet resolved.** In addition to checking the
early-stop epoch (already flagged), add a diagnostic run reducing
Mixup/CutMix strength — P1 raised `mix_prob` 0.2→0.4 at the same time as
adding backbone freezing + CoarseDropout + (optionally) SAM, so the
*combined* regularization may now be too aggressive, not any single piece:
```yaml
mix_prob: 0.1   # diagnostic — try low/near-zero to isolate whether
                 # P1's combined regularization overshot
```
Keep `patience: 25` unchanged (user decision — review logs before
touching this). Run one fold with `mix_prob: 0.1` and one with `0.0`,
compare val QWK trajectory against the `mix_prob: 0.4` run's log — this
isolates whether Mixup/CutMix specifically is the regression's cause
before concluding anything.