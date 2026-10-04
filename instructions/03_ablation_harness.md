# P3 — Ablation Harness Extension

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

### Design clarification on the 8-row table
The originally planned table's last two rows ("+Ordinal" and "Full
(Proposed)") have identical checkmarks (✓✓✓✓✓) — they're the same
configuration at two different **evaluation protocols**:
- Row 7 "+Ordinal" → single-fold, no-ensemble ablation protocol (fast,
  comparable to the other ablation rows)
- Row 8 "Full (Proposed)" → the full 5-fold ensemble + multi-scale TTA
  protocol (the paper's headline number)

So only **7 distinct configurations** need to be defined; row 8 reuses row
7's config through the existing full-ensemble `test` pipeline. Conveniently,
row 7's flags (`msda=hff=attnpool=auxhead=ordinal=True`) are **already
exactly what `proposed` uses today** (since config.yaml currently applies
those three heads unconditionally) — so the existing `proposed` 5-fold
logs already ARE row 8, no rerun needed. Only rows 1–6 are new.

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

### Execution protocol
Run rows 1–6 single-fold only:
```bash
python main.py --mode train --ablation arch_baseline --fold 0
python main.py --mode train --ablation arch_msda --fold 0
python main.py --mode train --ablation arch_hff --fold 0
python main.py --mode train --ablation arch_msda_hff --fold 0
python main.py --mode train --ablation arch_attnpool --fold 0
python main.py --mode train --ablation arch_auxhead --fold 0
```
Then test each the same way the existing single-fold-style evaluation is
done (note: `run_test`'s ensemble path expects all 5 fold checkpoints by
default — for single-fold ablation rows, either (a) set
`ensemble_folds: False` in a per-run config override and point
`model_path` logic at the fold-0 checkpoint, or (b) temporarily set
`n_folds: 1` for these runs. Document whichever is chosen consistently
across all 6 rows so the comparison is apples-to-apples.)
Row 7 (`arch_full`) and the existing `proposed` 5-fold results are
identical by construction — reuse, don't rerun.

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
