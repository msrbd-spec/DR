# P7 — Final Results-Generation Automation

## Objective
Today, every ablation table (`generate_architecture_ablation_table`,
`generate_ssl_ablation_table`, and P6's planned diffusion-ablation table)
either requires manually editing a `results/*.npz` file, or falls back to
an all-zero placeholder template (`main.py` lines ~739-752 for SSL; the
architecture table at least merges an existing CSV at lines 549-566 but
only for single-run-at-a-time `test` invocations). Goal: every `test` /
`external_validation` / SSL-pretrain / diffusion-generation run
auto-appends its own row to the right persistent results file, so one
`--mode generate_results` call at the end produces every table with zero
manual editing.

## Priority / Cost / Risk
Low cost, no risk — purely additive logging, doesn't change any existing
metric computation.

---

## New file: `src/evaluation/results_logger.py`
```python
"""
Shared persistent-results logger. Every training/testing run that
produces a row for one of the paper's ablation tables calls one of these
functions instead of requiring a manual results/*.npz edit.

Each log file stores a dict keyed by row name -> metrics dict, saved via
np.savez(..., allow_pickle=True) so it can be incrementally updated
across many separate process invocations (each `python main.py --mode
test ...` run is a fresh process).
"""

import os
import numpy as np


def _load_results_dict(path):
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return {k: data[k].item() for k in data.files}
    return {}


def _save_results_dict(path, results_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: np.array(v, dtype=object) for k, v in results_dict.items()})


def log_architecture_ablation_result(row_name, metrics, path='results/ablation_results.npz'):
    """metrics: dict from compute_all_metrics() (generate_tables.py) —
    accuracy, precision_macro, recall_macro, f1_macro, f1_weighted, qwk,
    kappa, auc_macro, plus per-class sensitivity/specificity."""
    results = _load_results_dict(path)
    results[row_name] = metrics
    _save_results_dict(path, results)


def log_ssl_ablation_result(row_name, val_acc, val_qwk, test_acc, test_qwk,
                             path='results/ssl_ablation_results.npz'):
    results = _load_results_dict(path)
    results[row_name] = {'val_acc': val_acc, 'val_qwk': val_qwk,
                          'test_acc': test_acc, 'test_qwk': test_qwk}
    _save_results_dict(path, results)


def log_diffusion_ablation_result(row_name, metrics, path='results/diffusion_ablation_results.npz'):
    """metrics: dict with fid, kid, per-class f1/recall (esp. Severe/
    Proliferative), ssim_leakage_max, generation_time_per_step."""
    results = _load_results_dict(path)
    results[row_name] = metrics
    _save_results_dict(path, results)
```

---

## Wiring into existing run functions

### `main.py` — `run_test` (starts line 173)
After `metrics = compute_metrics(...)` (near the end of the function,
where `logger.info(f"Test Metrics - ...")` is already called), add:
```python
    from src.evaluation.results_logger import log_architecture_ablation_result
    from src.evaluation.generate_tables import compute_all_metrics
    full_metrics = compute_all_metrics(all_targets, all_preds, y_prob=np.array(all_probs),
                                        num_classes=config.get("num_classes", 5))
    log_architecture_ablation_result(ablation, full_metrics)
```
This covers both the original 4 ablation names and P3's new
`arch_*` names automatically — no special-casing needed, `ablation` is
already the row-name string in both systems.

### `main.py` — SSL pretraining paths
Both the existing `run_pretrain` (self-supervised) and P3's new
`run_pretrain_eyepacs_supervised`, after producing a backbone checkpoint,
should trigger a `test` run using that backbone
(`ssl_pretrained_path` pointed at the new checkpoint, `ablation: proposed`)
and log the result under the matching SSL row name:
```python
    # after producing the backbone, run a quick val+test pass with it,
    # then:
    from src.evaluation.results_logger import log_ssl_ablation_result
    log_ssl_ablation_result(
        row_name,  # e.g. "Contrastive only", "EyePACS supervised", etc.
        val_acc=val_metrics['accuracy'], val_qwk=val_metrics['qwk'],
        test_acc=test_metrics['accuracy'], test_qwk=test_metrics['qwk']
    )
```
Map `ssl_use_contrastive`/`ssl_use_multitask`/mode to the 5 canonical row
names (`ImageNet pretrain`, `EyePACS supervised`, `Contrastive only`,
`Multi-task only`, `Full SSL (Proposed)`) via a small lookup table at the
top of whichever function orchestrates this — avoid hardcoding the row
name inline in multiple places.

### P6's `generation_ablation` mode
Same pattern — after each of the 8 rows' FID/KID/downstream-metrics are
computed, call `log_diffusion_ablation_result(row_name, metrics)`.

---

## `run_generate_results` (main.py, starts line 490) — replace the
manual/placeholder paths

**Architecture table (lines ~544-567):** replace the CSV-merge logic with
a direct read of the new persistent log:
```python
    from src.evaluation.results_logger import _load_results_dict
    arch_results = _load_results_dict('results/ablation_results.npz')
    if test_ablation not in arch_results and full_metrics is not None:
        arch_results[test_ablation] = full_metrics   # current run, if not already logged
    if arch_results:
        generate_architecture_ablation_table(arch_results)
```

**SSL table (lines ~739-752):** replace the placeholder-template fallback:
```python
    ssl_abl_results = _load_results_dict('results/ssl_ablation_results.npz')
    if not ssl_abl_results:
        logger.warning("No SSL ablation results logged yet. Run the SSL "
                        "ablation sweep (P3) before generate_results for real values.")
    else:
        generate_ssl_ablation_table(ssl_abl_results)
```
(Drop the all-zero placeholder dict entirely — a loud warning is more
useful than silently emitting a fake-looking table.)

**New: diffusion-augmentation table (P6)** — add a parallel block:
```python
    from src.evaluation.results_logger import _load_results_dict
    diffusion_results = _load_results_dict('results/diffusion_ablation_results.npz')
    if diffusion_results:
        generate_diffusion_ablation_table(diffusion_results)  # new function, generate_tables.py
```

### New function: `src/evaluation/generate_tables.py` — `generate_diffusion_ablation_table`
Mirrors the existing `generate_ssl_ablation_table`/
`generate_architecture_ablation_table` pattern (same `fmt()` helper,
same CSV-write-to-`results/tables/` convention):
```python
def generate_diffusion_ablation_table(results, output_dir='results/tables'):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for name, m in results.items():
        rows.append({
            'Configuration': name,
            'FID': fmt(m.get('fid', 0)),
            'KID': fmt(m.get('kid', 0), decimals=4),
            'Severe F1 (%)': fmt(m.get('severe_f1', 0) * 100),
            'Proliferative F1 (%)': fmt(m.get('proliferative_f1', 0) * 100),
            'Max SSIM (leakage)': fmt(m.get('ssim_leakage_max', 0), decimals=3),
            'Gen Time/Step (s)': fmt(m.get('generation_time_per_step', 0), decimals=3),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'ablation_diffusion.csv')
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df
```

---

## Validation steps
1. Run two different `--ablation` values through `test`, confirm
   `results/ablation_results.npz` accumulates both rows (doesn't overwrite).
2. Run `generate_results` with only 1 of 7 architecture rows logged —
   confirm the table renders with just that row (no crash, no fake
   placeholder rows).
3. Confirm re-running the same `--ablation` twice updates (not duplicates)
   its row.

## Expected outcome
One `python main.py --mode generate_results` call at the end of the whole
experiment sweep (P1 through P6) produces every table and chart needed for
the paper, with no manual npz editing at any point.
