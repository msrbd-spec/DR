# P3 — Ablation Harness & Experiment Tracking (v3 — supersedes all earlier versions of this file)

Goal: every experiment is ONE copy-paste command (no shell loops, no manual
`config.yaml` edits), each run is stored under its own name so nothing is ever
overwritten, and `generate_results` builds every paper table automatically.

## 0. Status — do NOT redo
Already implemented and verified in the current repo (leave untouched):
- `ARCHITECTURE_ABLATIONS`, `resolve_ablation_flags()` (3 call sites), `arch_*` argparse choices
- `EyePACSSupervisedModel`, `eyepacs_supervised_dataset.py`, `--mode pretrain_eyepacs_supervised`
- `--mode train` WITHOUT `--fold` already trains all 5 folds in one command (`train_kfold`)
- `--mode decoupled_retrain` WITHOUT `--fold` already loops all 5 folds
- `--mode test` / `external_validation` already do the 5-fold ensemble in one command

Keep `arch_full` in the registry, but it is IDENTICAL to `proposed` (same 5 flags).
**Never run `arch_full`; the "Full" row of the architecture table is the `proposed` run.**
(`baseline`/`msda_only`/`hff_only` are NOT duplicates of `arch_baseline`/`arch_msda`/`arch_hff`:
the old three keep AttnPool+AuxHead+Ordinal ON via config; the `arch_*` ones turn them OFF.)

## 1. Problems this version fixes (found by reading the code)
1. Later phases reuse `--ablation proposed` -> they would overwrite Phase-2 `proposed` checkpoints, logs and result rows.
2. `test --ablation proposed_decoupled` is invalid (not an argparse choice; `get_ablation_flags` raises on unknown names).
3. SSL variants: `SSLTrainer` resumes from `<ssl_save_dir>/ssl_training_checkpoint.pth`. A 2nd `pretrain` with the same
   `ssl_save_dir` prints "Already completed" and trains NOTHING. Also `run_pretrain` builds `SSLModel` from the file's
   `ssl_config` dict, not the merged/overridden `config`.
4. Nothing ever writes SSL-ablation rows (`log_ssl_ablation_result` is never called) and the training-strategy table only has placeholders.
5. Flat `checkpoints/` and `logs/` (55+ files) are unmanageable.
6. `results/test_predictions.npz` and `external_predictions.npz` are overwritten by every run.
7. Eval-time config can silently differ from train-time config (e.g. `use_rep_proj` changes the model structure).

## 2. Work items

### W1 — New CLI args + overrides + snapshots (`main.py`)
Add to argparse:
```python
parser.add_argument('--tag', type=str, default='', help='experiment tag; run name = <ablation>__<tag> (no spaces or slashes)')
parser.add_argument('--inherit', type=str, default='', help='start from the saved config of run <ablation>__<INHERIT>')
parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides, e.g. --set use_sam=true sam_rho=0.05')
```
Helpers (put in `src/utils/paths.py`, see W2):
```python
import yaml
def make_run_name(ablation, tag): return f"{ablation}__{tag}" if tag else ablation

def parse_overrides(pairs):
    out = {}
    for p in pairs:
        k, v = p.split('=', 1)
        out[k.strip()] = yaml.safe_load(v)      # true/false/null/numbers/strings all work
    return out
```
Flow in `main()` right after `config = load_config(args.config)`:
```python
run_name = make_run_name(args.ablation, args.tag)
EVAL_MODES = {'test', 'external_validation', 'xai', 'decoupled_retrain'}
inherit_name = make_run_name(args.ablation, args.inherit) if args.inherit else (run_name if args.mode in EVAL_MODES else None)
if inherit_name:
    snap = os.path.join('checkpoints', inherit_name, 'config_snapshot.yaml')
    if os.path.exists(snap):
        config.update(yaml.safe_load(open(snap)))      # restore train-time config (all keys)
    else:
        print(f"WARNING: no config snapshot for run '{inherit_name}', using configs/config.yaml")
overrides = parse_overrides(args.set)
config.update(overrides)
config['_cli_overrides'] = overrides                 # re-applied inside run_pretrain after its config merge (W5)
```
Snapshot saving: for `--mode train` save the final config (everything except `_cli_overrides`) to
`checkpoints/<run_name>/config_snapshot.yaml` BEFORE training starts; for `decoupled_retrain` save it to
`checkpoints/<run_name>__decoupled/config_snapshot.yaml`.
`--fold N` keeps working (diagnostics only).

### W2 — Folder-structured checkpoints and logs
New file `src/utils/paths.py` (together with the helpers above):
```python
import os, json

def get_checkpoint_path(run_name, fold=None, create_dir=True):
    folder = os.path.join('checkpoints', run_name)
    if create_dir: os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'fold{fold}.pth' if fold is not None else 'model.pth')

def get_log_path(mode, run_name, timestamp, create_dir=True):
    folder = os.path.join('logs', run_name)
    if create_dir: os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{mode}_{timestamp}.log')

def val_metrics_path(run_name, fold):
    return get_checkpoint_path(run_name, fold).replace('.pth', '_val_metrics.json')

def collect_val_metrics(run_name, n_folds):
    accs, qwks = [], []
    for f in range(n_folds):
        p = val_metrics_path(run_name, f)
        if os.path.exists(p):
            d = json.load(open(p)); accs.append(d['val_acc']); qwks.append(d['val_qwk'])
    if not accs: return {}
    return {'val_acc': sum(accs)/len(accs), 'val_qwk': sum(qwks)/len(qwks), 'val_folds': len(accs)}
```
Wiring:
- `train_single_fold(config, ablation, device, logger, fold_idx=None, run_name=None)` and `train_kfold(...)`: pass
  `ablation=run_name or ablation` into `DRTrainer` (the trainer only uses that string for naming).
- `trainer.py`: both `torch.save(..., f'checkpoints/best_model_{self.ablation}{suffix}.pth')` calls ->
  `torch.save(..., get_checkpoint_path(self.ablation, fold=self.fold_idx))`. At every new best (EMA path) and when the SWA
  model overrides, write `val_metrics_path(self.ablation, self.fold_idx)` = `{"epoch":E,"val_acc":A,"val_qwk":Q}`
  (for SWA use the SWA validation numbers).
- `main()`: `log_file_path = get_log_path(args.mode, run_name, timestamp)`.
- Architecture flags still come from `--ablation` (`create_model(config, ablation, ...)` unchanged); ONLY paths/names/result keys use `run_name`.
- `run_test`, `run_external_validation`, `run_xai`, `run_decoupled_retrain` get an extra `run_name` argument; every
  `checkpoints/best_model_{ablation}...` string becomes `get_checkpoint_path(run_name, fold=f, create_dir=False)`.
  Keep the existing "filter to existing files" logic, and **log an error and abort if fewer than `n_folds` checkpoints exist**
  (never silently ensemble a partial set).

### W3 — Decoupled re-training naming
`run_decoupled_retrain(config, ablation, device, logger, fold_idx, run_name)`:
load base checkpoint from `get_checkpoint_path(run_name, fold_idx, create_dir=False)`; train with
`DRTrainer(..., ablation=f"{run_name}__decoupled", ...)`. Result: `checkpoints/proposed__stepB__decoupled/foldN.pth`.
It is evaluated with `--ablation proposed --tag stepB__decoupled` (run name = `proposed__stepB__decoupled`).
Loss flags come from the restored snapshot of the base run (W1), so they never need to be re-typed.

### W4 — Unified result logging (replaces the per-table npz hooks of 07)
`results_logger.py`: add (keep the old functions, just stop calling them):
```python
def log_run(run_name, kind, metrics, path='results/all_runs.npz'):
    data = _load_results_dict(path)
    entry = data.get(run_name, {}); entry[kind] = metrics; data[run_name] = entry
    _save_results_dict(path, data)
```
- `compute_all_metrics` (generate_tables.py): also return `f1_class0 ... f1_class4` (`f1_score(..., average=None, labels=range(num_classes))`).
- `run_test`: `m = compute_all_metrics(...)`; `m.update(collect_val_metrics(run_name, n_folds))`; `log_run(run_name, 'test', m)`.
  Save predictions to `results/predictions/<run_name>_test.npz` (and also the legacy `results/test_predictions.npz`).
- `run_external_validation`: `log_run(run_name, 'external', compute_all_metrics(...))`; save to
  `results/predictions/<run_name>_external.npz` (and the legacy `external_predictions.npz`).

### W5 — SSL variants must be able to run side by side
- `run_pretrain`: after `for key,val in ssl_config.items(): config[key]=val`, immediately re-apply
  `config.update(config.get('_cli_overrides', {}))`. Then read `use_contrastive`, `use_multitask`, `batch_size`,
  `num_workers`, `img_size`, `mask_size`, `backbone_name`, `projection_dim`, `lesion_label_dir`, `image_dirs`
  from the merged `config` (not from `ssl_config`), so `--set ssl_use_contrastive=false` really builds the right model.
- `ssl_trainer._save_loss_history`: ALSO save `ssl_loss_history.npz` inside `self.save_dir`.
- Each variant uses its own `ssl_save_dir` (given on the command line, see commands.md), so its resume file is fresh.
  Never change the default `ssl_save_dir: checkpoints` — the original "Full SSL" backbone stays where it is.

### W6 — `configs/tables.yaml` + table building in `generate_results`
Create `configs/tables.yaml` (user edits the `<WINNER...>` names once at the end):
```yaml
main_run: proposed                 # run used for confusion matrix / ROC / PR / radar / XAI figures
architecture:                      # 7-row table (heads switched off progressively)
  "Baseline": arch_baseline
  "+MSDA": arch_msda
  "+HFF": arch_hff
  "+MSDA+HFF": arch_msda_hff
  "+AttnPool": arch_attnpool
  "+AuxHead": arch_auxhead
  "+Ordinal = Full (Proposed)": proposed
architecture_heads_on:             # secondary table (AttnPool+Aux+Ordinal always on)
  "Baseline": baseline
  "MSDA only": msda_only
  "HFF only": hff_only
  "MSDA+HFF (Proposed)": proposed
ssl:
  "ImageNet pretrain": proposed__ssl_imagenet
  "EyePACS supervised": proposed__ssl_eyepacs_sup
  "Contrastive only": proposed__ssl_contrastive
  "Multi-task only": proposed__ssl_multitask
  "Full SSL (Proposed)": proposed
training:
  "Baseline (focal + inverse weights)": proposed
  "+ Logit Adjustment": proposed__stepA
  "+ DRW": proposed__stepB
  "+ Class-Balanced Loss": proposed__stepC
  "LDAM + DRW": proposed__stepD
  "+ Decoupled re-training": <WINNER_STEP>__decoupled      # e.g. proposed__stepB__decoupled
  "+ SAM": proposed__sam
  "+ RepConv (efficiency)": proposed__rep
external:
  "Locked config, FDA off": <WINNER>      # e.g. proposed__stepB__decoupled or proposed__sam
  "Locked config + FDA": proposed__fda
```
`run_generate_results` reads `all_runs.npz` + `tables.yaml` and builds each table as a VIEW over logged runs:
- architecture / architecture_heads_on: `entry['test']` is already the full metrics dict (add a `filename` parameter to
  `generate_architecture_ablation_table`, write `ablation_architecture.csv` and `ablation_architecture_heads_on.csv`).
- ssl: `{'val_acc': t['val_acc'], 'val_qwk': t['val_qwk'], 'test_acc': t['accuracy'], 'test_qwk': t['qwk']}`.
- training: `{'test_acc': t['accuracy'], 'test_qwk': t['qwk'], 'test_f1': t['f1_macro'], 'test_auc': t['auc_macro']}` and extend
  `generate_training_ablation_table` with `Severe F1 (%)` / `Proliferative F1 (%)` columns from `f1_class3` / `f1_class4`.
- external: `entry['external']` -> `generate_external_validation_table`.
- A row whose run is missing is written with `-` and a WARNING is logged (never zeros, never crash).
- Confusion matrix / ROC / PR / radar / SOTA row read `results/predictions/<main_run>_test.npz` (fall back to the legacy file).
Remove the all-zero placeholder templates for the SSL and training tables.

### W7 — Acceptance checks (run before reporting back)
```bash
python -c "import ast; ast.parse(open('main.py').read())"
python test_model.py
python main.py --mode train --ablation proposed --tag smoke --fold 0 --set epochs=1 warmup_epochs=1
# expect: checkpoints/proposed__smoke/fold0.pth, fold0_val_metrics.json, config_snapshot.yaml and logs/proposed__smoke/train_*.log
python main.py --mode train --ablation proposed --tag smoke2 --fold 0 --set epochs=1 use_sam=true
# expect checkpoints/proposed__smoke2/config_snapshot.yaml to contain use_sam: true
```
Delete the `smoke*` folders afterwards. Report every file changed and why.

## 3. Diagnostic (no code change)
`mix_prob` regression check is just `--set mix_prob=0.1` / `--set mix_prob=0.0` with a tag (see commands.md, Phase 2). Keep `patience: 25`.