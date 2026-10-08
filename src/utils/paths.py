import os
import json
import yaml

def make_run_name(ablation, tag): return f"{ablation}__{tag}" if tag else ablation

def parse_overrides(pairs):
    out = {}
    for p in pairs:
        k, v = p.split('=', 1)
        out[k.strip()] = yaml.safe_load(v)      # true/false/null/numbers/strings all work
    return out

def get_checkpoint_path(run_name: str, fold: int = None, create_dir: bool = True) -> str:
    """
    checkpoints/{run_name}/fold{N}.pth  (or model.pth if fold is None).
    """
    folder = os.path.join('checkpoints', run_name)
    if create_dir:
        os.makedirs(folder, exist_ok=True)
    filename = f'fold{fold}.pth' if fold is not None else 'model.pth'
    return os.path.join(folder, filename)

def get_log_path(mode: str, run_name: str, timestamp: str, create_dir: bool = True) -> str:
    """logs/{run_name}/{mode}_{timestamp}.log"""
    folder = os.path.join('logs', run_name)
    if create_dir:
        os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{mode}_{timestamp}.log')

def val_metrics_path(run_name: str, fold: int) -> str:
    return get_checkpoint_path(run_name, fold, create_dir=True).replace('.pth', '_val_metrics.json')

def collect_val_metrics(run_name: str, n_folds: int):
    accs, qwks = [], []
    for f in range(n_folds):
        p = val_metrics_path(run_name, f)
        if os.path.exists(p):
            d = json.load(open(p))
            accs.append(d['val_acc'])
            qwks.append(d['val_qwk'])
    if not accs: return {}
    return {'val_acc': sum(accs)/len(accs), 'val_qwk': sum(qwks)/len(qwks), 'val_folds': len(accs)}
