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
