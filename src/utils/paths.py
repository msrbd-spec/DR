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
