import os
# Force HuggingFace to use local cache only — no network requests (cluster has no internet)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import yaml
import torch
import logging
import datetime
import numpy as np

from src.utils.logger import setup_logger
from src.data.datamodule import get_dataloaders
from src.models.dr_model import ICCIT_DR_Net
from src.models.components import OrdinalRegressionHead
from src.training.loss import CombinedLoss
from src.training.trainer import DRTrainer
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizer import plot_training_curves, plot_confusion_matrix, plot_roc_curve
from src.evaluation.xai import generate_heatmap
from src.evaluation.tta import predict_with_tta, predict_multiscale_tta
from src.evaluation.ensemble import load_ensemble_models, evaluate_ensemble
import cv2


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_ablation_flags(ablation):
    if ablation == 'baseline':
        return False, False
    elif ablation == 'msda_only':
        return True, False
    elif ablation == 'hff_only':
        return False, True
    elif ablation == 'proposed':
        return True, True
    else:
        raise ValueError(f"Unknown ablation mode: {ablation}")


def create_model(config, ablation, device):
    """Create model with config parameters."""
    use_msda, use_hff = get_ablation_flags(ablation)
    use_ordinal = config.get("use_ordinal_loss", True)
    drop_path_rate = config.get("drop_path_rate", 0.3)
    dropout = config.get("dropout", 0.5)
    num_classes = config.get("num_classes", 5)

    model = ICCIT_DR_Net(
        use_msda=use_msda,
        use_hff=use_hff,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        dropout=dropout,
        use_ordinal=use_ordinal
    ).to(device)
    return model


def create_criterion(config, class_weights, device):
    """Create loss function with config parameters."""
    use_ordinal = config.get("use_ordinal_loss", True)
    ordinal_weight = config.get("ordinal_loss_weight", 0.3)
    label_smoothing = config.get("label_smoothing", 0.1)
    num_classes = config.get("num_classes", 5)

    criterion = CombinedLoss(
        class_weights=class_weights,
        device=device,
        label_smoothing=label_smoothing,
        use_ordinal=use_ordinal,
        ordinal_loss_weight=ordinal_weight,
        num_classes=num_classes
    )
    return criterion


def train_single_fold(config, ablation, device, logger, fold_idx=None):
    """Train a single fold (or standard train/val split)."""
    train_loader, val_loader, _, _, class_weights = get_dataloaders(config, fold_idx=fold_idx)

    model = create_model(config, ablation, device)
    criterion = create_criterion(config, class_weights, device)
    trainer = DRTrainer(
        model, train_loader, val_loader, criterion, device, config,
        ablation=ablation, fold_idx=fold_idx
    )

    train_losses, val_losses, train_accs, val_accs = trainer.train()
    return train_losses, val_losses, train_accs, val_accs


def train_kfold(config, ablation, device, logger, timestamp):
    """Train K-fold cross-validation models."""
    n_folds = config.get("n_folds", 5)
    all_train_losses = []
    all_val_losses = []
    all_val_accs = []

    for fold_idx in range(n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Fold {fold_idx + 1}/{n_folds}")
        logger.info(f"{'='*60}")

        train_losses, val_losses, train_accs, val_accs = train_single_fold(
            config, ablation, device, logger, fold_idx=fold_idx
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_val_accs.append(val_accs)

        # Plot training curves for this fold
        plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            filename=os.path.join('results', f'training_curves_{ablation}_fold{fold_idx}_{timestamp}.png')
        )

    logger.info(f"\nK-Fold training complete. {n_folds} models saved.")
    return all_train_losses, all_val_losses, all_val_accs


def run_test(config, ablation, device, logger, timestamp):
    """Run test evaluation with optional ensemble."""
    _, _, test_loader, _, _ = get_dataloaders(config, fold_idx=None)

    use_kfold = config.get("use_kfold", True)
    n_folds = config.get("n_folds", 5)
    use_tta = config.get("use_tta", True)
    tta_scales = config.get("tta_scales", [0.9, 1.0, 1.1])
    tta_flips = config.get("tta_flips", True)
    tta_rotations = config.get("tta_rotations", True)
    ensemble_folds = config.get("ensemble_folds", True)

    if use_kfold and ensemble_folds:
        # Ensemble inference across K-fold models
        model_paths = [f'checkpoints/best_model_{ablation}_fold{f}.pth' for f in range(n_folds)]
        model_paths = [p for p in model_paths if os.path.exists(p)]

        if len(model_paths) == 0:
            logger.error("No fold models found. Please train with K-fold first.")
            return

        logger.info(f"Loading ensemble of {len(model_paths)} fold models...")

        use_msda, use_hff = get_ablation_flags(ablation)
        use_ordinal = config.get("use_ordinal_loss", True)
        model_kwargs = {
            'use_msda': use_msda,
            'use_hff': use_hff,
            'num_classes': config.get("num_classes", 5),
            'drop_path_rate': config.get("drop_path_rate", 0.3),
            'dropout': config.get("dropout", 0.5),
            'use_ordinal': use_ordinal,
        }

        models = load_ensemble_models(ICCIT_DR_Net, model_paths, device, **model_kwargs)

        logger.info("Running ensemble inference with multi-scale TTA...")
        metrics, all_preds, all_targets, all_probs = evaluate_ensemble(
            models, test_loader, device,
            use_tta=use_tta,
            use_multiscale=True,
            scales=tuple(tta_scales),
            use_flips=tta_flips,
            use_rotations=tta_rotations
        )
    else:
        # Single model inference
        model = create_model(config, ablation, device)
        model_path = f'checkpoints/best_model_{ablation}.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                if use_tta:
                    probs, preds = predict_multiscale_tta(
                        model, inputs,
                        scales=tuple(tta_scales),
                        use_flips=tta_flips,
                        use_rotations=tta_rotations
                    )
                else:
                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                        ordinal_logits = outputs.get('ordinal_logits')
                        if ordinal_logits is not None:
                            cls_probs = torch.softmax(logits, dim=1)
                            ord_probs = OrdinalRegressionHead.ordinal_logits_to_class_probs(ordinal_logits)
                            probs = 0.5 * cls_probs + 0.5 * ord_probs
                        else:
                            probs = torch.softmax(logits, dim=1)
                    else:
                        probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(all_targets, all_preds)

    logger.info(f"Test Metrics - Acc: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_macro']:.4f}, QWK: {metrics['qwk']:.4f}")
    logger.info(f"Test Classification Report:\n{metrics['classification_report']}")

    plot_confusion_matrix(all_targets, all_preds, filename=os.path.join('results', f'confusion_matrix_{ablation}_{timestamp}.png'))
    plot_roc_curve(all_targets, np.array(all_probs), filename=os.path.join('results', f'roc_multiclass_{ablation}_{timestamp}.png'))


def run_external_validation(config, ablation, device, logger, timestamp):
    """Run external validation on Messidor-2 with optional ensemble."""
    _, _, _, ext_loader, _ = get_dataloaders(config, fold_idx=None)

    if ext_loader is None:
        logger.error("External validation dataset not configured.")
        return

    use_kfold = config.get("use_kfold", True)
    n_folds = config.get("n_folds", 5)
    use_tta = config.get("use_tta", True)
    tta_scales = config.get("tta_scales", [0.9, 1.0, 1.1])
    tta_flips = config.get("tta_flips", True)
    tta_rotations = config.get("tta_rotations", True)
    ensemble_folds = config.get("ensemble_folds", True)

    if use_kfold and ensemble_folds:
        model_paths = [f'checkpoints/best_model_{ablation}_fold{f}.pth' for f in range(n_folds)]
        model_paths = [p for p in model_paths if os.path.exists(p)]

        if len(model_paths) == 0:
            logger.error("No fold models found. Please train with K-fold first.")
            return

        logger.info(f"Loading ensemble of {len(model_paths)} fold models for external validation...")

        use_msda, use_hff = get_ablation_flags(ablation)
        use_ordinal = config.get("use_ordinal_loss", True)
        model_kwargs = {
            'use_msda': use_msda,
            'use_hff': use_hff,
            'num_classes': config.get("num_classes", 5),
            'drop_path_rate': config.get("drop_path_rate", 0.3),
            'dropout': config.get("dropout", 0.5),
            'use_ordinal': use_ordinal,
        }

        models = load_ensemble_models(ICCIT_DR_Net, model_paths, device, **model_kwargs)

        metrics, all_preds, all_targets, all_probs = evaluate_ensemble(
            models, ext_loader, device,
            use_tta=use_tta,
            use_multiscale=True,
            scales=tuple(tta_scales),
            use_flips=tta_flips,
            use_rotations=tta_rotations
        )
    else:
        model = create_model(config, ablation, device)
        model_path = f'checkpoints/best_model_{ablation}.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for inputs, targets in ext_loader:
                inputs = inputs.to(device)
                if use_tta:
                    probs, preds = predict_multiscale_tta(
                        model, inputs,
                        scales=tuple(tta_scales),
                        use_flips=tta_flips,
                        use_rotations=tta_rotations
                    )
                else:
                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        probs = torch.softmax(outputs['logits'], dim=1)
                    else:
                        probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(all_targets, all_preds)

    logger.info(f"External Validation Metrics - Acc: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_macro']:.4f}, QWK: {metrics['qwk']:.4f}")
    logger.info(f"External Validation Classification Report:\n{metrics['classification_report']}")

    plot_confusion_matrix(all_targets, all_preds, filename=os.path.join('results', f'confusion_matrix_external_{ablation}_{timestamp}.png'))


def run_xai(config, ablation, device, logger, timestamp):
    """Generate XAI heatmaps."""
    _, _, test_loader, _, _ = get_dataloaders(config, fold_idx=None)

    model = create_model(config, ablation, device)

    # Try to load fold 0 model, fall back to single model
    model_path = f'checkpoints/best_model_{ablation}_fold0.pth'
    if not os.path.exists(model_path):
        model_path = f'checkpoints/best_model_{ablation}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Get one sample from test_loader
    inputs, targets = next(iter(test_loader))
    single_tensor = inputs[0:1].to(device)

    # For visualization, denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = single_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    generate_heatmap(single_tensor, model, img_np, out_name=os.path.join('results', f'xai_heatmap_{ablation}_{timestamp}.png'))
    logger.info(f"XAI heatmap generated as results/xai_heatmap_{ablation}_{timestamp}.png")


def main():
    parser = argparse.ArgumentParser(description="ICCIT DR Classification Project")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test', 'external_validation', 'xai'],
                        help="Execution mode.")
    parser.add_argument('--ablation', type=str, default='proposed',
                        choices=['baseline', 'msda_only', 'hff_only', 'proposed'],
                        help="Ablation configuration for the model.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Path to config.yaml")
    parser.add_argument('--fold', type=int, default=None,
                        help="Train only a specific fold (0-indexed). If not specified, trains all folds.")

    args = parser.parse_args()

    config = load_config(args.config)

    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join('logs', f'{args.mode}_{args.ablation}_{timestamp}.log')

    logger = setup_logger(log_file=log_file_path)
    logger.info(f"Starting execution in mode: {args.mode}, ablation: {args.ablation}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.mode == 'train':
        use_kfold = config.get("use_kfold", True)

        if use_kfold:
            if args.fold is not None:
                # Train single fold
                logger.info(f"Training single fold: {args.fold}")
                train_losses, val_losses, train_accs, val_accs = train_single_fold(
                    config, args.ablation, device, logger, fold_idx=args.fold
                )
                plot_training_curves(
                    train_losses, val_losses, train_accs, val_accs,
                    filename=os.path.join('results', f'training_curves_{args.ablation}_fold{args.fold}_{timestamp}.png')
                )
            else:
                # Train all folds
                train_kfold(config, args.ablation, device, logger, timestamp)
        else:
            # Standard train/val split
            train_losses, val_losses, train_accs, val_accs = train_single_fold(
                config, args.ablation, device, logger, fold_idx=None
            )
            plot_training_curves(
                train_losses, val_losses, train_accs, val_accs,
                filename=os.path.join('results', f'training_curves_{args.ablation}_{timestamp}.png')
            )

        logger.info("Training complete.")

    elif args.mode == 'test':
        run_test(config, args.ablation, device, logger, timestamp)

    elif args.mode == 'external_validation':
        run_external_validation(config, args.ablation, device, logger, timestamp)

    elif args.mode == 'xai':
        run_xai(config, args.ablation, device, logger, timestamp)


if __name__ == '__main__':
    main()
