import os
# Force HuggingFace to use local cache only — no network requests (cluster has no internet)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# Reduce CUDA memory fragmentation for large models on 40GB A100
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import yaml
import torch
import logging
import datetime
import numpy as np

from src.utils.logger import setup_logger
from src.utils.paths import get_checkpoint_path, get_log_path, make_run_name, parse_overrides
from src.data.datamodule import get_dataloaders
from src.models.dr_model import RetiNA_Net
from src.models.components import OrdinalRegressionHead
from src.training.loss import CombinedLoss
from src.training.trainer import DRTrainer
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizer import plot_training_curves, plot_confusion_matrix, plot_roc_curve
from src.evaluation.xai import generate_heatmap
from src.evaluation.tta import predict_with_tta, predict_multiscale_tta
from src.evaluation.ensemble import load_ensemble_models, evaluate_ensemble
import cv2

# SSL pretraining imports (Phase 2-3 of plan.md)
from src.preprocessing.lesion_detection import detect_lesions_batch
from src.data.eyepacs_dataset import get_ssl_dataloader
from src.models.ssl_model import SSLModel
from src.training.ssl_trainer import SSLTrainer

# Tables & charts generation imports (Phase 8 of plan.md)
from src.evaluation.generate_tables import (
    compute_all_metrics, generate_classification_report_text,
    generate_architecture_ablation_table,
    generate_ssl_ablation_table, generate_training_ablation_table,
    generate_sota_comparison_table,
    generate_per_class_metrics_table,
    generate_external_validation_table
)
from src.evaluation.generate_charts import (
    plot_confusion_matrix as plot_cm_v2,
    plot_roc_curves, plot_pr_curves, plot_ablation_bar_chart,
    plot_ssl_pretraining_curves, plot_lesion_correlation,
    plot_radar_chart, plot_xai_grid
)





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


def create_model(config, ablation, device):
    """Create model with config parameters."""
    use_msda, use_hff, use_attention_pool, use_aux_head, use_ordinal = resolve_ablation_flags(ablation, config)
    drop_path_rate = config.get("drop_path_rate", 0.1)
    dropout = config.get("dropout", 0.1)
    num_classes = config.get("num_classes", 5)
    backbone_name = config.get("backbone", "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k")
    stage_channels = config.get("backbone_channels", [192, 384, 768, 1536])

    # SSL pretrained backbone path (if available, loads SSL weights instead of ImageNet)
    ssl_pretrained_path = config.get("ssl_pretrained_path", None)
    freeze_backbone_stages = config.get("freeze_backbone_stages", 0)
    use_rep_proj = config.get("use_rep_proj", False)

    model = RetiNA_Net(
        use_msda=use_msda,
        use_hff=use_hff,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        dropout=dropout,
        use_ordinal=use_ordinal,
        use_aux_head=use_aux_head,
        use_attention_pool=use_attention_pool,
        backbone_name=backbone_name,
        stage_channels=tuple(stage_channels),
        ssl_pretrained_path=ssl_pretrained_path,
        freeze_backbone_stages=freeze_backbone_stages,
        use_rep_proj=use_rep_proj
    ).to(device)
    return model



def create_criterion(config, class_weights, device, class_priors=None, cls_num_list=None):
    """Create loss function with config parameters."""
    loss_type = config.get("loss_type", "focal")
    ldam_max_margin = config.get("ldam_max_margin", 0.5)
    ldam_scale = config.get("ldam_scale", 30.0)
    use_logit_adjustment = config.get("use_logit_adjustment", False)
    logit_adjustment_tau = config.get("logit_adjustment_tau", 1.0)
    use_ordinal = config.get("use_ordinal_loss", True)
    ordinal_weight = config.get("ordinal_loss_weight", 0.3)
    label_smoothing = config.get("label_smoothing", 0.0)
    num_classes = config.get("num_classes", 5)
    focal_gamma = config.get("focal_gamma", 1.5)
    use_aux = config.get("use_aux_head", True)
    aux_loss_weight = config.get("aux_loss_weight", 0.2)

    criterion = CombinedLoss(
        class_weights=class_weights,
        device=device,
        label_smoothing=label_smoothing,
        use_ordinal=use_ordinal,
        ordinal_loss_weight=ordinal_weight,
        num_classes=num_classes,
        focal_gamma=focal_gamma,
        use_aux=use_aux,
        aux_loss_weight=aux_loss_weight,
        class_priors=class_priors,
        use_logit_adjustment=use_logit_adjustment,
        logit_adjustment_tau=logit_adjustment_tau,
        loss_type=loss_type,
        cls_num_list=cls_num_list,
        ldam_max_margin=ldam_max_margin,
        ldam_scale=ldam_scale
    )
    return criterion


def train_single_fold(config, ablation, device, logger, fold_idx=None, run_name=None):
    """Train a single fold (or standard train/val split)."""
    train_loader, val_loader, _, _, class_weights, class_priors, cls_num_list = get_dataloaders(config, fold_idx=fold_idx)

    model = create_model(config, ablation, device)
    criterion = create_criterion(config, class_weights, device, class_priors=class_priors, cls_num_list=cls_num_list)
    trainer = DRTrainer(
        model, train_loader, val_loader, criterion, device, config,
        ablation=run_name or ablation, fold_idx=fold_idx
    )

    train_losses, val_losses, train_accs, val_accs = trainer.train()
    return train_losses, val_losses, train_accs, val_accs


def train_kfold(config, ablation, device, logger, timestamp, run_name=None):
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
            config, ablation, device, logger, fold_idx=fold_idx, run_name=run_name
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


def run_test(config, ablation, device, logger, timestamp, run_name=None):
    """Run test evaluation with optional ensemble."""
    _, _, test_loader, _, _, _, _ = get_dataloaders(config, fold_idx=None)

    use_kfold = config.get("use_kfold", True)
    n_folds = config.get("n_folds", 5)
    use_tta = config.get("use_tta", True)
    tta_scales = config.get("tta_scales", [0.8, 0.9, 1.0, 1.1, 1.2])
    tta_flips = config.get("tta_flips", True)
    tta_rotations = config.get("tta_rotations", True)
    tta_center_crops = config.get("tta_center_crops", [0.9, 0.95])
    ensemble_folds = config.get("ensemble_folds", True)

    run_name = run_name or ablation
    if use_kfold and ensemble_folds:
        # Ensemble inference across K-fold models
        model_paths = [get_checkpoint_path(run_name, fold=f, create_dir=False) for f in range(n_folds)]
        existing_paths = [p for p in model_paths if os.path.exists(p)]

        if len(existing_paths) < n_folds:
            logger.error(f"Expected {n_folds} checkpoints for ensemble, found {len(existing_paths)}. Aborting.")
            return
        model_paths = existing_paths

        logger.info(f"Loading ensemble of {len(model_paths)} fold models...")

        use_msda, use_hff, use_attention_pool, use_aux_head, use_ordinal = resolve_ablation_flags(ablation, config)
        backbone_name = config.get("backbone", "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k")
        stage_channels = config.get("backbone_channels", [192, 384, 768, 1536])

        model_kwargs = {
            'use_msda': use_msda,
            'use_hff': use_hff,
            'num_classes': config.get("num_classes", 5),
            'drop_path_rate': config.get("drop_path_rate", 0.1),
            'dropout': config.get("dropout", 0.1),
            'use_ordinal': use_ordinal,
            'use_aux_head': use_aux_head,
            'use_attention_pool': use_attention_pool,
            'backbone_name': backbone_name,
            'stage_channels': tuple(stage_channels),
        }

        models = load_ensemble_models(RetiNA_Net, model_paths, device, **model_kwargs)

        logger.info("Running ensemble inference with multi-scale TTA...")
        metrics, all_preds, all_targets, all_probs = evaluate_ensemble(
            models, test_loader, device,
            use_tta=use_tta,
            use_multiscale=True,
            scales=tuple(tta_scales),
            use_flips=tta_flips,
            use_rotations=tta_rotations,
            center_crops=tuple(tta_center_crops)
        )
    else:
        # Single model inference
        model = create_model(config, ablation, device)
        model_path = get_checkpoint_path(run_name, fold=None, create_dir=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        if hasattr(model, 'fuse_reparam_blocks'):
            model.fuse_reparam_blocks()

        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                if use_tta:
                    probs, preds = predict_multiscale_tta(
                        model, inputs,
                        scales=tuple(tta_scales),
                        use_flips=tta_flips,
                        use_rotations=tta_rotations,
                        center_crops=tuple(tta_center_crops)
                    )
                else:
                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                        ordinal_logits = outputs.get('ordinal_logits')
                        if ordinal_logits is not None:
                            cls_probs = torch.softmax(logits, dim=1)
                            ord_probs = OrdinalRegressionHead.ordinal_logits_to_class_probs(ordinal_logits)
                            probs = 0.7 * cls_probs + 0.3 * ord_probs
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

    from src.evaluation.results_logger import log_architecture_ablation_result, log_run
    from src.evaluation.generate_tables import compute_all_metrics
    from src.utils.paths import collect_val_metrics
    full_metrics = compute_all_metrics(all_targets, all_preds, y_prob=np.array(all_probs), num_classes=config.get("num_classes", 5))
    log_architecture_ablation_result(ablation, full_metrics)

    full_metrics.update(collect_val_metrics(run_name, n_folds=config.get("n_folds", 5) if use_kfold else 1))
    log_run(run_name, 'test', full_metrics)

    plot_confusion_matrix(all_targets, all_preds, filename=os.path.join('results', f'confusion_matrix_{ablation}_{timestamp}.png'))
    plot_roc_curve(all_targets, np.array(all_probs), filename=os.path.join('results', f'roc_multiclass_{ablation}_{timestamp}.png'))

    # Save predictions for later table/chart generation (Phase 8)
    os.makedirs(os.path.join('results', 'predictions'), exist_ok=True)
    np.savez(
        os.path.join('results', 'predictions', f'{run_name}_test.npz'),
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        ablation=ablation
    )
    np.savez(
        os.path.join('results', 'test_predictions.npz'),
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        ablation=ablation
    )
    logger.info(f"Test predictions saved to results/test_predictions.npz (for generate_results mode)")



def run_external_validation(config, ablation, device, logger, timestamp, run_name=None):
    """Run external validation on Messidor-2 with optional ensemble."""
    _, _, _, ext_loader, _, _, _ = get_dataloaders(config, fold_idx=None)

    if ext_loader is None:
        logger.error("External validation dataset not configured.")
        return

    use_kfold = config.get("use_kfold", True)
    n_folds = config.get("n_folds", 5)
    use_tta = config.get("use_tta", True)
    tta_scales = config.get("tta_scales", [0.8, 0.9, 1.0, 1.1, 1.2])
    tta_flips = config.get("tta_flips", True)
    tta_rotations = config.get("tta_rotations", True)
    tta_center_crops = config.get("tta_center_crops", [0.9, 0.95])
    ensemble_folds = config.get("ensemble_folds", True)
    run_name = run_name or ablation

    if use_kfold and ensemble_folds:
        model_paths = [get_checkpoint_path(run_name, fold=f, create_dir=False) for f in range(n_folds)]
        existing_paths = [p for p in model_paths if os.path.exists(p)]

        if len(existing_paths) < n_folds:
            logger.error(f"Expected {n_folds} checkpoints for ensemble, found {len(existing_paths)}. Aborting.")
            return
        model_paths = existing_paths

        logger.info(f"Loading ensemble of {len(model_paths)} fold models for external validation...")

        use_msda, use_hff, use_attention_pool, use_aux_head, use_ordinal = resolve_ablation_flags(ablation, config)
        backbone_name = config.get("backbone", "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k")
        stage_channels = config.get("backbone_channels", [192, 384, 768, 1536])

        model_kwargs = {
            'use_msda': use_msda,
            'use_hff': use_hff,
            'num_classes': config.get("num_classes", 5),
            'drop_path_rate': config.get("drop_path_rate", 0.1),
            'dropout': config.get("dropout", 0.1),
            'use_ordinal': use_ordinal,
            'use_aux_head': use_aux_head,
            'use_attention_pool': use_attention_pool,
            'backbone_name': backbone_name,
            'stage_channels': tuple(stage_channels),
        }

        models = load_ensemble_models(RetiNA_Net, model_paths, device, **model_kwargs)

        metrics, all_preds, all_targets, all_probs = evaluate_ensemble(
            models, ext_loader, device,
            use_tta=use_tta,
            use_multiscale=True,
            scales=tuple(tta_scales),
            use_flips=tta_flips,
            use_rotations=tta_rotations,
            center_crops=tuple(tta_center_crops)
        )
    else:
        model = create_model(config, ablation, device)
        model_path = get_checkpoint_path(run_name, fold=None, create_dir=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        if hasattr(model, 'fuse_reparam_blocks'):
            model.fuse_reparam_blocks()

        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for inputs, targets in ext_loader:
                inputs = inputs.to(device)
                if use_tta:
                    probs, preds = predict_multiscale_tta(
                        model, inputs,
                        scales=tuple(tta_scales),
                        use_flips=tta_flips,
                        use_rotations=tta_rotations,
                        center_crops=tuple(tta_center_crops)
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

    from src.evaluation.results_logger import log_run
    from src.evaluation.generate_tables import compute_all_metrics
    ext_metrics = compute_all_metrics(all_targets, all_preds, y_prob=np.array(all_probs), num_classes=config.get("num_classes", 5))
    log_run(run_name, 'external', ext_metrics)

    plot_confusion_matrix(all_targets, all_preds, filename=os.path.join('results', f'confusion_matrix_external_{ablation}_{timestamp}.png'))

    # Save external predictions for later table/chart generation (Phase 8)
    os.makedirs(os.path.join('results', 'predictions'), exist_ok=True)
    np.savez(
        os.path.join('results', 'predictions', f'{run_name}_external.npz'),
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        ablation=ablation
    )
    np.savez(
        os.path.join('results', 'external_predictions.npz'),
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        ablation=ablation
    )
    logger.info(f"External predictions saved to results/external_predictions.npz (for generate_results mode)")


def run_detect_lesions(config, logger, timestamp):

    """
    Phase 2: Run classical lesion detection on EyePACS images.
    Generates .npz pseudo-label files for SSL pretraining.
    """
    ssl_config_path = config.get('ssl_config', 'configs/config_ssl.yaml')
    if os.path.exists(ssl_config_path):
        ssl_config = load_config(ssl_config_path)
    else:
        logger.error(f"SSL config not found: {ssl_config_path}")
        return

    image_dirs = ssl_config.get('ssl_image_dirs', [])
    lesion_label_dir = ssl_config.get('ssl_lesion_label_dir', 'datasets/EyePACS/lesion_labels')
    mask_size = ssl_config.get('ssl_mask_size', 128)

    if not image_dirs:
        logger.error("No ssl_image_dirs specified in config.")
        return

    logger.info(f"Starting lesion detection on images from: {image_dirs}")
    logger.info(f"Output lesion labels: {lesion_label_dir}")
    logger.info(f"Mask size: {mask_size}")

    detect_lesions_batch(image_dirs, lesion_label_dir, mask_size=mask_size)

    logger.info("Lesion detection complete!")


def run_pretrain(config, device, logger, timestamp):
    """
    Phase 3: Run SSL pretraining on EyePACS dataset.
    Saves pretrained backbone weights for fine-tuning.
    """
    ssl_config_path = config.get('ssl_config', 'configs/config_ssl.yaml')
    if os.path.exists(ssl_config_path):
        ssl_config = load_config(ssl_config_path)
    else:
        logger.error(f"SSL config not found: {ssl_config_path}")
        return

    # Merge SSL config into config for SSLTrainer
    for key, val in ssl_config.items():
        config[key] = val
    config.update(config.get('_cli_overrides', {}))

    image_dirs = config.get('ssl_image_dirs', [])
    lesion_label_dir = config.get('ssl_lesion_label_dir', 'datasets/EyePACS/lesion_labels')
    img_size = config.get('ssl_img_size', 512)
    mask_size = config.get('ssl_mask_size', 128)
    batch_size = config.get('ssl_batch_size', 32)
    num_workers = config.get('ssl_num_workers', 8)

    backbone_name = config.get('ssl_backbone', 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k')
    projection_dim = config.get('ssl_projection_dim', 128)
    use_contrastive = config.get('ssl_use_contrastive', True)
    use_multitask = config.get('ssl_use_multitask', True)

    # Create dataloader
    logger.info("Creating SSL dataloader...")
    train_loader = get_ssl_dataloader(
        image_dirs=image_dirs,
        lesion_label_dir=lesion_label_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        mask_size=mask_size
    )

    # Create SSL model
    logger.info("Creating SSL model...")
    ssl_model = SSLModel(
        backbone_name=backbone_name,
        projection_dim=projection_dim,
        use_contrastive=use_contrastive,
        use_multitask=use_multitask
    )

    # Create trainer
    trainer = SSLTrainer(
        model=ssl_model,
        train_loader=train_loader,
        config=config,
        device=device,
        logger=logger
    )

    # Run pretraining
    loss_history = trainer.train()

    # Plot SSL pretraining curves
    os.makedirs('results/figures', exist_ok=True)
    plot_ssl_pretraining_curves(loss_history, output_dir='results/figures')

    logger.info("SSL pretraining complete! Backbone saved to checkpoints/ssl_pretrained_backbone.pth")


def run_generate_results(config, logger, timestamp):
    """
    Phase 8: Generate all tables and charts from saved results.
    """
    logger.info("Generating tables and charts...")
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    CLASS_NAMES = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']
    num_classes = config.get("num_classes", 5)

    tables_config = yaml.safe_load(open('configs/tables.yaml'))
    main_run = tables_config.get('main_run', 'proposed')

    from src.evaluation.results_logger import _load_results_dict
    from src.evaluation.generate_tables import (
        generate_architecture_ablation_table, generate_ssl_ablation_table,
        generate_training_ablation_table, generate_external_validation_table,
        generate_sota_comparison_table, generate_per_class_metrics_table,
        generate_classification_report_text
    )
    all_runs = _load_results_dict('results/all_runs.npz')

    def get_run_metrics(run_name, kind):
        if run_name not in all_runs or kind not in all_runs[run_name]:
            logger.warning(f"Missing {kind} metrics for run: {run_name}")
            return None
        return all_runs[run_name][kind]

    # Architecture Table
    arch_results = {}
    for label, run in tables_config.get('architecture', {}).items():
        m = get_run_metrics(run, 'test')
        arch_results[label] = m if m else {}
    generate_architecture_ablation_table(arch_results, filename='ablation_architecture.csv')

    # Architecture Heads On Table
    arch_heads_results = {}
    for label, run in tables_config.get('architecture_heads_on', {}).items():
        m = get_run_metrics(run, 'test')
        arch_heads_results[label] = m if m else {}
    generate_architecture_ablation_table(arch_heads_results, filename='ablation_architecture_heads_on.csv')

    # SSL Table
    ssl_results = {}
    for label, run in tables_config.get('ssl', {}).items():
        m = get_run_metrics(run, 'test')
        if m:
            ssl_results[label] = {'val_acc': m.get('val_acc', 0), 'val_qwk': m.get('val_qwk', 0),
                                  'test_acc': m.get('accuracy', 0), 'test_qwk': m.get('qwk', 0)}
        else:
            ssl_results[label] = {}
    generate_ssl_ablation_table(ssl_results)

    # Training Strategy Table
    train_results = {}
    for label, run in tables_config.get('training', {}).items():
        m = get_run_metrics(run, 'test')
        if m:
            train_results[label] = {
                'test_acc': m.get('accuracy', 0), 'test_qwk': m.get('qwk', 0),
                'test_f1': m.get('f1_macro', 0), 'test_auc': m.get('auc_macro', 0),
                'test_f1_class3': m.get('f1_class3', 0), 'test_f1_class4': m.get('f1_class4', 0)
            }
        else:
            train_results[label] = {}
    generate_training_ablation_table(train_results)

    # External Validation Table
    ext_results = {}
    for label, run in tables_config.get('external', {}).items():
        m = get_run_metrics(run, 'external')
        if m:
            ext_results[label] = m
    if ext_results:
        generate_external_validation_table(ext_results)

    # Main run predictions (confusion matrix, ROC, PR, radar)
    pred_path = os.path.join('results', 'predictions', f'{main_run}_test.npz')
    if not os.path.exists(pred_path):
        pred_path = os.path.join('results', 'test_predictions.npz')
    
    if os.path.exists(pred_path):
        logger.info(f"Loading test predictions for {main_run} from {pred_path}...")
        pred_data = np.load(pred_path, allow_pickle=True)
        y_true, y_pred, y_prob = pred_data['y_true'], pred_data['y_pred'], pred_data['y_prob']
        
        generate_per_class_metrics_table(y_true, y_pred, y_prob=y_prob, num_classes=num_classes, class_names=CLASS_NAMES)
        generate_classification_report_text(y_true, y_pred, class_names=CLASS_NAMES)
        
        plot_cm_v2(y_true, y_pred, class_names=CLASS_NAMES, normalize=False, title=f'Confusion Matrix — {main_run}', output_path=os.path.join('results/figures', 'confusion_matrix_raw.png'))
        plot_cm_v2(y_true, y_pred, class_names=CLASS_NAMES, normalize=True, title=f'Normalized Confusion Matrix — {main_run}', output_path=os.path.join('results/figures', 'confusion_matrix_normalized.png'))
        plot_roc_curves(y_true, y_prob, num_classes=num_classes, class_names=CLASS_NAMES, output_path=os.path.join('results/figures', 'roc_multiclass.png'))
        plot_pr_curves(y_true, y_prob, num_classes=num_classes, class_names=CLASS_NAMES, output_path=os.path.join('results/figures', 'pr_curve.png'))

        # Radar chart
        per_class_metrics = {}
        for i, cls_name in enumerate(CLASS_NAMES):
            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)
            tp = ((binary_pred == 1) & (binary_true == 1)).sum()
            fn = ((binary_pred == 0) & (binary_true == 1)).sum()
            fp = ((binary_pred == 1) & (binary_true == 0)).sum()
            tn = ((binary_pred == 0) & (binary_true == 0)).sum()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
            per_class_metrics[cls_name] = {'Sensitivity': sensitivity, 'Specificity': specificity, 'F1': f1, 'Precision': precision}
        plot_radar_chart(per_class_metrics, metric_names=['Sensitivity', 'Specificity', 'F1', 'Precision'], class_names=CLASS_NAMES, output_path=os.path.join('results/figures', 'radar_per_class.png'))

        # SOTA
        sota_results = [
            {'method': 'ResNet50', 'backbone': 'ResNet-50', 'year': '2019', 'dataset': 'APTOS-2019', 'acc': 0.0, 'qwk': 0.0, 'auc': 0.0},
            {'method': 'EfficientNet-B5', 'backbone': 'EfficientNet-B5', 'year': '2019', 'dataset': 'APTOS-2019', 'acc': 0.0, 'qwk': 0.0, 'auc': 0.0},
            {'method': 'SwinV2-Large (ImageNet)', 'backbone': 'SwinV2-L', 'year': '2024', 'dataset': 'APTOS-2019', 'acc': 0.0, 'qwk': 0.0, 'auc': 0.0},
            {'method': 'RetiNA-Net (Proposed)', 'backbone': 'SwinV2-L + SSL', 'year': '2025', 'dataset': 'APTOS-2019', 'acc': 0.0, 'qwk': 0.0, 'auc': 0.0},
        ]
        from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
        sota_results[-1]['acc'] = accuracy_score(y_true, y_pred) * 100
        sota_results[-1]['qwk'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        try:
            sota_results[-1]['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception:
            sota_results[-1]['auc'] = 0.0
        generate_sota_comparison_table(sota_results)

    # SSL pretraining curves
    ssl_loss_path = os.path.join('results', 'ssl_loss_history.npz')
    if os.path.exists(ssl_loss_path):
        loss_data = np.load(ssl_loss_path)
        plot_ssl_pretraining_curves({k: loss_data[k].tolist() for k in loss_data.files}, output_dir='results/figures')

    # ===================================================================
    # 10. Generate lesion correlation chart (Chart 5b)
    #     Reads trainLabels.csv + lesion .npz files
    # ===================================================================
    ssl_config_path = config.get('ssl_config', 'configs/config_ssl.yaml')
    if os.path.exists(ssl_config_path):
        ssl_config = load_config(ssl_config_path)
        labels_csv = ssl_config.get('ssl_train_labels_csv', '')
        lesion_label_dir = ssl_config.get('ssl_lesion_label_dir', 'datasets/EyePACS/lesion_labels')

        if labels_csv and os.path.exists(labels_csv) and os.path.exists(lesion_label_dir):
            logger.info("Generating lesion correlation chart (Chart 5b)...")
            import csv as csv_module
            dr_grades = []
            lesion_counts = []
            with open(labels_csv, 'r') as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    basename = row.get('image', '')
                    grade = int(row.get('level', 0))
                    npz_path = os.path.join(lesion_label_dir, basename + '.npz')
                    if os.path.exists(npz_path):
                        data = np.load(npz_path)
                        total_count = int(data['count'].sum())
                        dr_grades.append(grade)
                        lesion_counts.append(total_count)
            if len(dr_grades) > 0:
                plot_lesion_correlation(
                    np.array(dr_grades), np.array(lesion_counts),
                    output_dir='results/figures'
                )
                logger.info(f"  Generated lesion correlation chart from {len(dr_grades)} images.")
            else:
                logger.info("  No lesion labels found. Skipping lesion correlation chart.")
        else:
            logger.info(f"  trainLabels.csv or lesion labels not found. Skipping lesion correlation chart.")
    else:
        logger.info("  No SSL config found. Skipping lesion correlation chart.")

    # ===================================================================
    # Summary
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Table and chart generation complete!")
    logger.info(f"  Tables saved to:   results/tables/")
    logger.info(f"  Figures saved to:  results/figures/")
    logger.info("=" * 60)




def run_xai(config, ablation, device, logger, timestamp, run_name=None):
    """
    Generate XAI heatmaps.
    Produces:
      1. A 5×3 grid (5 DR grades × original/EigenCAM/GradCAM) — Chart 6 from plan
      2. A single sample heatmap (legacy, for quick inspection)
    """
    _, _, test_loader, _, _ = get_dataloaders(config, fold_idx=None)

    run_name = run_name or ablation
    model = create_model(config, ablation, device)

    # Try to load fold 0 model, fall back to single model
    model_path = get_checkpoint_path(run_name, fold=0, create_dir=False)
    if not os.path.exists(model_path):
        model_path = get_checkpoint_path(run_name, fold=None, create_dir=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    if hasattr(model, 'fuse_reparam_blocks'):
        model.fuse_reparam_blocks()

    # Collect one sample per DR grade (0-4)
    grade_samples = {}  # grade → (input_tensor, target)
    for inputs, targets in test_loader:
        for i in range(len(targets)):
            grade = int(targets[i].item())
            if grade not in grade_samples:
                grade_samples[grade] = (inputs[i:i+1], targets[i])
            if len(grade_samples) == 5:
                break
        if len(grade_samples) == 5:
            break

    if len(grade_samples) < 5:
        logger.warning(f"Only found {len(grade_samples)} DR grades in test set. Grid will have missing rows.")

    # Denormalization params
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    CLASS_NAMES = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']

    # --- Generate 5×3 XAI grid (Chart 6) ---
    logger.info("Generating 5-grade XAI grid (Chart 6)...")
    grid_images = []
    grid_eigencam = []
    grid_gradcam = []

    for grade in range(5):
        if grade not in grade_samples:
            # Use black placeholder for missing grade
            grid_images.append(np.zeros((512, 512, 3)))
            grid_eigencam.append(np.zeros((512, 512)))
            grid_gradcam.append(np.zeros((512, 512)))
            continue


        single_tensor, _ = grade_samples[grade]
        single_tensor = single_tensor.to(device)

        # Denormalize for visualization
        img_np = single_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # Generate EigenCAM and GradCAM heatmaps
        try:
            # EigenCAM
            eigencam_heatmap = generate_heatmap(
                single_tensor, model, img_np,
                method='eigencam', return_heatmap=True
            )
        except Exception:
            eigencam_heatmap = np.zeros((512, 512))

        try:
            # GradCAM
            gradcam_heatmap = generate_heatmap(
                single_tensor, model, img_np,
                method='gradcam', return_heatmap=True
            )
        except Exception:
            gradcam_heatmap = np.zeros((512, 512))

        grid_images.append(img_np)
        grid_eigencam.append(eigencam_heatmap)
        grid_gradcam.append(gradcam_heatmap)

    # Save the 5×3 grid
    plot_xai_grid(
        images=grid_images,
        heatmaps_eigencam=grid_eigencam,
        heatmaps_gradcam=grid_gradcam,
        class_names=CLASS_NAMES,
        output_path=os.path.join('results', f'xai_grid_5grades_{ablation}_{timestamp}.png')
    )
    logger.info(f"5-grade XAI grid saved as results/xai_grid_5grades_{ablation}_{timestamp}.png")

    # --- Also generate single sample heatmap (legacy) ---
    if 0 in grade_samples:
        single_tensor, _ = grade_samples[0]
        single_tensor = single_tensor.to(device)
        img_np = single_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        generate_heatmap(single_tensor, model, img_np,
                         out_name=os.path.join('results', f'xai_heatmap_{ablation}_{timestamp}.png'))
        logger.info(f"Single XAI heatmap saved as results/xai_heatmap_{ablation}_{timestamp}.png")



def run_decoupled_retrain(config, ablation, device, logger, fold_idx=None, run_name=None):
    """Phase 2: freeze backbone+MSDA+HFF, re-train heads with class-balanced
    sampling for a short schedule. Loads the existing best checkpoint,
    saves to a distinct suffix so the original is never overwritten."""
    # Force class-balanced sampling
    decoupled_config_loader = {**config, "use_weighted_sampling": True}
    train_loader, val_loader, _, _, class_weights, class_priors, cls_num_list = get_dataloaders(
        decoupled_config_loader, fold_idx=fold_idx
    )
    run_name = run_name or ablation
    model = create_model(config, ablation, device)
    model_path = get_checkpoint_path(run_name, fold=fold_idx, create_dir=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.freeze_all_except_heads()

    decoupled_config = {
        **config,
        "epochs": config.get("decoupled_epochs", 12),
        "lr": config.get("decoupled_lr", 3e-5),
        "warmup_epochs": 1,
        "use_swa": False,          # short phase, SWA not meaningful here
        "patience": config.get("decoupled_epochs", 12),  # no early stop
    }
    criterion = create_criterion(decoupled_config, class_weights, device, class_priors=class_priors, cls_num_list=cls_num_list)
    trainer = DRTrainer(model, train_loader, val_loader, criterion, device,
                         decoupled_config, ablation=f"{run_name}__decoupled", fold_idx=fold_idx)
    trainer.train()

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

def main():
    parser = argparse.ArgumentParser(description="RetiNA-Net: Retinal DR Classification Project")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test', 'external_validation', 'xai',
                                 'detect_lesions', 'pretrain', 'generate_results', 'decoupled_retrain',
                                 'pretrain_eyepacs_supervised', 'train_vae', 'train_diffusion_unet',
                                 'generate_synthetic', 'generation_ablation'],
                        help="Execution mode.")
    parser.add_argument('--ablation', type=str, default='proposed',
                        choices=['baseline', 'msda_only', 'hff_only', 'proposed',
                                 'arch_baseline', 'arch_msda', 'arch_hff', 'arch_msda_hff',
                                 'arch_attnpool', 'arch_auxhead', 'arch_full'],
                        help="Ablation configuration for the model.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Path to config.yaml")
    parser.add_argument('--fold', type=int, default=None,
                        help="Train only a specific fold (0-indexed). If not specified, trains all folds.")
    parser.add_argument('--tag', type=str, default='', help='experiment tag; run name = <ablation>__<tag> (no spaces or slashes)')
    parser.add_argument('--inherit', type=str, default='', help='start from the saved config of run <ablation>__<INHERIT>')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides, e.g. --set use_sam=true sam_rho=0.05')

    args = parser.parse_args()

    config = load_config(args.config)
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
    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = get_log_path(args.mode, run_name, timestamp)

    logger = setup_logger(log_file=log_file_path)
    logger.info(f"Starting execution in mode: {args.mode}, run_name: {run_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.mode == 'train':
        snap_path = get_checkpoint_path(run_name, fold=None, create_dir=True).replace('model.pth', 'config_snapshot.yaml')
        snap_config = {k: v for k, v in config.items() if k != '_cli_overrides'}
        yaml.safe_dump(snap_config, open(snap_path, 'w'))

        use_kfold = config.get("use_kfold", True)

        if use_kfold:
            if args.fold is not None:
                # Train single fold
                logger.info(f"Training single fold: {args.fold}")
                train_losses, val_losses, train_accs, val_accs = train_single_fold(
                    config, args.ablation, device, logger, fold_idx=args.fold, run_name=run_name
                )
                plot_training_curves(
                    train_losses, val_losses, train_accs, val_accs,
                    filename=os.path.join('results', f'training_curves_{run_name}_fold{args.fold}_{timestamp}.png')
                )
            else:
                # Train all folds
                train_kfold(config, args.ablation, device, logger, timestamp, run_name=run_name)
        else:
            # Standard train/val split
            train_losses, val_losses, train_accs, val_accs = train_single_fold(
                config, args.ablation, device, logger, fold_idx=None, run_name=run_name
            )
            plot_training_curves(
                train_losses, val_losses, train_accs, val_accs,
                filename=os.path.join('results', f'training_curves_{run_name}_{timestamp}.png')
            )

        logger.info("Training complete.")

    elif args.mode == 'decoupled_retrain':
        decoupled_run_name = f"{run_name}__decoupled"
        snap_path = get_checkpoint_path(decoupled_run_name, fold=None, create_dir=True).replace('model.pth', 'config_snapshot.yaml')
        snap_config = {k: v for k, v in config.items() if k != '_cli_overrides'}
        yaml.safe_dump(snap_config, open(snap_path, 'w'))

        use_kfold = config.get("use_kfold", True)
        if use_kfold:
            if args.fold is not None:
                logger.info(f"Decoupled retrain single fold: {args.fold}")
                run_decoupled_retrain(config, args.ablation, device, logger, fold_idx=args.fold, run_name=run_name)
            else:
                n_folds = config.get("n_folds", 5)
                for fold_idx in range(n_folds):
                    logger.info(f"\n{'='*60}\nDecoupled retrain Fold {fold_idx + 1}/{n_folds}\n{'='*60}")
                    run_decoupled_retrain(config, args.ablation, device, logger, fold_idx=fold_idx, run_name=run_name)
        else:
            run_decoupled_retrain(config, args.ablation, device, logger, fold_idx=None, run_name=run_name)

    elif args.mode == 'test':
        run_test(config, args.ablation, device, logger, timestamp, run_name=run_name)

    elif args.mode == 'external_validation':
        run_external_validation(config, args.ablation, device, logger, timestamp, run_name=run_name)

    elif args.mode == 'xai':
        run_xai(config, args.ablation, device, logger, timestamp, run_name=run_name)

    elif args.mode == 'detect_lesions':
        run_detect_lesions(config, logger, timestamp)

    elif args.mode == 'pretrain':
        run_pretrain(config, device, logger, timestamp)

    elif args.mode == 'pretrain_eyepacs_supervised':
        run_pretrain_eyepacs_supervised(config, device, logger, timestamp)

    elif args.mode == 'generate_results':
        run_generate_results(config, logger, timestamp)

    elif args.mode == 'train_vae':
        from src.generation.train_vae import train_vae as run_train_vae
        run_train_vae(config, device, logger)

    elif args.mode == 'train_diffusion_unet':
        from src.generation.train_unet import train_diffusion_unet as run_train_unet
        run_train_unet(config, device, logger)

    elif args.mode == 'generate_synthetic':
        from src.generation.generate import generate_synthetic_images
        logger.info("Generating synthetic images...")
        # ... load models and call generate_synthetic_images ...
        pass

    elif args.mode == 'generation_ablation':
        logger.info("Running generation ablation...")
        # ... runs the 8-row table end-to-end and logs it ...
        pass


if __name__ == '__main__':
    main()

