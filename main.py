import argparse
import yaml
import torch
import logging
from src.utils.logger import setup_logger
from src.data.datamodule import get_dataloaders
from src.models.dr_model import ICCIT_DR_Net
from src.training.loss import CombinedLoss
from src.training.trainer import DRTrainer
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizer import plot_training_curves, plot_confusion_matrix, plot_roc_curve
from src.evaluation.xai import generate_heatmap
from src.evaluation.tta import predict_with_tta
import cv2
import numpy as np

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

def main():
    parser = argparse.ArgumentParser(description="ICCIT DR Classification Project")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'external_validation', 'xai'],
                        help="Execution mode.")
    parser.add_argument('--ablation', type=str, default='proposed',
                        choices=['baseline', 'msda_only', 'hff_only', 'proposed'],
                        help="Ablation configuration for the model.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Path to config.yaml")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Dynamically create logs and results directories to prevent overwrites
    import os
    import datetime
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join('logs', f'{args.mode}_{args.ablation}_{timestamp}.log')
    
    logger = setup_logger(log_file=log_file_path)
    logger.info(f"Starting execution in mode: {args.mode}, ablation: {args.ablation}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    use_msda, use_hff = get_ablation_flags(args.ablation)
    model = ICCIT_DR_Net(
        use_msda=use_msda,
        use_hff=use_hff,
        num_classes=5,
        drop_path_rate=config.get("drop_path_rate", 0.2)
    ).to(device)
    
    if args.mode == 'train':
        train_loader, val_loader, _, _, class_weights = get_dataloaders(config)
        criterion = CombinedLoss(class_weights=class_weights, device=device, label_smoothing=config.get("label_smoothing", 0.05))
        trainer = DRTrainer(model, train_loader, val_loader, criterion, device, config, ablation=args.ablation)
        
        train_losses, val_losses, train_accs, val_accs = trainer.train()
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, filename=os.path.join('results', f'training_curves_{args.ablation}_{timestamp}.png'))
        logger.info("Training complete.")
        
    elif args.mode == 'test':
        _, _, test_loader, _, _ = get_dataloaders(config)
        model.load_state_dict(torch.load(f'best_iccit_model_{args.ablation}.pth', map_location=device))
        model.eval()
        
        all_preds, all_targets, all_probs = [], [], []
        use_tta = config.get("use_tta", True)
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                probs, preds = predict_with_tta(model, inputs, use_tta=use_tta)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())
                
        metrics = compute_metrics(all_targets, all_preds)
        
        logger.info(f"Test Metrics - Acc: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_macro']:.4f}, QWK: {metrics['qwk']:.4f}")
        logger.info(f"Test Classification Report:\n{metrics['classification_report']}")
        
        plot_confusion_matrix(all_targets, all_preds, filename=os.path.join('results', f'confusion_matrix_{args.ablation}_{timestamp}.png'))
        plot_roc_curve(all_targets, np.array(all_probs), filename=os.path.join('results', f'roc_multiclass_{args.ablation}_{timestamp}.png'))
        
    elif args.mode == 'external_validation':
        _, _, _, ext_loader, _ = get_dataloaders(config)
        model.load_state_dict(torch.load(f'best_iccit_model_{args.ablation}.pth', map_location=device))
        model.eval()
        
        all_preds, all_targets, all_probs = [], [], []
        use_tta = config.get("use_tta", True)
        with torch.no_grad():
            for inputs, targets in ext_loader:
                inputs = inputs.to(device)
                probs, preds = predict_with_tta(model, inputs, use_tta=use_tta)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())
                
        metrics = compute_metrics(all_targets, all_preds)
        
        logger.info(f"External Validation Metrics - Acc: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_macro']:.4f}, QWK: {metrics['qwk']:.4f}")
        logger.info(f"External Validation Classification Report:\n{metrics['classification_report']}")
        
        plot_confusion_matrix(all_targets, all_preds, filename=os.path.join('results', f'confusion_matrix_external_{args.ablation}_{timestamp}.png'))
        
    elif args.mode == 'xai':
        _, _, test_loader, _, _ = get_dataloaders(config)
        model.load_state_dict(torch.load(f'best_iccit_model_{args.ablation}.pth', map_location=device))
        
        # Get one sample from test_loader
        inputs, targets = next(iter(test_loader))
        single_tensor = inputs[0:1].to(device)
        
        # For visualization, we need the original image. We can approx it from tensor by denormalizing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = single_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        generate_heatmap(single_tensor, model, img_np, out_name=os.path.join('results', f'xai_heatmap_{args.ablation}_{timestamp}.png'))
        logger.info(f"XAI heatmap generated as results/xai_heatmap_{args.ablation}_{timestamp}.png")

if __name__ == '__main__':
    main()
