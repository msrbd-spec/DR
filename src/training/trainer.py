import copy
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from tqdm import tqdm
from ..evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class ModelEMA:
    """
    Exponential moving average of model weights.

    Why: with a small, class-imbalanced val split, raw per-epoch weights are
    noisy — a handful of borderline predictions flipping is enough to swing
    accuracy/precision/recall by 10-20 points even when the model itself is
    fine (see QWK staying stable while accuracy dips). Evaluating/checkpointing
    on an EMA shadow model averages out that noise at the weight level instead
    of just hiding it in the metric, which is a strictly better fix.

    Uses a ramped decay (min(target_decay, (1+step)/(10+step))) rather than a
    fixed decay: with only ~160 optimizer steps/epoch on this dataset size, a
    static decay=0.999 has a ~700-step half-life — it would still be mostly
    at its random init after 4+ epochs, actively lagging the real model
    during exactly the short training window (~12-15 epochs before early
    stop) this project runs for. Ramping lets it track closely at the start
    and only smooth aggressively once there's enough signal to average over.
    """
    def __init__(self, model, decay: float = 0.999):
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        ema_params = dict(self.ema_model.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].mul_(d).add_(p.detach(), alpha=1 - d)
        ema_buffers = dict(self.ema_model.named_buffers())
        for name, b in model.named_buffers():
            ema_buffers[name].copy_(b)

class DRTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, device, config, ablation='proposed'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.epochs = config.get("epochs", 50)
        self.ablation = ablation
        
        lr = config.get("lr", 1e-4)
        weight_decay = config.get("weight_decay", 0.05)
        backbone_lr_mult = config.get("backbone_lr_mult", 0.1)

        # ⚠️ FIX: previously all params (pretrained backbone + randomly-init
        # MSDA/HFF/FC head) shared one LR. lr=1e-4 is too aggressive for a
        # pretrained SwinV2-Base (risks catastrophic forgetting of ImageNet
        # features on a ~3.6k-image dataset) while being the *only* signal
        # driving the freshly-initialized head. Split into two param groups
        # so the backbone fine-tunes gently and the head learns fast.
        backbone_params = [p for n, p in self.model.named_parameters() if n.startswith("backbone")]
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith("backbone")]

        self.optimizer = AdamW(
            [
                {"params": backbone_params, "lr": lr * backbone_lr_mult},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs - 5)
        # Assuming a custom implementation of Warmup is handled inside step, or simplified:
        # A 5-epoch linear warmup can be done manually or via chained schedulers.
        self.warmup_epochs = 5
        self.warmup_factor = 1.0 / self.warmup_epochs
        
        self.patience = config.get("patience", 10)
        self.accumulation_steps = config.get("accumulation_steps", 4)
        self.scaler = torch.cuda.amp.GradScaler()
        self.grad_clip_norm = config.get("grad_clip_norm", 1.0)

        self.use_ema = config.get("use_ema", True)
        self.ema = ModelEMA(self.model, decay=config.get("ema_decay", 0.999)) if self.use_ema else None

        self.best_val_loss = float('inf')
        self.best_val_qwk = -1.0
        self.epochs_without_improvement = 0

    def adjust_learning_rate(self, epoch):
        # ⚠️ FIX: this used to read param_groups[0]'s initial_lr and broadcast
        # it to ALL param groups, silently erasing the backbone/head LR split
        # for the entire 5-epoch warmup window. Each group now scales from its
        # own initial_lr instead.
        if epoch < self.warmup_epochs:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * (epoch + 1) * self.warmup_factor
        else:
            self.scheduler.step(epoch - self.warmup_epochs)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        # Save initial lr before adjustment if not saved
        if 'initial_lr' not in self.optimizer.param_groups[0]:
            for param_group in self.optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']
                
        self.adjust_learning_rate(epoch)
        
        self.optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                # Normalize loss to account for accumulation
                loss = loss / self.accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            # Perform optimization step only after accumulation_steps batches
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                # ⚠️ FIX: no gradient clipping previously existed. Fine-tuning a
                # pretrained transformer with a freshly-initialized head is prone
                # to occasional large gradient spikes (e.g. the epoch-4 val
                # loss/acc spike) — must unscale AMP-scaled grads before clipping
                # or the norm is computed on the wrong scale.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update(self.model)
            
            # Re-multiply by accumulation_steps to get the true batch loss for logging
            running_loss += loss.item() * self.accumulation_steps * inputs.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
        train_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_metrics(all_targets, all_preds)
        return train_loss, metrics

    def val_epoch(self, epoch, use_ema: bool = False):
        eval_model = self.ema.ema_model if (use_ema and self.use_ema) else self.model
        eval_model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = eval_model(inputs)
                    loss = self.criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss = running_loss / len(self.val_loader.dataset)
        metrics = compute_metrics(all_targets, all_preds)
        
        return val_loss, metrics

    def train(self):
        logger.info("Starting training...")
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(self.epochs):
            train_loss, train_metrics = self.train_epoch(epoch)
            # ⚠️ FIX: evaluate + select best checkpoint using the EMA shadow
            # model rather than the raw per-epoch weights. This is what
            # actually suppresses the epoch-to-epoch val accuracy/loss noise
            # (e.g. the epoch-4 spike) instead of just averaging it away in
            # a plot after the fact.
            val_loss, val_metrics = self.val_epoch(epoch, use_ema=True)
            
            val_qwk = val_metrics['qwk']
            val_acc = val_metrics['accuracy']
            train_acc = train_metrics['accuracy']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            logger.info(f"[Train] Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1_macro']:.4f}, QWK: {train_metrics['qwk']:.4f}")
            logger.info(f"[Val] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_macro']:.4f}, QWK: {val_qwk:.4f}")
            
            # Early Stopping and Checkpointing on QWK
            if val_qwk > self.best_val_qwk or (val_qwk == self.best_val_qwk and val_loss < self.best_val_loss):
                self.best_val_qwk = val_qwk
                self.best_val_loss = min(self.best_val_loss, val_loss)
                self.epochs_without_improvement = 0
                save_model = self.ema.ema_model if self.use_ema else self.model
                torch.save(save_model.state_dict(), f'best_iccit_model_{self.ablation}.pth')
                logger.info(f"New best model saved at epoch {epoch+1} with QWK {val_qwk:.4f}")
            else:
                self.epochs_without_improvement += 1
                
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
        return train_losses, val_losses, train_accs, val_accs
