import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # ⚠️ ADD: label smoothing was absent. With a small dataset and a
        # confident cross-entropy objective, the model can drive logits to
        # extreme confidence on training examples it has memorized — visible
        # here as train QWK/acc pulling well ahead of val while val loss
        # rises. Label smoothing caps target confidence at 1-label_smoothing,
        # directly discouraging that overfitting mode. 0.05 is conservative;
        # raise toward 0.1 if the train/val gap is still large next run.
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, weights=None):
        # inputs: logits, targets: class indices
        ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, class_weights, device, label_smoothing=0.05):
        super(CombinedLoss, self).__init__()
        
        # Mandatory device check: ensure class_weights is a FloatTensor and on the correct device
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.FloatTensor(class_weights)
        self.class_weights = class_weights.to(device)
        
        self.focal_loss = FocalLoss(gamma=2.0, label_smoothing=label_smoothing)
        
    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets, weights=self.class_weights)
