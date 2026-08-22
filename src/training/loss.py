import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing.

    Focal Loss downweights well-classified examples and focuses gradient
    signal on hard, minority-class examples — critical for the heavily
    imbalanced APTOS-2019 dataset.
    """

    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, weights=None):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=weights, reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class OrdinalLoss(nn.Module):
    """
    Ordinal regression loss using cumulative link model.

    For K classes, predicts K-1 cumulative logits P(y > k).
    Loss = sum of binary cross-entropy on each cumulative threshold.

    This enforces the ordinal structure of DR grades: misclassifying
    class 0 as class 4 is penalized more than class 2→3.
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, cum_logits, targets):
        """
        Args:
            cum_logits: (B, K-1) cumulative logits from ordinal head
            targets: (B,) integer labels in [0, K-1]
        """
        # Create binary targets: for each threshold k, target is 1 if y > k
        # targets shape: (B,) → (B, K-1)
        targets_expanded = targets.unsqueeze(1)  # (B, 1)
        thresholds = torch.arange(self.num_classes - 1, device=targets.device).unsqueeze(0)  # (1, K-1)
        binary_targets = (targets_expanded > thresholds).float()  # (B, K-1)

        # Binary cross-entropy with logits
        loss = F.binary_cross_entropy_with_logits(cum_logits, binary_targets, reduction='mean')
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Focal Loss (weighted) + Ordinal Loss.

    The classification head is trained with Focal Loss + class weights.
    The ordinal head is trained with OrdinalLoss.
    Total loss = focal_loss + ordinal_weight * ordinal_loss.

    The model forward() returns a dict with 'logits' and 'ordinal_logits'.
    This loss function handles both.
    """

    def __init__(self, class_weights, device, label_smoothing=0.1,
                 use_ordinal=True, ordinal_loss_weight=0.3, num_classes=5):
        super(CombinedLoss, self).__init__()

        # Device placement for class weights
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.FloatTensor(class_weights)
        self.class_weights = class_weights.to(device)

        self.focal_loss = FocalLoss(gamma=2.0, label_smoothing=label_smoothing)
        self.use_ordinal = use_ordinal

        if use_ordinal:
            self.ordinal_loss = OrdinalLoss(num_classes=num_classes)
            self.ordinal_loss_weight = ordinal_loss_weight

    def forward(self, pred, targets):
        """
        Args:
            pred: dict with 'logits' (B, K) and 'ordinal_logits' (B, K-1) or None
            targets: (B,) integer labels

        Returns:
            Total combined loss
        """
        # Handle both dict (new model) and tensor (old model) predictions
        if isinstance(pred, dict):
            logits = pred['logits']
            ordinal_logits = pred.get('ordinal_logits', None)
        else:
            logits = pred
            ordinal_logits = None

        # Focal loss with class weights
        cls_loss = self.focal_loss(logits, targets, weights=self.class_weights)

        total_loss = cls_loss

        # Ordinal loss
        if self.use_ordinal and ordinal_logits is not None:
            ord_loss = self.ordinal_loss(ordinal_logits, targets)
            total_loss = total_loss + self.ordinal_loss_weight * ord_loss

        return total_loss
