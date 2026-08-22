import torch
import torch.nn as nn
import timm

from .components import MSDABlock, HFFBlock, MultiScaleHead, OrdinalRegressionHead


class ICCIT_DR_Net(nn.Module):
    """
    ICCIT DR Classification Network.

    Architecture:
      - Backbone: SwinV2-Base (pretrained, features_only)
      - MSDA: Multi-Scale Deformable Attention on Stage 3 & 4
      - HFF: Hierarchical Feature Fusion (Stage 1 → Stage 4, all 4 stages)
      - Head: Multi-scale pooling + 2-layer MLP
      - Ordinal head: Optional ordinal regression head for ordinal-aware loss

    The model returns a dict with:
      - 'logits': classification logits (B, num_classes)
      - 'ordinal_logits': ordinal cumulative logits (B, num_classes-1) or None
    """

    def __init__(self, use_msda: bool = True, use_hff: bool = True,
                 num_classes: int = 5, drop_path_rate: float = 0.3,
                 dropout: float = 0.5, use_ordinal: bool = True,
                 backbone_name: str = 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'):
        super().__init__()
        self.use_msda = use_msda
        self.use_hff = use_hff
        self.use_ordinal = use_ordinal
        self.num_classes = num_classes

        # Backbone: SwinV2-Base with dynamic image size for 384x384 input
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            dynamic_img_size=True,
            img_size=384,
            window_size=12,
            drop_path_rate=drop_path_rate
        )

        if self.use_msda:
            self.msda3 = MSDABlock(in_channels=512, kernel_size=3)
            self.msda4 = MSDABlock(in_channels=1024, kernel_size=3)

        if self.use_hff:
            self.hff = HFFBlock()

        # Multi-scale classification head (dual pooling + 2-layer MLP)
        self.head = MultiScaleHead(
            in_channels=1024,
            num_classes=num_classes,
            dropout=dropout
        )

        # Optional ordinal regression head
        if self.use_ordinal:
            self.ordinal_head = OrdinalRegressionHead(
                in_channels=1024,
                num_classes=num_classes,
                dropout=dropout
            )

    def forward(self, x):
        features = self.backbone(x)
        stage1 = features[0]
        stage2 = features[1]
        stage3 = features[2]
        stage4 = features[3]

        # Swin Transformers return features in (B, H, W, C) format.
        # Permute to (B, C, H, W) if the last dimension matches expected channel count.
        if stage4.shape[-1] == 1024:
            stage1 = stage1.permute(0, 3, 1, 2).contiguous()
            stage2 = stage2.permute(0, 3, 1, 2).contiguous()
            stage3 = stage3.permute(0, 3, 1, 2).contiguous()
            stage4 = stage4.permute(0, 3, 1, 2).contiguous()

        if self.use_msda:
            stage3 = self.msda3(stage3)
            stage4 = self.msda4(stage4)

        if self.use_hff:
            fused_features = self.hff(stage1, stage2, stage3, stage4)
        else:
            fused_features = stage4

        # Classification logits
        logits = self.head(fused_features)

        # Ordinal logits (if enabled)
        ordinal_logits = None
        if self.use_ordinal:
            ordinal_logits = self.ordinal_head(fused_features)

        return {
            'logits': logits,
            'ordinal_logits': ordinal_logits
        }

    def get_classification_logits(self, x):
        """Convenience method returning only classification logits."""
        return self.forward(x)['logits']

    def predict(self, x):
        """
        Inference method returning softmax probabilities.
        Combines classification and ordinal heads for final prediction.
        """
        out = self.forward(x)
        cls_probs = torch.softmax(out['logits'], dim=1)

        if self.use_ordinal and out['ordinal_logits'] is not None:
            ord_probs = OrdinalRegressionHead.ordinal_logits_to_class_probs(out['ordinal_logits'])
            # Average the two probability distributions
            final_probs = 0.5 * cls_probs + 0.5 * ord_probs
        else:
            final_probs = cls_probs

        return final_probs
