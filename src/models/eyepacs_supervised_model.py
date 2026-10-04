import torch
import torch.nn as nn
import timm


class EyePACSSupervisedModel(nn.Module):
    """Plain SwinV2 backbone + linear head, trained with standard
    supervised CE on EyePACS's own DR labels (trainLabels.csv). Produces
    the 'EyePACS supervised' row for the SSL pretraining ablation table —
    the counterfactual to our self-supervised (contrastive+multitask) SSL."""

    def __init__(self, backbone_name, num_classes=5, drop_path_rate=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True,
            dynamic_img_size=True, img_size=512, drop_path_rate=drop_path_rate
        )
        stage4_channels = self.backbone.feature_info[-1]['num_chs']
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage4_channels, num_classes)

    def forward(self, x):
        stage4 = self.backbone(x)[-1]
        if stage4.shape[-1] == self.fc.in_features:
            stage4 = stage4.permute(0, 3, 1, 2).contiguous()
        pooled = self.pool(stage4).flatten(1)
        return self.fc(pooled)

    def save_backbone(self, path):
        """Save ONLY backbone weights — same format SSLModel.save_backbone()
        produces, so it's drop-in compatible with config's ssl_pretrained_path."""
        torch.save(self.backbone.state_dict(), path)
