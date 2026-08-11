import torch
import torch.nn as nn
import timm

from .components import MSDABlock, HFFBlock

class ICCIT_DR_Net(nn.Module):
    def __init__(self, use_msda: bool = True, use_hff: bool = True, num_classes: int = 5, drop_path_rate: float = 0.2):
        super().__init__()
        self.use_msda = use_msda
        self.use_hff = use_hff
        
        # ⚠️ Critical Implementation Note — Window/Position-Embedding Resizing
        # Must include dynamic_img_size=True to avoid shape-mismatch error.
        #
        # drop_path_rate (stochastic depth) was previously left at timm's
        # default of 0.0 — no regularization inside the backbone at all
        # while fully fine-tuning an 88M-param pretrained transformer on
        # ~2.5k images. That's a direct contributor to the train/val QWK
        # gap (0.95+ train vs ~0.91 val plateau) — the backbone has more
        # than enough capacity to memorize a dataset this size unless
        # regularized. 0.2 is a standard fine-tuning value for Swin-family
        # models on small-to-mid datasets.
        self.backbone = timm.create_model(
            'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k', 
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
            
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        stage1 = features[0]
        stage2 = features[1]
        stage3 = features[2]
        stage4 = features[3]
        
        # Swin Transformers often return features in (B, H, W, C) format.
        # If the last dimension matches the expected channel count, permute to (B, C, H, W)
        if stage4.shape[-1] == 1024:
            stage1 = stage1.permute(0, 3, 1, 2).contiguous()
            stage2 = stage2.permute(0, 3, 1, 2).contiguous()
            stage3 = stage3.permute(0, 3, 1, 2).contiguous()
            stage4 = stage4.permute(0, 3, 1, 2).contiguous()
        
        if self.use_msda:
            stage3 = self.msda3(stage3)
            stage4 = self.msda4(stage4)
            
        if self.use_hff:
            fused_features = self.hff(stage2, stage4)
        else:
            fused_features = stage4
            
        out = self.pool(fused_features)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
