import torch
import torch.nn as nn
import torchvision.ops as ops

class MSDABlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, deform_groups: int = 4):
        """
        Multi-Scale Deformable Attention Block wrapping DeformConv2d.

        ⚠️ FIX (was: num_offset_channels scaled with out_channels):
        In standard DCNv2, the offset/mask predictor size depends on
        `deform_groups` (how many independent 2D offset fields are learned),
        NOT on the number of feature channels. The previous implementation set
        num_offset_channels = 3*K^2*out_channels, which for out_channels=1024
        produced a 254M-parameter offset conv alone (larger than the entire
        SwinV2-Base backbone). That is both computationally wasteful and, with
        only ~3.6k training images, essentially untrainable — it acts mostly as
        noise injected into backbone gradients, which is consistent with the
        unstable/oscillating QWK observed in training.

        deform_groups=4 is a reasonable default: enough spatial flexibility
        to capture a few distinct lesion-shape deformation patterns per
        stage, while keeping the offset predictor lightweight (~hundreds of
        thousands of params instead of hundreds of millions).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.deform_groups = deform_groups

        # Offset (2 channels) + mask (1 channel) per deform group, per kernel tap.
        num_offset_channels = 3 * (kernel_size ** 2) * deform_groups

        self.offset_conv = nn.Conv2d(
            in_channels,
            num_offset_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True
        )
        
        # Initialize offsets to 0
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.deform_conv = ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )

    def forward(self, x):
        # Predict offsets and mask
        offset_and_mask = self.offset_conv(x)
        
        # Split into offsets (2 * K^2 * deform_groups) and mask (1 * K^2 * deform_groups)
        o1 = 2 * (self.kernel_size ** 2) * self.deform_groups
        o2 = 1 * (self.kernel_size ** 2) * self.deform_groups
        
        offsets, mask = torch.split(offset_and_mask, [o1, o2], dim=1)
        mask = torch.sigmoid(mask)
        
        out = self.deform_conv(x, offsets, mask=mask)
        return out


class HFFBlock(nn.Module):
    def __init__(self):
        """
        Hierarchical Feature Fusion Block.
        Bridges Stage 2 (48x48, 256c) into Stage 4 (12x12, 1024c).
        """
        super().__init__()
        # Projects 256 -> 1024, downsamples 48x48 -> 12x12
        self.projection = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=4,
            stride=4
        )

    def forward(self, stage2_features, stage4_features):
        projected_stage2 = self.projection(stage2_features)
        fused_features = torch.add(stage4_features, projected_stage2)
        return fused_features
