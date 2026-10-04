import torch
import torch.nn as nn
from src.models.components import RepProjConv

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net with AdaGN time+class conditioning, 
    lesion-mask spatial conditioning, ordinal-aware conditioning, 
    and optional RepConv-reparameterized ResBlocks.
    """
    def __init__(self, 
                 base_ch=128, 
                 use_lesion_conditioning=False, 
                 use_ordinal_conditioning=False,
                 use_rep_blocks=False):
        super().__init__()
        self.use_lesion_conditioning = use_lesion_conditioning
        self.use_ordinal_conditioning = use_ordinal_conditioning
        self.use_rep_blocks = use_rep_blocks

        # Latent channels = 4
        in_ch = 4
        if self.use_lesion_conditioning:
            # +3 for the lesion mask (microaneurysm/hemorrhage/exudate)
            in_ch += 3
        
        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Class conditioning
        if self.use_ordinal_conditioning:
            # Monotonic severity embedding (e.g. 1-D scalar-to-vector mapping)
            self.class_emb = nn.Sequential(
                nn.Linear(1, 128),
                nn.SiLU(),
                nn.Linear(128, 512)
            )
        else:
            self.class_emb = nn.Embedding(5, 512)

        # Time conditioning
        self.time_emb = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )

        # Example blocks
        self.down1 = self._make_block(base_ch, base_ch*2)
        self.mid = self._make_block(base_ch*2, base_ch*2) # Bottleneck
        self.up1 = self._make_block(base_ch*2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 4, 3, padding=1)

    def _make_block(self, in_c, out_c):
        if self.use_rep_blocks:
            return RepProjConv(in_c, out_c)
        else:
            return nn.Conv2d(in_c, out_c, 3, padding=1)

    def forward(self, x, t, class_labels, lesion_mask=None):
        if self.use_lesion_conditioning and lesion_mask is not None:
            # lesion_mask should be downsampled to latent resolution (64x64)
            x = torch.cat([x, lesion_mask], dim=1)
        
        h = self.input_conv(x)

        if self.use_ordinal_conditioning:
            # Normalize class_labels to [0,1]
            c_emb = self.class_emb(class_labels.float().unsqueeze(-1) / 4.0)
        else:
            c_emb = self.class_emb(class_labels)

        # Placeholder forward pass
        h = self.down1(h)
        h = self.mid(h)
        h = self.up1(h)
        out = self.out_conv(h)
        return out
