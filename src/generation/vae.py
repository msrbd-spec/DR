import torch
import torch.nn as nn

def lesion_weighted_recon_loss(recon, target, lesion_mask, l1_weight=1.0, lesion_boost=2.0):
    """
    Weight the L1/LPIPS reconstruction term by the LesionDetector mask.
    lesion_mask: (B, 3, H, W) or (B, 1, H, W)
    """
    pixel_weight = 1.0 + lesion_boost * lesion_mask.amax(dim=1, keepdim=True).clamp(0, 1)
    l1 = (recon - target).abs() * pixel_weight
    return l1_weight * l1.mean()

class DRVAE(nn.Module):
    """
    Wrapper for a VAE (kl-f8 style).
    Could wrap diffusers.AutoencoderKL or be a custom implementation.
    """
    def __init__(self, use_lesion_weighted_loss=False):
        super().__init__()
        self.use_lesion_weighted_loss = use_lesion_weighted_loss
        # Placeholder for actual VAE
        # self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        self.encoder = nn.Conv2d(3, 4, 3, padding=1)
        self.decoder = nn.Conv2d(4, 3, 3, padding=1)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
