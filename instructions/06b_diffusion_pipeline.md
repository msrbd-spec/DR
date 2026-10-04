# P6b — Fix: Complete the Diffusion Pipeline (diffusers-based)

## Context
The current `src/generation/` implementation is non-functional beyond
stubs — not just missing training loops. Confirmed by direct inspection:
- `diffusion.py`: `q_sample` is wrong (missing the `√ᾱ_t·x₀ + √(1-ᾱ_t)·ε`
  formula), `ddim_sample` is `pass`
- `vae.py`: `DRVAE` is a single `Conv2d` encoder + single `Conv2d`
  decoder — no KL regularization, no reparameterization trick
- `unet.py`: 3-block toy network, `time_emb` is computed but never used
  in `forward()`, no skip connections
- `generate.py`: `generate_synthetic_images` returns `torch.randn(...)`
  — literal noise
- `dataset.py`: `__getitem__` returns `torch.randn(3,512,512)` — doesn't
  load real images or real lesion masks

## Directive: use HuggingFace `diffusers` for all non-novel machinery
Do **not** hand-derive DDPM/DDIM math, VAE reparameterization, or write a
U-Net's down/up/skip structure from scratch. Use `diffusers`' tested
primitives for everything that is NOT one of the four novel pieces below.
This is a deliberate engineering choice, not a shortcut that reduces
novelty — the paper's actual novelty claims stay fully custom.

**Add to `requirements.txt`:**
```
diffusers>=0.27.0
accelerate>=0.30.0
```

## The four pieces that MUST stay custom (do not replace with diffusers equivalents)
1. Lesion-mask spatial conditioning (extra input channels to the U-Net,
   built from `LesionDetector` output)
2. Classifier-in-the-loop filtering (`generate.py`'s `filter_synthetic_batch`
   — already correctly implemented, keep as-is)
3. Spectral/FFT fidelity loss (`spectral_loss.py` — already correctly
   implemented, keep as-is)
4. Lesion-weighted VAE reconstruction loss (`vae.py`'s
   `lesion_weighted_recon_loss` — already correctly implemented, keep
   as-is, just needs a real VAE to attach it to)

---

## Fix 1: `src/generation/vae.py`

Replace `DRVAE` entirely with a wrapper around `diffusers.AutoencoderKL`,
loaded from a **pretrained** checkpoint (matches DR-LDS's own protocol —
their paper explicitly starts from a pretrained "kl-f8" checkpoint, not a
VAE trained from random initialization):

```python
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


def lesion_weighted_recon_loss(recon, target, lesion_mask, l1_weight=1.0, lesion_boost=2.0):
    """Unchanged — keep this function exactly as-is."""
    pixel_weight = 1.0 + lesion_boost * lesion_mask.amax(dim=1, keepdim=True).clamp(0, 1)
    l1 = (recon - target).abs() * pixel_weight
    return l1_weight * l1.mean()


class DRVAE(nn.Module):
    """
    Wraps a pretrained diffusers AutoencoderKL (kl-f8, 8x spatial
    compression, 4 latent channels at 512px input -> 64x64x4 latent —
    matches DR-LDS's reported compression factor). Fine-tuned on APTOS
    with the standard reconstruction+KL loss, optionally boosted by
    lesion_weighted_recon_loss on lesion-containing pixels (our novelty).
    """

    def __init__(self, pretrained_path="stabilityai/sd-vae-ft-mse", use_lesion_weighted_loss=False):
        super().__init__()
        self.use_lesion_weighted_loss = use_lesion_weighted_loss
        # pretrained_path must point at a LOCAL cache dir on offline
        # clusters — see "Offline cluster" section below.
        self.vae = AutoencoderKL.from_pretrained(pretrained_path)

    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, z):
        return self.vae.decode(z / self.vae.config.scaling_factor).sample

    def forward(self, x):
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample()
        recon = self.vae.decode(z).sample
        return recon, z, posterior  # posterior needed for KL term in train_vae.py
```

## Fix 2: `src/generation/train_vae.py` — real training loop
```python
import os
import torch
from torch.utils.data import DataLoader
from .vae import DRVAE, lesion_weighted_recon_loss
from .dataset import DRDiffusionDataset


def train_vae(config, device, logger):
    logger.info("Starting VAE training...")
    use_lesion_loss = config.get("vae_use_lesion_weighted_loss", False)
    epochs = config.get("vae_epochs", 50)
    pretrained_path = config.get("vae_pretrained_path", "stabilityai/sd-vae-ft-mse")

    dataset = DRDiffusionDataset.from_config(config)  # see Fix 4
    loader = DataLoader(dataset, batch_size=config.get("vae_batch_size", 8),
                         shuffle=True, num_workers=4, drop_last=True)

    model = DRVAE(pretrained_path=pretrained_path, use_lesion_weighted_loss=use_lesion_loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("vae_lr", 1e-5))
    kl_weight = config.get("vae_kl_weight", 1e-6)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, labels, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            recon, z, posterior = model(imgs)

            if use_lesion_loss:
                recon_loss = lesion_weighted_recon_loss(recon, imgs, masks)
            else:
                recon_loss = torch.nn.functional.l1_loss(recon, imgs)

            kl_loss = posterior.kl().mean()
            loss = recon_loss + kl_weight * kl_loss

            loss.backward()
            optimizer.step()
            running += loss.item()

        logger.info(f"[VAE] Epoch {epoch+1}/{epochs} Loss: {running/len(loader):.4f}")

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            os.makedirs('checkpoints/generation', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/generation/vae.pth')

    logger.info("VAE training complete.")
```

## Fix 3: `src/generation/unet.py` — rebuild on `diffusers.UNet2DModel`
```python
import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ConditionalUNet(nn.Module):
    """
    Wraps diffusers.UNet2DModel for the proven down/up/skip/attention
    backbone + native class conditioning (num_class_embeds). Lesion-mask
    spatial conditioning (our novelty) is added as extra input channels,
    concatenated before the first conv — UNet2DModel's in_channels is
    simply set to account for them.
    """

    def __init__(self, base_ch=128, use_lesion_conditioning=False,
                 use_ordinal_conditioning=False, use_rep_blocks=False):
        super().__init__()
        self.use_lesion_conditioning = use_lesion_conditioning
        self.use_ordinal_conditioning = use_ordinal_conditioning

        in_ch = 4 + (3 if use_lesion_conditioning else 0)  # 4 = VAE latent channels

        self.model = UNet2DModel(
            sample_size=64,              # latent resolution at 512px input / 8x VAE
            in_channels=in_ch,
            out_channels=4,              # predict noise in latent space (4 channels)
            layers_per_block=2,
            block_out_channels=(base_ch, base_ch * 2, base_ch * 4, base_ch * 4),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            num_class_embeds=None if use_ordinal_conditioning else 5,
        )

        if use_ordinal_conditioning:
            # Monotonic severity embedding replacing UNet2DModel's default
            # discrete class-embedding table — projects normalized grade
            # (y/4) to the same conditioning dim UNet2DModel expects.
            self.ordinal_proj = nn.Sequential(
                nn.Linear(1, base_ch), nn.SiLU(), nn.Linear(base_ch, base_ch * 4)
            )

        # use_rep_blocks (P5's RepProjConv): NOT wired into UNet2DModel's
        # internal ResBlocks for this fix — treat as a separate, later
        # ablation if pursued (UNet2DModel's blocks aren't trivially
        # swappable without subclassing diffusers internals). Leave
        # use_rep_blocks as a no-op flag for now; log a warning if True.

    def forward(self, x, t, class_labels, lesion_mask=None):
        if self.use_lesion_conditioning and lesion_mask is not None:
            x = torch.cat([x, lesion_mask], dim=1)

        if self.use_ordinal_conditioning:
            class_emb = self.ordinal_proj(class_labels.float().unsqueeze(-1) / 4.0)
            return self.model(x, t, class_labels=None, encoder_hidden_states=None,
                               class_emb_override=class_emb).sample  # see note below
        else:
            return self.model(x, t, class_labels=class_labels).sample
```
**Note on ordinal conditioning:** `UNet2DModel` doesn't natively accept a
pre-computed class embedding override — if `use_ordinal_conditioning` is
pursued, either (a) subclass `UNet2DModel` to accept an injected embedding
in place of its internal `class_embedding` lookup, or (b) treat ordinal
conditioning as deferred (it's already marked "extended/if time permits"
in `06_diffusion_pipeline.md`) and ship with `num_class_embeds=5` (plain
discrete conditioning) first.

## Fix 4: `src/generation/dataset.py` — load real data
```python
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.preprocessing.lesion_detection import LesionDetector
from src.data.dataset import DRDataset  # reuse crop/CLAHE/Ben-Graham


class DRDiffusionDataset(Dataset):
    """APTOS train-split images + LesionDetector masks, for VAE/U-Net training."""

    def __init__(self, image_dir, labels_df, img_size=512, mask_size=64):
        self.helper = DRDataset(image_dir=image_dir, labels_df=labels_df, transform=None, img_size=img_size)
        self.labels_df = labels_df
        self.detector = LesionDetector(mask_size=mask_size)
        self.img_size = img_size

    @classmethod
    def from_config(cls, config):
        import pandas as pd
        paths = config["dataset_paths"]
        df = pd.read_csv(paths["train_csv"])
        return cls(image_dir=paths["train_images"], labels_df=df, img_size=config.get("img_size", 512))

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img = self.helper._load_image(str(row['id_code']))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = self.helper._crop_fundus(img)
        img = self.helper._apply_clahe(img)
        img = self.helper._apply_ben_graham(img)

        lesion_result = self.detector.detect_all(img)  # mask: (mask_size, mask_size, 3)

        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5 - 1.0  # [-1, 1]

        mask_tensor = torch.from_numpy(lesion_result['mask']).permute(2, 0, 1).float()

        label = int(row['diagnosis'])
        return img_tensor, label, mask_tensor
```

## Fix 5: `src/generation/train_unet.py` — real DDPM training loop via diffusers scheduler
```python
import os
import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from .unet import ConditionalUNet
from .vae import DRVAE
from .dataset import DRDiffusionDataset
from .spectral_loss import spectral_loss


def train_diffusion_unet(config, device, logger):
    logger.info("Starting Conditional U-Net training...")
    use_spectral = config.get("use_spectral_loss", False)
    spectral_weight = config.get("spectral_loss_weight", 0.1)
    epochs = config.get("unet_epochs", 150)

    dataset = DRDiffusionDataset.from_config(config)
    loader = DataLoader(dataset, batch_size=config.get("unet_batch_size", 16),
                         shuffle=True, num_workers=4, drop_last=True)

    vae = DRVAE(pretrained_path=config.get("vae_pretrained_path", "stabilityai/sd-vae-ft-mse")).to(device)
    vae.load_state_dict(torch.load('checkpoints/generation/vae.pth', map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    unet = ConditionalUNet(
        base_ch=config.get("unet_base_channels", 128),
        use_lesion_conditioning=config.get("unet_use_lesion_conditioning", False),
        use_ordinal_conditioning=config.get("unet_use_ordinal_conditioning", False),
        use_rep_blocks=config.get("unet_use_rep_blocks", False),
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    for epoch in range(epochs):
        unet.train()
        running = 0.0
        for imgs, labels, masks in loader:
            imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                latents = vae.encode(imgs)
                lesion_cond = torch.nn.functional.interpolate(masks, size=latents.shape[-2:], mode='nearest') \
                    if config.get("unet_use_lesion_conditioning", False) else None

            noise = torch.randn_like(latents)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, t)

            pred_noise = unet(noisy_latents, t, labels, lesion_mask=lesion_cond)

            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            if use_spectral:
                loss = loss + spectral_weight * spectral_loss(pred_noise, noise)

            loss.backward()
            optimizer.step()
            running += loss.item()

        logger.info(f"[U-Net] Epoch {epoch+1}/{epochs} Loss: {running/len(loader):.4f}")

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            os.makedirs('checkpoints/generation', exist_ok=True)
            torch.save(unet.state_dict(), 'checkpoints/generation/unet.pth')

    logger.info("U-Net training complete.")
```

## Fix 6: `src/generation/diffusion.py` + `generate.py` — real DDIM sampling via diffusers
Replace `GaussianDiffusion` entirely — use `diffusers.DDIMScheduler`
directly in the generation function instead of a custom class:
```python
# src/generation/generate.py — replace generate_synthetic_images
import torch
from diffusers import DDIMScheduler


def generate_synthetic_images(unet, vae, class_labels, lesion_masks=None,
                               device='cuda', steps=50, guidance_scale=3.0):
    unet.eval()
    vae.eval()
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(steps)

    B = len(class_labels)
    latents = torch.randn(B, 4, 64, 64, device=device)
    class_labels = class_labels.to(device)
    lesion_masks = lesion_masks.to(device) if lesion_masks is not None else None

    with torch.no_grad():
        for t in scheduler.timesteps:
            t_batch = t.repeat(B).to(device)
            noise_pred = unet(latents, t_batch, class_labels, lesion_mask=lesion_masks)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        images = vae.decode(latents)
        images = (images.clamp(-1, 1) + 1) / 2 * 255  # back to [0, 255]

    return images
```
Delete `diffusion.py` (`GaussianDiffusion`) entirely — `DDPMScheduler`
(training, Fix 5) and `DDIMScheduler` (sampling, above) replace it fully.
Update `filter_synthetic_batch` in the same file — keep it exactly as
currently implemented, no changes needed there.

---

## Offline cluster — pretrained VAE checkpoint
`main.py`/`test_model.py` already set `HF_HUB_OFFLINE=1`/
`TRANSFORMERS_OFFLINE=1` (cluster has no internet, same as the existing
SwinV2 backbone setup). Before running `train_vae`:

1. **On a machine WITH internet access**, run once:
   ```bash
   python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').save_pretrained('./vae_pretrained_cache')"
   ```
2. Transfer the resulting `./vae_pretrained_cache` directory to the
   offline cluster (same way the SwinV2 weights are already cached there).
3. Set `vae_pretrained_path: "./vae_pretrained_cache"` (a **local path**,
   not the HuggingFace Hub ID) in `configs/config_diffusion.yaml`.

No download is needed for the U-Net — it trains from random
initialization (matches DR-LDS's own protocol; no public pretrained
checkpoint is conditioned on our specific 5 DR grades anyway).

---

## `configs/config_diffusion.yaml` — add
```yaml
vae_pretrained_path: "./vae_pretrained_cache"   # local path on the offline cluster
vae_batch_size: 8
vae_lr: 1.0e-5
vae_kl_weight: 1.0e-6
unet_batch_size: 16
```

## Validation steps
1. `pip install diffusers accelerate` — confirm import succeeds.
2. Pre-download + cache the VAE checkpoint as above.
3. Run `train_vae` for 2-3 epochs on a small subset, confirm loss
   decreases and `checkpoints/generation/vae.pth` is written.
4. Run `train_diffusion_unet` for a few epochs the same way, confirm loss
   decreases.
5. Run `generate_synthetic_images` with the resulting checkpoints, visually
   inspect a handful of outputs — confirm they look like retinal fundus
   images, not noise.
6. Only then proceed to the full `generation_ablation` 8-row sweep.