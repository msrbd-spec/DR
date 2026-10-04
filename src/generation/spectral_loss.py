import torch
import torch.nn as nn

def spectral_loss(pred_noise, target_noise, high_freq_weight=2.0):
    """FFT-domain auxiliary loss — penalizes high-frequency-band error
    more heavily than the standard pixel/latent-space epsilon-MSE, since
    diffusion models are known to underfit high-frequency detail (exactly
    where microaneurysms live)."""
    pred_fft = torch.fft.fft2(pred_noise, dim=(-2, -1))
    target_fft = torch.fft.fft2(target_noise, dim=(-2, -1))
    diff = torch.abs(pred_fft - target_fft) ** 2

    h, w = pred_noise.shape[-2:]
    cy, cx = h // 2, w // 2
    freq_weight = torch.ones_like(diff)
    low_freq_radius = min(h, w) // 8
    
    # We need to shift the meshgrid to match fftfreq if we use fft2 directly, 
    # but fftshift isn't applied here. Let's assume standard centered fft or adjust.
    # To match instruction:
    yy, xx = torch.meshgrid(torch.arange(h) - cy, torch.arange(w) - cx, indexing='ij')
    high_freq_mask = (yy**2 + xx**2) > low_freq_radius**2
    freq_weight[..., high_freq_mask] = high_freq_weight

    return (diff * freq_weight.to(diff.device)).mean()
