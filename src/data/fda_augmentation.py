"""
Fourier Domain Adaptation (FDA) — Yang & Soatto, CVPR 2020.

Swaps the low-frequency amplitude spectrum of a source image with that of
a target-domain image, keeping the source's phase spectrum intact. This
transfers low-level domain "style" (illumination, camera color response)
without altering content/structure — critical here since lesion geometry
must be preserved.
"""

import numpy as np
import cv2


def fda_source_to_target(source_img: np.ndarray, target_img: np.ndarray, beta: float = 0.01) -> np.ndarray:
    """
    Args:
        source_img: HxWx3 uint8/float array (the APTOS training image,
                    after the existing crop/CLAHE/Ben-Graham preprocessing)
        target_img: HxWx3 array (a sampled Messidor-2 style-pool image,
                    same preprocessing pipeline applied beforehand)
        beta: fraction of the spectrum (centered, low-frequency) to swap.
              Small values (0.01-0.09) are standard — too large destroys
              structure.

    Returns:
        HxWx3 uint8 array — source content, target low-frequency style.
    """
    h, w = source_img.shape[:2]
    target_resized = cv2.resize(target_img, (w, h))

    src = source_img.transpose(2, 0, 1).astype(np.float32)
    trg = target_resized.transpose(2, 0, 1).astype(np.float32)

    fft_src = np.fft.fft2(src, axes=(-2, -1))
    fft_trg = np.fft.fft2(trg, axes=(-2, -1))

    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
    amp_trg = np.abs(fft_trg)

    amp_src_shift = np.fft.fftshift(amp_src, axes=(-2, -1))
    amp_trg_shift = np.fft.fftshift(amp_trg, axes=(-2, -1))

    b = int(np.floor(min(h, w) * beta))
    cy, cx = h // 2, w // 2
    amp_src_shift[:, cy - b:cy + b, cx - b:cx + b] = amp_trg_shift[:, cy - b:cy + b, cx - b:cx + b]

    amp_mixed = np.fft.ifftshift(amp_src_shift, axes=(-2, -1))
    fft_mixed = amp_mixed * np.exp(1j * pha_src)
    mixed = np.real(np.fft.ifft2(fft_mixed, axes=(-2, -1)))
    mixed = np.clip(mixed, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    return mixed
