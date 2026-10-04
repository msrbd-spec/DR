# P4 — Fourier Domain Adaptation (External Generalization)

## Objective
Hit the 90%+ external (Messidor-2) validation target by reducing the
domain gap (camera/illumination differences between APTOS-19 and
Messidor-2) during training. Technique: FDA (Yang & Soatto, CVPR 2020) —
swap the low-frequency amplitude spectrum of a source (APTOS) training
image with that of a randomly sampled target-domain (Messidor-2) image,
keeping the source's phase spectrum (which carries structural/content
information, including lesion geometry) unchanged. Borrowed technique,
used honestly as a generalization tool, not claimed as a novelty.

## Priority / Cost / Risk
Low cost. **Leakage-avoidance requirement (important):** split Messidor-2
into a small, fixed "FDA style pool" (used only for sampling amplitude
spectra, no labels ever touched) and the remainder reserved for actual
external-validation reporting — never use the exact same images for both,
even though FDA itself never touches labels. Document this split
explicitly in the paper's methodology section.

---

## Core FDA function

### New file: `src/data/fda_augmentation.py`
```python
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
```

---

## Leak-safe style pool construction

### `src/data/datamodule.py` — new helper, add near the top (after imports):
```python
def build_fda_style_pool(external_csv, external_images, pool_frac=0.2, seed=42, max_images=200):
    """
    Reserves a fixed random subset of Messidor-2 purely for FDA amplitude
    sampling — NEVER used for external-validation metric reporting. Applies
    the same crop/CLAHE/Ben-Graham preprocessing as DRDataset so style
    statistics match what the model actually sees.
    """
    import cv2
    df = pd.read_csv(external_csv)
    if "adjudicated_gradable" in df.columns:
        df = df[df["adjudicated_gradable"] == 1]
    rng = np.random.RandomState(seed)
    pool_df = df.sample(frac=pool_frac, random_state=rng).head(max_images)

    helper = DRDataset(image_dir=external_images, labels_df=pool_df, transform=None)
    pool_images = []
    for idx in range(len(pool_df)):
        row = pool_df.iloc[idx]
        img = helper._load_image(str(row['id_code']) if 'id_code' in pool_df.columns else str(row.iloc[0]))
        if img is None:
            continue
        img = helper._crop_fundus(img)
        img = helper._apply_clahe(img)
        img = helper._apply_ben_graham(img)
        pool_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pool_images, pool_df.index.tolist()
```
(`messidor_data.csv`'s actual ID column name needs verifying against the
real file — adapt the `row['id_code']`/`row.iloc[0]` fallback to whatever
column `DRDataset._load_image` is actually called with elsewhere for the
external loader.)

### `get_dataloaders` (same file) — after the `ext_loader` block (~line 145),
before the `return` (line 151):
```python
    fda_pool_images, fda_pool_indices = (None, [])
    if config.get("use_fda", False):
        fda_pool_images, fda_pool_indices = build_fda_style_pool(
            paths["external_csv"], paths["external_images"],
            pool_frac=config.get("fda_pool_frac", 0.2),
            seed=config.get("fda_pool_seed", 42)
        )
        # Exclude pool images from the external-validation reporting set —
        # rebuild ext_df/ext_dataset/ext_loader excluding fda_pool_indices.
        ext_df_eval = ext_df.drop(index=[i for i in fda_pool_indices if i in ext_df.index])
        ext_dataset = DRDataset(image_dir=paths["external_images"], labels_df=ext_df_eval,
                                 transform=val_test_transform, img_size=img_size)
        ext_loader = DataLoader(ext_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
```
Pass `fda_pool_images` into the **train** dataset constructor (both the
K-fold `CombinedImageDataset` and standard-mode `DRDataset` branches, near
lines 70-85) as a new `fda_pool=None, fda_prob=0.0, fda_beta=0.01` kwarg.

---

## `src/data/dataset.py` — wire FDA into `DRDataset.__getitem__`

**Constructor (line ~22):** add `fda_pool=None, fda_prob: float = 0.0, fda_beta: float = 0.01`.

**`__getitem__`:** insert right after Ben Graham normalization
(`img = self._apply_ben_graham(img)`), before `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`:
```python
        # P4: FDA style augmentation — only applied to training data (fda_pool
        # is None for val/test/external datasets, so this is a no-op there)
        if self.fda_pool and np.random.random() < self.fda_prob:
            import random
            target_img = random.choice(self.fda_pool)
            from .fda_augmentation import fda_source_to_target
            img = fda_source_to_target(img, target_img, beta=self.fda_beta)
```
`CombinedImageDataset` (`datamodule.py`, subclasses `DRDataset`) inherits
this automatically since it reuses `DRDataset.__init__`'s attribute names —
just ensure its own `__init__` (which currently does NOT call
`super().__init__`, see existing code) also stores `self.fda_pool`,
`self.fda_prob`, `self.fda_beta` the same way.

---

## `configs/config.yaml`
```yaml
use_fda: False
fda_pool_frac: 0.2      # fraction of Messidor-2 reserved as style pool only
fda_pool_seed: 42
fda_prob: 0.3           # probability of applying FDA per training sample
fda_beta: 0.03          # low-frequency swap fraction
```

---

## Validation steps
1. Visual sanity check: run `fda_source_to_target` on a few APTOS/Messidor
   pairs, confirm lesion structures are visually unchanged (only
   illumination/color shifts) — save side-by-side comparisons.
2. Confirm `ext_loader`'s reporting set size shrinks by exactly the pool
   fraction, and that no pool image ID appears in both the pool and the
   reporting set.
3. Train one short run with `use_fda: True` vs `False`, compare external
   accuracy on the held-out (non-pool) Messidor-2 subset.

## Expected outcome
Improved external accuracy toward the 90%+ target, with a documented,
leak-safe protocol that can withstand reviewer scrutiny on "did you peek
at the target domain."
