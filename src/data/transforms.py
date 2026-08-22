import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 384):
    """
    Strong augmentation pipeline for the training set.

    Key additions over the original pipeline:
      - RandomResizedCrop: scale jittering forces scale-invariant features
      - CLAHE: additional contrast enhancement as augmentation
      - CoarseDropout: forces model to not rely on single image regions
      - OpticalDistortion + GlassBlur: simulates camera lens variation
      - Sharpen: enhances lesion edge visibility
      - RandomBrightnessContrast: targeted for fundus illumination variation

    NOTE: Uses `size=(h, w)` API for newer albumentations versions (>=1.4).
    """
    return A.Compose([
        # RandomResizedCrop replaces fixed Resize — introduces scale invariance
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),        # crop between 80%-100% of image
            ratio=(0.9, 1.1),         # mild aspect ratio variation
            interpolation=cv2.INTER_CUBIC,
            p=1.0
        ),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        # Reduced destructive augmentations — were destroying fine lesion details
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.OpticalDistortion(distort_limit=0.05, p=0.1),
        # Removed GlassBlur — it destroys microaneurysm/hemorrhage details
        A.Affine(translate_percent=0.1, scale=0.9, rotate=45, p=0.3),
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0, p=0.3
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_test_transforms(img_size: int = 384):
    """
    Validation / Test / External transform pipeline.
    No augmentation — only resize, normalisation, and tensor conversion.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_tta_transforms(img_size: int = 384, scale: float = 1.0):
    """
    Transform pipeline for a single TTA view at a given scale.
    The scale parameter controls the multi-scale crop size before final resize.
    """
    crop_size = int(img_size * scale)
    return A.Compose([
        A.Resize(height=crop_size, width=crop_size),
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
