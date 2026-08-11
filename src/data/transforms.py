import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size: int = 384):
    """
    Albumentations pipeline for the training set.
    Includes resizing, heavy augmentation, normalization, and tensor conversion.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_test_transforms(img_size: int = 384):
    """
    Albumentations pipeline for validation, test, and external sets.
    Includes only resizing, normalization, and tensor conversion. No augmentation.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
