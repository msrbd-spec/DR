import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

from .dataset import DRDataset
from .transforms import get_train_transforms, get_val_test_transforms


def compute_cb_weights(y_train, beta=0.9999):
    classes = np.unique(y_train)
    counts = np.bincount(y_train, minlength=len(classes))
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.sum() * len(classes)
    return weights


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


def get_dataloaders(config: dict, fold_idx: int = None):
    """
    Reads pre-split CSVs and builds dataloaders for all 4 splits.
    Computes class weights from train split ONLY and returns as torch.FloatTensor.

    If fold_idx is provided (K-fold mode), the train and val sets are merged
    and re-split using StratifiedKFold. The test and external sets remain fixed.

    Args:
        config: Configuration dict.
        fold_idx: If not None, uses K-fold cross-validation. fold_idx in [0, n_folds).
                  When None, uses the pre-split train/val CSVs directly.
    """
    img_size = config.get("img_size", 384)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)
    paths = config.get("dataset_paths", {})

    # Transforms
    train_transform = get_train_transforms(img_size)
    val_test_transform = get_val_test_transforms(img_size)

    fda_pool_images, fda_pool_indices = (None, [])
    if config.get("use_fda", False) and "external_csv" in paths and "external_images" in paths:
        fda_pool_images, fda_pool_indices = build_fda_style_pool(
            paths["external_csv"], paths["external_images"],
            pool_frac=config.get("fda_pool_frac", 0.2),
            seed=config.get("fda_pool_seed", 42)
        )
    fda_prob = config.get("fda_prob", 0.3)
    fda_beta = config.get("fda_beta", 0.03)

    if fold_idx is not None:
        # K-Fold mode: merge train + val, re-split with StratifiedKFold
        n_folds = config.get("n_folds", 5)
        kfold_seed = config.get("kfold_seed", 42)

        train_df = pd.read_csv(paths["train_csv"])
        val_df = pd.read_csv(paths["val_csv"])
        merged_df = pd.concat([train_df, val_df], ignore_index=True)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=kfold_seed)
        all_splits = list(skf.split(merged_df, merged_df["diagnosis"]))

        train_indices, val_indices = all_splits[fold_idx]
        fold_train_df = merged_df.iloc[train_indices].reset_index(drop=True)
        fold_val_df = merged_df.iloc[val_indices].reset_index(drop=True)

        # Use merged image directories — need to handle both train_images and val_images
        # Create a combined image directory reference by using a custom dataset wrapper
        train_dataset = CombinedImageDataset(
            image_dirs=[paths["train_images"], paths["val_images"]],
            labels_df=fold_train_df,
            transform=train_transform,
            img_size=img_size,
            fda_pool=fda_pool_images, fda_prob=fda_prob, fda_beta=fda_beta
        )
        val_dataset = CombinedImageDataset(
            image_dirs=[paths["train_images"], paths["val_images"]],
            labels_df=fold_val_df,
            transform=val_test_transform,
            img_size=img_size
        )

        y_train = fold_train_df["diagnosis"].values
    else:
        # Standard mode: use pre-split CSVs directly
        train_df = pd.read_csv(paths["train_csv"])
        train_dataset = DRDataset(
            image_dir=paths["train_images"],
            labels_df=train_df,
            transform=train_transform,
            img_size=img_size,
            fda_pool=fda_pool_images, fda_prob=fda_prob, fda_beta=fda_beta
        )
        val_df = pd.read_csv(paths["val_csv"])
        val_dataset = DRDataset(
            image_dir=paths["val_images"],
            labels_df=val_df,
            transform=val_test_transform,
            img_size=img_size
        )
        y_train = train_df["diagnosis"].values

    # Class-balanced sampling: oversample minority classes (APTOS is ~49% class 0).
    # Controlled by config flag `use_weighted_sampling` (default True).
    # Note: when using a sampler, shuffle must be False (sampler replaces it).
    classes = np.unique(y_train)
    use_weighted_sampling = config.get("use_weighted_sampling", True)

    if use_weighted_sampling:
        class_counts = np.bincount(y_train, minlength=len(classes))
        class_weights = 1.0 / np.maximum(class_counts, 1)  # avoid div-by-zero
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    # 3. Internal Test Set (always fixed)
    test_df = pd.read_csv(paths["test_csv"])
    test_dataset = DRDataset(
        image_dir=paths["test_images"],
        labels_df=test_df,
        transform=val_test_transform,
        img_size=img_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    # 4. External Validation Set (Messidor-2)
    ext_loader = None
    if "external_csv" in paths and "external_images" in paths:
        ext_df = pd.read_csv(paths["external_csv"])
        if "adjudicated_gradable" in ext_df.columns:
            ext_df = ext_df[ext_df["adjudicated_gradable"] == 1]
            
        ext_df_eval = ext_df.drop(index=[i for i in fda_pool_indices if i in ext_df.index])
        
        ext_dataset = DRDataset(
            image_dir=paths["external_images"],
            labels_df=ext_df_eval,
            transform=val_test_transform,
            img_size=img_size
        )
        ext_loader = DataLoader(
            ext_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )

    # Compute class weights for loss function
    class_weight_strategy = config.get("class_weight_strategy", "inverse")
    if class_weight_strategy == "effective_num":
        cb_beta = config.get("cb_beta", 0.9999)
        weights = compute_cb_weights(y_train, beta=cb_beta)
    else:
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.FloatTensor(weights)

    class_counts_for_prior = np.bincount(y_train, minlength=len(classes))
    class_priors = class_counts_for_prior / class_counts_for_prior.sum()
    class_priors_tensor = torch.FloatTensor(class_priors)

    return train_loader, val_loader, test_loader, ext_loader, class_weights_tensor, class_priors_tensor, class_counts_for_prior.tolist()


class CombinedImageDataset(DRDataset):
    """
    Dataset that searches for images across multiple directories.
    Used in K-fold mode where train and val images come from both
    train_images/ and val_images/ directories.
    """

    def __init__(self, image_dirs, labels_df, transform=None, img_size=384, fda_pool=None, fda_prob=0.0, fda_beta=0.01):
        # Don't call super().__init__ with a single dir — we handle multiple dirs
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.labels_df = labels_df
        self.transform = transform
        self.img_size = img_size
        self.fda_pool = fda_pool
        self.fda_prob = fda_prob
        self.fda_beta = fda_beta

    def _load_image_multi_dir(self, img_name):
        """
        Load image trying multiple directories and multiple extensions.
        Returns None if image cannot be found.
        """
        for img_dir in self.image_dirs:
            # If extension already provided, try as-is first
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(img_dir, img_name)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        return img

            # Try each extension
            for ext in ['.png', '.jpg', '.jpeg']:
                img_path = os.path.join(img_dir, img_name + ext)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        return img

        return None

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_name = str(row['id_code'])

        # Load image from multiple directories with multiple extensions
        img = self._load_image_multi_dir(img_name)
        if img is None:
            # Graceful fallback: return a black placeholder image so training doesn't crash
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            # Apply same preprocessing as DRDataset
            img = self._crop_fundus(img)
            img = self._apply_clahe(img)
            img = self._apply_ben_graham(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        label = int(row['diagnosis'])
        return img, label

