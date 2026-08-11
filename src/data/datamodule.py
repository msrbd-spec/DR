import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from .dataset import DRDataset
from .transforms import get_train_transforms, get_val_test_transforms

def get_dataloaders(config: dict):
    """
    Reads pre-split CSVs directly and builds dataloaders for all 4 splits.
    Computes class weights from train split ONLY and returns as torch.FloatTensor.
    """
    img_size = config.get("img_size", 384)
    batch_size = config.get("batch_size", 32)
    paths = config.get("dataset_paths", {})
    
    # Transforms
    train_transform = get_train_transforms(img_size)
    val_test_transform = get_val_test_transforms(img_size)
    
    # 1. Training Set
    train_df = pd.read_csv(paths["train_csv"])
    train_dataset = DRDataset(
        image_dir=paths["train_images"],
        labels_df=train_df,
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Validation Set
    val_df = pd.read_csv(paths["val_csv"])
    val_dataset = DRDataset(
        image_dir=paths["val_images"],
        labels_df=val_df,
        transform=val_test_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Internal Test Set
    test_df = pd.read_csv(paths["test_csv"])
    test_dataset = DRDataset(
        image_dir=paths["test_images"],
        labels_df=test_df,
        transform=val_test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. External Validation Set (Messidor-2)
    ext_df = pd.read_csv(paths["external_csv"])
    # Filter to adjudicated_gradable == 1
    ext_df = ext_df[ext_df["adjudicated_gradable"] == 1]
    ext_dataset = DRDataset(
        image_dir=paths["external_images"],
        labels_df=ext_df,
        transform=val_test_transform
    )
    ext_loader = DataLoader(ext_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Compute Class Weights from train_df ONLY
    y_train = train_df["diagnosis"].values
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    # Mandatory conversion to torch.FloatTensor here so it's ready for device placement later
    class_weights = torch.FloatTensor(weights)
    
    return train_loader, val_loader, test_loader, ext_loader, class_weights
