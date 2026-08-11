import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DRDataset(Dataset):
    def __init__(self, image_dir: str, labels_df: pd.DataFrame, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            labels_df (pd.DataFrame): DataFrame containing 'id_code' and 'diagnosis' columns.
            transform (albumentations.Compose, optional): Transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # 1. Image path resolution
        row = self.labels_df.iloc[idx]
        img_name = str(row['id_code'])
        # Append .png if not present
        if not (img_name.endswith('.png') or img_name.endswith('.jpg') or img_name.endswith('.jpeg')):
            img_name += '.png'
            
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read image with cv2
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        # Automated OpenCV Preprocessing Logic
        
        # 1. Read image with cv2 and convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 2. Apply cv2.threshold to isolate the fundus contour
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # 3. Use cv2.boundingRect to get (x, y, w, h) and crop
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = img[y:y+h, x:x+w]
            
        # 4. Resize to 384x384
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_CUBIC)
        
        # 5. Apply Ben Graham's enhancement
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
        
        # 6. Convert back to RGB, apply transform, return tensor and label
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        label = int(row['diagnosis'])
        return img, label
