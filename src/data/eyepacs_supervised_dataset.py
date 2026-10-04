import pandas as pd
from .dataset import DRDataset


def build_eyepacs_supervised_dataset(ssl_config, transform, img_size=512):
    df = pd.read_csv(ssl_config['ssl_train_labels_csv'])
    df = df.rename(columns={'image': 'id_code', 'level': 'diagnosis'})
    return DRDataset(
        image_dir=ssl_config['ssl_image_dirs'][0],
        labels_df=df, transform=transform, img_size=img_size
    )
