from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMG_DIR = DATA_DIR / "train_images"


def rle_decode(mask_rle, shape=(256, 1600)):
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, str(mask_rle).split()))
    starts, lengths = s[0::2], s[1::2]

    starts = np.array(starts) - 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for start, end in zip(starts, ends):
        img[start:end] = 1

    return img.reshape(shape, order="F")


class SeverstalBinarySegmentationDataset(Dataset):
    def __init__(self, df, img_size=128):
        self.df = df
        self.img_size = img_size
        self.image_ids = df["ImageId"].unique()
        self.grouped = df.groupby("ImageId")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        rows = self.grouped.get_group(image_id)

        img_path = IMG_DIR / image_id
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(img_path)

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0

        # Binary mask (combine all classes)
        mask = np.zeros((256, 1600), dtype=np.uint8)

        for _, row in rows.iterrows():
            mask = np.maximum(mask, rle_decode(row["EncodedPixels"]))

        mask = cv2.resize(mask, (self.img_size, self.img_size))

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )