from __future__ import annotations

import os

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

NUM_CLASSES = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 1600


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------

def decode_rle(rle: str, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH) -> np.ndarray:
    """RLE → binary mask (H, W), column-major (Fortran order)."""
    if not isinstance(rle, str) or not rle.strip():
        return np.zeros(height * width, dtype=np.uint8).reshape(height, width)
    nums = list(map(int, rle.split()))
    starts, lengths = nums[::2], nums[1::2]
    pixels = np.zeros(height * width, dtype=np.uint8)
    for s, l in zip(starts, lengths):
        pixels[s - 1: s - 1 + l] = 1
    return pixels.reshape(height, width, order="F")


def encode_rle(mask: np.ndarray) -> str:
    """Binary mask (H, W) → RLE string, column-major (Fortran order)."""
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs) if runs.size > 0 else ""


# ---------------------------------------------------------------------------
# Dataframe loading
# ---------------------------------------------------------------------------

def load_train_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # CSV has pre-split ImageId and ClassId columns
    df["ClassId"] = df["ClassId"].astype(int)
    return df


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """One row per image with columns class_1..class_4 holding RLE strings."""
    pivot = df.pivot_table(
        index="ImageId", columns="ClassId", values="EncodedPixels", aggfunc="first"
    ).reindex(columns=[1, 2, 3, 4])
    pivot.columns = [f"class_{c}" for c in [1, 2, 3, 4]]
    return pivot


# ---------------------------------------------------------------------------
# Albumentations transform factories
# ---------------------------------------------------------------------------

def get_transforms(height: int, width: int, phase: str) -> A.Compose:
    if phase == "train":
        return A.Compose([
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-5, 5), p=0.3),
            # GridDistortion helps with surface-texture defects
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            # Dropout wide horizontal patches to prevent relying on local texture cues
            A.CoarseDropout(
                num_holes_range=(4, 8),
                hole_height_range=(8, 16),
                hole_width_range=(32, 96),
                p=0.3,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ])
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=True),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SteelDataset(Dataset):
    """Training / validation dataset with RLE-decoded masks."""

    def __init__(
        self,
        pivot: pd.DataFrame,
        image_dir: str,
        transform: A.Compose,
    ) -> None:
        self.pivot = pivot
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids: list[str] = pivot.index.tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = np.array(
            Image.open(os.path.join(self.image_dir, image_id)).convert("RGB")
        )
        h, w = image.shape[:2]

        mask = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
        row = self.pivot.loc[image_id]
        for i, col in enumerate([f"class_{c}" for c in range(1, 5)]):
            rle = row[col]
            if isinstance(rle, str):
                mask[:, :, i] = decode_rle(rle, h, w).astype(np.float32)

        aug = self.transform(image=image, mask=mask)
        return aug["image"].float(), aug["mask"].float()


class SteelTestDataset(Dataset):
    """Test-time dataset — returns (image_id, image_tensor)."""

    def __init__(self, image_ids: list[str], image_dir: str, transform: A.Compose) -> None:
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = np.array(
            Image.open(os.path.join(self.image_dir, image_id)).convert("RGB")
        )
        aug = self.transform(image=image)
        return image_id, aug["image"].float()
