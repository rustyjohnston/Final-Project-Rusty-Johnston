import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMG_DIR = DATA_DIR / "train_images"
OUTPUT_DIR = PROJECT_ROOT / "Code" / "outputs" / "inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_COLORS = {
    1: np.array([255, 0, 0], dtype=np.uint8),      # red
    2: np.array([0, 255, 0], dtype=np.uint8),      # green
    3: np.array([0, 0, 255], dtype=np.uint8),      # blue
    4: np.array([255, 255, 0], dtype=np.uint8),    # yellow
}


def rle_decode(mask_rle: str | float, shape: tuple[int, int] = (256, 1600)) -> np.ndarray:
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


def blend_mask(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    output = image.copy()
    mask_bool = mask.astype(bool)

    for c in range(3):
        output[..., c][mask_bool] = (
            (1.0 - alpha) * output[..., c][mask_bool] + alpha * color[c]
        ).astype(np.uint8)

    return output


def build_multiclass_overlay(image: np.ndarray, rows: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    overlay = image.copy()
    pixel_counts: dict[int, int] = {}

    for _, row in rows.iterrows():
        class_id = int(row["ClassId"])
        encoded_pixels = row["EncodedPixels"]

        if pd.isna(encoded_pixels):
            continue

        mask = rle_decode(encoded_pixels)
        pixel_counts[class_id] = int(mask.sum())

        overlay = blend_mask(
            image=overlay,
            mask=mask,
            color=CLASS_COLORS[class_id],
            alpha=0.35,
        )

    return overlay, pixel_counts


def choose_interesting_image(df: pd.DataFrame) -> str:
    defect_rows = df[df["EncodedPixels"].notna()].copy()
    counts = defect_rows.groupby("ImageId")["ClassId"].nunique().sort_values(ascending=False)
    return counts.index[0]


def main() -> None:
    df = pd.read_csv(DATA_DIR / "train.csv")

    print("\nBasic dataset stats:")
    print("Total rows:", len(df))
    print("Unique images:", df["ImageId"].nunique())

    print("\nDefects per class:")
    print(df[df["EncodedPixels"].notna()]["ClassId"].value_counts().sort_index())

    print("\nImages with at least one defect:")
    images_with_defects = df.groupby("ImageId")["EncodedPixels"].apply(lambda x: x.notna().any()).sum()
    print(images_with_defects)

    image_id = choose_interesting_image(df)
    rows = df[df["ImageId"] == image_id].sort_values("ClassId")

    print(f"\nUsing image: {image_id}")
    print("Classes present in selected image:", rows[rows["EncodedPixels"].notna()]["ClassId"].tolist())

    image_path = IMG_DIR / image_id
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    overlay_rgb, pixel_counts = build_multiclass_overlay(image_rgb, rows)

    original_out = OUTPUT_DIR / "original_example.png"
    overlay_out = OUTPUT_DIR / "overlay_multiclass_example.png"

    cv2.imwrite(str(original_out), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_out), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    print("\nPixel counts by class in selected image:")
    print(pixel_counts)
    print(f"Saved original image to: {original_out}")
    print(f"Saved multiclass overlay to: {overlay_out}")


if __name__ == "__main__":
    main()