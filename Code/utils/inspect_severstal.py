import numpy as np
import pandas as pd
import cv2
import os

DATA_DIR = "/home/ubuntu/Final-Project-Rusty-Johnston/data/raw"
IMG_DIR = os.path.join(DATA_DIR, "train_images")


def rle_decode(mask_rle, shape=(256, 1600)):
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]

    starts = np.array(starts) - 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for start, end in zip(starts, ends):
        img[start:end] = 1

    return img.reshape(shape, order='F')


def overlay_mask(image, mask):
    overlay = image.copy()
    overlay[mask == 1] = [255, 0, 0]  # red mask
    return overlay


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    # pick a row with a defect
    row = df[df["EncodedPixels"].notna()].iloc[0]

    image_id = row["ImageId"]
    class_id = row["ClassId"]
    rle = row["EncodedPixels"]

    print(f"Using image: {image_id}, class: {class_id}")

    image_path = os.path.join(IMG_DIR, image_id)
    image = cv2.imread(image_path)

    mask = rle_decode(rle)

    # convert grayscale to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overlay = overlay_mask(image, mask)

    out_path = "overlay_example.png"
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()