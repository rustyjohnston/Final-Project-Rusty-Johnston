from pathlib import Path
import torch
import cv2
import numpy as np
import pandas as pd

from Code.models.train_unet_segmentation import SmallUNet
from Code.data.severstal_segmentation_dataset import SeverstalSegmentationDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "Code" / "outputs" / "figures"
MODEL_PATH = PROJECT_ROOT / "Code" / "outputs" / "models" / "unet_segmentation_baseline.pt"

FIG_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128


def overlay_mask(image, mask, color):
    overlay = image.copy()
    overlay[mask > 0.5] = color
    return overlay


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATA_DIR / "train.csv")
    df = df[df["EncodedPixels"].notna()]

    dataset = SeverstalSegmentationDataset(df, img_size=IMG_SIZE)

    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    for i in range(5):
        x, y = dataset[i]

        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(x)).cpu().numpy()[0]

        image = x.cpu().numpy()[0][0]
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gt_mask = y.numpy()
        pred_mask = pred

        gt_overlay = image.copy()
        pred_overlay = image.copy()

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ]

        for c in range(4):
            gt_overlay = overlay_mask(gt_overlay, gt_mask[c], colors[c])
            pred_overlay = overlay_mask(pred_overlay, pred_mask[c], colors[c])

        combined = np.hstack([image, gt_overlay, pred_overlay])

        out_path = FIG_DIR / f"example_{i}.png"
        cv2.imwrite(str(out_path), combined)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()