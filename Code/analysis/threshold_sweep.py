from pathlib import Path
import numpy as np
import pandas as pd
import torch

from Code.models.train_unet_segmentation import SmallUNet
from Code.data.severstal_segmentation_dataset import SeverstalSegmentationDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "Code" / "outputs" / "models" / "unet_segmentation_baseline.pt"

IMG_SIZE = 128


def dice_score(pred, target, eps=1e-7):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2.0 * intersection + eps) / (union + eps)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATA_DIR / "train.csv")
    df = df[df["EncodedPixels"].notna()]

    dataset = SeverstalSegmentationDataset(df, img_size=IMG_SIZE)

    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    thresholds = np.linspace(0.1, 0.9, 9)

    for t in thresholds:
        dice_vals = []

        for i in range(50):  # small sample for speed
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = torch.sigmoid(model(x)).cpu().numpy()[0]

            pred_bin = (pred >= t).astype(np.float32)
            gt = y.numpy()

            dice_vals.append(dice_score(pred_bin, gt))

        print(f"Threshold {t:.2f} → Dice {np.mean(dice_vals):.4f}")


if __name__ == "__main__":
    main()