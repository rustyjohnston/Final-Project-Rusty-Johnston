from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from Code.data.severstal_binary_segmentation_dataset import SeverstalBinarySegmentationDataset
from Code.models.train_unet_segmentation import SmallUNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Code" / "outputs" / "results"
MODELS_DIR = PROJECT_ROOT / "Code" / "outputs" / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
SEED = 42


def dice_score(pred, target, eps=1e-7):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + eps) / (union + eps)


def train():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(TRAIN_CSV)
    df = df[df["EncodedPixels"].notna()]

    image_ids = df["ImageId"].drop_duplicates()
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=SEED)

    train_df = df[df["ImageId"].isin(train_ids)]
    val_df = df[df["ImageId"].isin(val_ids)]

    train_loader = DataLoader(
        SeverstalBinarySegmentationDataset(train_df, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        SeverstalBinarySegmentationDataset(val_df, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    model = SmallUNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    best_dice = 0.0
    best_path = MODELS_DIR / "binary_unet_segmentation.pt"
    history = []

    for epoch in range(EPOCHS):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        dices = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = torch.sigmoid(model(x))
                pred = (pred > 0.4).float()

                dices.append(dice_score(pred, y).item())

        val_dice = np.mean(dices)
        print(f"Epoch {epoch+1} Dice: {val_dice:.4f}")
        history.append({
            "epoch": epoch + 1,
            "val_dice": val_dice,
        })
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)

    print("\nBest Binary Dice:", best_dice)
    print(f"Saved best model to: {best_path}")

    history_path = RESULTS_DIR / "binary_unet_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved binary training history to: {history_path}")

    results_path = RESULTS_DIR / "binary_unet_segmentation_results.csv"
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": "train_unet_binary_segmentation.py",
        "framework": "PyTorch",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "best_binary_dice": best_dice,
    }

    file_exists = results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    train()