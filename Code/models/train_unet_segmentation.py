from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from Code.data.severstal_segmentation_dataset import SeverstalSegmentationDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Code" / "outputs" / "results"
MODELS_DIR = PROJECT_ROOT / "Code" / "outputs" / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42
LR = 1e-3


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bridge(p2)

        u2 = self.up2(b)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(c1)


def dice_coefficient(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    dims = (0, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    union = torch.sum(preds, dims) + torch.sum(targets, dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def soft_dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    union = torch.sum(probs, dims) + torch.sum(targets, dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def train_one_epoch(model, loader, optimizer, bce_loss, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = 0.5 * bce_loss(logits, y) + 0.5 * soft_dice_loss(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, bce_loss, device):
    model.eval()
    total_loss = 0.0
    dice_values = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = 0.5 * bce_loss(logits, y) + 0.5 * soft_dice_loss(logits, y)

            total_loss += loss.item() * x.size(0)
            dice_values.append(dice_coefficient(logits, y))

    return total_loss / len(loader.dataset), float(np.mean(dice_values))


def append_results(row):
    results_path = RESULTS_DIR / "unet_segmentation_results.csv"
    file_exists = results_path.exists()

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nSaved results to: {results_path}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device_name}")

    df = pd.read_csv(TRAIN_CSV)
    df = df[df["EncodedPixels"].notna()].copy()

    image_ids = df["ImageId"].drop_duplicates()
    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=0.2,
        random_state=SEED,
    )

    train_df = df[df["ImageId"].isin(train_ids)].copy()
    val_df = df[df["ImageId"].isin(val_ids)].copy()

    train_dataset = SeverstalSegmentationDataset(train_df, img_size=IMG_SIZE)
    val_dataset = SeverstalSegmentationDataset(val_df, img_size=IMG_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = SmallUNet(in_channels=1, out_channels=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce_loss = nn.BCEWithLogitsLoss()

    best_dice = -1.0
    best_path = MODELS_DIR / "unet_segmentation_baseline.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, device)
        val_loss, val_dice = evaluate(model, val_loader, bce_loss, device)

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": "train_unet_segmentation.py",
        "framework": "PyTorch",
        "device": device_name,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "best_val_dice": best_dice,
        "checkpoint_path": str(best_path),
    }

    append_results(row)

    print("\n=== U-Net Segmentation Baseline Results ===")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Saved best checkpoint to: {best_path}")


if __name__ == "__main__":
    main()