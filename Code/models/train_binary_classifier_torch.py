from pathlib import Path
from datetime import datetime
import csv

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMG_DIR = DATA_DIR / "train_images"
MANIFEST_PATH = PROJECT_ROOT / "Code" / "outputs" / "manifests" / "classification_manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "Code" / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 5
SEED = 42


class SeverstalBinaryDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = IMG_DIR / row["ImageId"]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        label = np.float32(row["has_defect"])
        return torch.tensor(img), torch.tensor(label)


class SimpleBinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).cpu().numpy().astype(int)

            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().astype(int).tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def append_results(metrics, device_name):
    results_path = RESULTS_DIR / "binary_cnn_results.csv"
    file_exists = results_path.exists()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": "train_binary_classifier_torch.py",
        "framework": "PyTorch",
        "device": device_name,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        **metrics,
    }

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nSaved tracked results to: {results_path}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device_name}")

    df = pd.read_csv(MANIFEST_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["has_defect"],
    )

    train_loader = DataLoader(
        SeverstalBinaryDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    test_loader = DataLoader(
        SeverstalBinaryDataset(test_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    model = SimpleBinaryCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch}/{EPOCHS} - loss: {loss:.4f}")

    metrics = evaluate(model, test_loader, device)

    print("\n=== PyTorch Binary Classification Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    append_results(metrics, device_name)


if __name__ == "__main__":
    main()