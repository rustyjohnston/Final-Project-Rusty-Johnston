import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# === TRAINING CURVE ===
hist = pd.read_csv(ROOT / "Code/outputs/results/unet_training_history.csv")

plt.figure()
plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
plt.plot(hist["epoch"], hist["val_dice"], label="val_dice")
plt.legend()
plt.xlabel("Epoch")
plt.title("U-Net Training Curve")
plt.savefig(ROOT / "Code/outputs/figures/training_curve.png")

# === RESULTS BAR CHART ===
labels = ["Classification F1", "Multiclass Dice", "Binary Dice"]
values = [0.818, 0.325, 0.564]

plt.figure()
plt.bar(labels, values)
plt.title("Model Performance Comparison")
plt.savefig(ROOT / "Code/outputs/figures/results_comparison.png")

print("Saved plots")