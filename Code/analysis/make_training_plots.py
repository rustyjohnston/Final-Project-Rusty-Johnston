import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "Code/outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# === MULTICLASS ===
multi = pd.read_csv(ROOT / "Code/outputs/results/unet_training_history.csv")

fig, ax1 = plt.subplots()
ax1.plot(multi["epoch"], multi["train_loss"], label="train_loss")
ax1.plot(multi["epoch"], multi["val_loss"], label="val_loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

ax2 = ax1.twinx()
ax2.plot(multi["epoch"], multi["val_dice"], label="val_dice", color="green")
ax2.set_ylabel("Dice")

ax1.set_title("Multiclass U-Net Training Curve")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.savefig(FIG_DIR / "multiclass_training_curve.png")
plt.close()

# === BINARY ===
binary = pd.read_csv(ROOT / "Code/outputs/results/binary_unet_training_history.csv")

plt.figure()
plt.plot(binary["epoch"], binary["val_dice"], label="val_dice")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Binary U-Net Training Curve")
plt.legend()

plt.savefig(FIG_DIR / "binary_training_curve.png")
plt.close()

print("Saved training plots")