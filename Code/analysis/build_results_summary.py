from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Code" / "outputs" / "results"
SUMMARY_PATH = RESULTS_DIR / "final_results_summary.csv"


def latest_row(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df.iloc[-1]


def main():
    rows = []

    binary_path = RESULTS_DIR / "binary_cnn_results.csv"
    if binary_path.exists():
        r = latest_row(binary_path)
        rows.append({
            "task": "Binary defect classification",
            "model": "Simple CNN",
            "framework": r.get("framework", "PyTorch"),
            "device": r.get("device", ""),
            "primary_metric": "F1",
            "primary_value": r.get("f1", ""),
            "accuracy": r.get("accuracy", ""),
            "precision": r.get("precision", ""),
            "recall": r.get("recall", ""),
            "notes": "Defect vs no-defect classification baseline",
        })

    unet_path = RESULTS_DIR / "unet_segmentation_results.csv"
    if unet_path.exists():
        r = latest_row(unet_path)
        rows.append({
            "task": "Defect localization / segmentation",
            "model": "Small U-Net",
            "framework": r.get("framework", "PyTorch"),
            "device": r.get("device", ""),
            "primary_metric": "Dice coefficient",
            "primary_value": r.get("best_val_dice", ""),
            "accuracy": "",
            "precision": "",
            "recall": "",
            "notes": "Four-channel segmentation baseline using encoded defect masks",
        })

    summary = pd.DataFrame(rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()