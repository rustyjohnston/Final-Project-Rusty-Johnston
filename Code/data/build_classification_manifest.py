from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMG_DIR = DATA_DIR / "train_images"
OUTPUT_DIR = PROJECT_ROOT / "Code" / "outputs" / "manifests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")

    # All images on disk
    all_images = sorted([p.name for p in IMG_DIR.glob("*.jpg")])

    # Initialize manifest
    manifest = pd.DataFrame({"ImageId": all_images})

    # Initialize columns
    for c in [1, 2, 3, 4]:
        manifest[f"class_{c}"] = 0

    # Fill defect labels
    for _, row in df.iterrows():
        if pd.notna(row["EncodedPixels"]):
            img = row["ImageId"]
            cls = int(row["ClassId"])
            manifest.loc[manifest["ImageId"] == img, f"class_{cls}"] = 1

    # Binary label
    class_cols = [f"class_{c}" for c in [1, 2, 3, 4]]
    manifest["has_defect"] = manifest[class_cols].max(axis=1)

    out_path = OUTPUT_DIR / "classification_manifest.csv"
    manifest.to_csv(out_path, index=False)

    print("\n=== Classification Manifest Summary ===")
    print(f"Total images: {len(manifest)}")
    print(f"Defect images: {manifest['has_defect'].sum()}")
    print(f"No-defect images: {(manifest['has_defect'] == 0).sum()}")

    print("\nClass distribution:")
    for c in class_cols:
        print(f"{c}: {manifest[c].sum()}")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()