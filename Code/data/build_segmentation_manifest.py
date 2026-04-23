from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "Code" / "outputs" / "manifests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")

    # Only rows with actual masks
    df_defects = df[df["EncodedPixels"].notna()].copy()

    # Aggregate to one row per image
    grouped = df_defects.groupby("ImageId")

    rows = []

    for image_id, group in grouped:
        class_ids = sorted(group["ClassId"].astype(int).unique())

        row = {
            "ImageId": image_id,
            "num_classes": len(class_ids),
            "has_class_1": int(1 in class_ids),
            "has_class_2": int(2 in class_ids),
            "has_class_3": int(3 in class_ids),
            "has_class_4": int(4 in class_ids),
        }

        rows.append(row)

    manifest = pd.DataFrame(rows)

    out_path = OUTPUT_DIR / "segmentation_manifest.csv"
    manifest.to_csv(out_path, index=False)

    print("\n=== Segmentation Manifest Summary ===")
    print(f"Total defect images: {len(manifest)}")

    print("\nImages by number of defect classes:")
    print(manifest["num_classes"].value_counts().sort_index())

    print("\nClass presence counts:")
    for c in [1, 2, 3, 4]:
        print(f"class_{c}: {manifest[f'has_class_{c}'].sum()}")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()