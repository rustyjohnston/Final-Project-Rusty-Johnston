from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMG_DIR = DATA_DIR / "train_images"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "train.csv")

    csv_images = set(df["ImageId"].unique())
    disk_images = {p.name for p in IMG_DIR.glob("*.jpg")}

    images_only_on_disk = sorted(disk_images - csv_images)
    images_only_in_csv = sorted(csv_images - disk_images)

    print("=== Severstal Dataset Inventory ===")
    print(f"Rows in CSV: {len(df)}")
    print(f"Unique images in CSV: {len(csv_images)}")
    print(f"Image files on disk: {len(disk_images)}")
    print(f"Images on disk but not in CSV: {len(images_only_on_disk)}")
    print(f"Images in CSV but not on disk: {len(images_only_in_csv)}")

    if images_only_on_disk:
        print("\nSample images on disk but not in CSV:")
        for name in images_only_on_disk[:10]:
            print(name)

    if images_only_in_csv:
        print("\nSample images in CSV but not on disk:")
        for name in images_only_in_csv[:10]:
            print(name)

    defect_counts = (
        df[df["EncodedPixels"].notna()]
        .groupby("ImageId")["ClassId"]
        .nunique()
        .value_counts()
        .sort_index()
    )

    print("\nNumber of classes present per labeled image:")
    print(defect_counts.to_string())

    class_pixel_totals = {}
    for class_id, group in df[df["EncodedPixels"].notna()].groupby("ClassId"):
        class_pixel_totals[int(class_id)] = len(group)

    print("\nRows per defect class:")
    for class_id, count in sorted(class_pixel_totals.items()):
        print(f"Class {class_id}: {count}")


if __name__ == "__main__":
    main()