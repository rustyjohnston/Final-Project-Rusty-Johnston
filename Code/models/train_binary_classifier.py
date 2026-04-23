import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMG_DIR = DATA_DIR / "train_images"
MANIFEST_PATH = PROJECT_ROOT / "Code" / "outputs" / "manifests" / "classification_manifest.csv"


IMG_SIZE = 128


def load_data():
    df = pd.read_csv(MANIFEST_PATH)

    X = []
    y = []

    for _, row in df.iterrows():
        img_path = IMG_DIR / row["ImageId"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(row["has_defect"])

    X = np.array(X)
    y = np.array(y)

    X = X[..., np.newaxis]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model():
    model = models.Sequential([
        layers.Conv2D(16, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = build_model()

    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )

    preds = model.predict(X_test)
    preds = (preds > 0.5).astype(int)

    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)

    print("\n=== Binary Classification Results ===")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()