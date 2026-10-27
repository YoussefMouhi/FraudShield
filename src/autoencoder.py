from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
tf.keras.utils.set_random_seed(SEED)

DATA_PATH = Path("data/creditcard.csv")
MODEL_DIR = Path("models")


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Reproduce the same splits as the other models.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42,
    )

    # Train only on legitimate transactions.
    X_train_normal = X_train[y_train == 0]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(
        X_train_normal
    ).astype("float32")

    val_scaled = scaler.transform(
        X_val
    ).astype("float32")

    input_dim = train_scaled.shape[1]

    # Autoencoder: compress and reconstruct transactions.
    autoencoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(6, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(input_dim, activation="linear"),
    ])

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
    )

    print("Training Autoencoder...")

    autoencoder.fit(
        train_scaled,
        train_scaled,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    # Reconstruction error becomes the anomaly score.
    reconstructed = autoencoder.predict(
        val_scaled,
        verbose=0,
    )

    errors = np.mean(
        np.square(val_scaled - reconstructed),
        axis=1,
    )

    precision, recall, thresholds = precision_recall_curve(
        y_val,
        errors,
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    threshold = float(thresholds[best_index])

    predictions = (errors >= threshold).astype(int)

    print("\nAUTOENCODER VALIDATION RESULTS")
    print("--------------------------------")
    print(f"PR-AUC: {average_precision_score(y_val, errors):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, errors):.4f}")
    print(f"Precision: {precision_score(y_val, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, predictions, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_val, predictions, zero_division=0):.4f}")
    print(f"Threshold: {threshold:.6f}")

    MODEL_DIR.mkdir(exist_ok=True)

    autoencoder.save(
        MODEL_DIR / "autoencoder.keras"
    )

    joblib.dump(
        {
            "scaler": scaler,
            "threshold": threshold,
            "features": X.columns.tolist(),
        },
        MODEL_DIR / "autoencoder_preprocessing.joblib",
    )

    print("\nAutoencoder saved successfully.")
    print("Reserved test set remains unused.")


if __name__ == "__main__":
    main()
