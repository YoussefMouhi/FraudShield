from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models/isolation_forest.joblib")


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Reproduce our existing train/validation/test split.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42,
    )

    # Fit only on legitimate training transactions.
    X_train_normal = X_train[y_train == 0]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=200,
            max_samples=256,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training Isolation Forest...")
    model.fit(X_train_normal)

    # Higher score means more anomalous.
    anomaly_scores = -model.decision_function(X_val)

    precision, recall, thresholds = precision_recall_curve(
        y_val, anomaly_scores
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    threshold = float(thresholds[best_index])

    predictions = (anomaly_scores >= threshold).astype(int)

    print("\nISOLATION FOREST VALIDATION RESULTS")
    print("-----------------------------------")
    print(f"PR-AUC: {average_precision_score(y_val, anomaly_scores):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, anomaly_scores):.4f}")
    print(f"Precision: {precision_score(y_val, predictions):.4f}")
    print(f"Recall: {recall_score(y_val, predictions):.4f}")
    print(f"F1-score: {f1_score(y_val, predictions):.4f}")
    print(f"Threshold: {threshold:.6f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "features": X.columns.tolist(),
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Reserved test set remains unused.")


if __name__ == "__main__":
    main()
