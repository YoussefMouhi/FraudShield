from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models/logistic_regression.joblib")


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Reserve 20% for final testing.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # Split development data into training and validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42,
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Reserved test samples:", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_index])

    predictions = (probabilities >= best_threshold).astype(int)

    print("\nVALIDATION RESULTS")
    print("------------------")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Precision: {precision_score(y_val, predictions):.4f}")
    print(f"Recall: {recall_score(y_val, predictions):.4f}")
    print(f"F1-score: {f1_score(y_val, predictions):.4f}")
    print(
        "PR-AUC:",
        round(average_precision_score(y_val, probabilities), 4)
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "features": X.columns.tolist(),
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Final test set remains unused.")


if __name__ == "__main__":
    main()
