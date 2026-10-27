from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models/xgboost_fraud_detector.joblib")
OUTPUT_DIR = Path("assets")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load the selected model and validation-selected threshold.
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    threshold = bundle["threshold"]
    features = bundle["features"]

    df = pd.read_csv(DATA_PATH)

    X = df[features]
    y = df["Class"]

    # Reproduce the original held-out test split.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print("FINAL HELD-OUT TEST EVALUATION")
    print("------------------------------")
    print(f"Test transactions: {len(X_test):,}")
    print(f"Fraud transactions: {int(y_test.sum()):,}")
    print(f"Selected threshold: {threshold:.6f}")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "PR-AUC": average_precision_score(
            y_test, probabilities
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print(cm)

    # Save metrics.
    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "final_test_metrics.csv",
        index=False,
    )

    # Save confusion matrix visualization.
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"],
    ).plot(cmap="Blues")

    plt.title("FraudShield AI - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    # Save precision-recall curve.
    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities,
    )

    plt.title("FraudShield AI - Test Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "precision_recall_curve.png",
        dpi=200,
    )
    plt.close()

    print("\nSaved:")
    print("- assets/final_test_metrics.csv")
    print("- assets/confusion_matrix.png")
    print("- assets/precision_recall_curve.png")


if __name__ == "__main__":
    main()
