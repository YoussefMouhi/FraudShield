from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_PATH = Path("models/xgboost_fraud_detector.joblib")
DATA_PATH = Path("data/creditcard.csv")
OUTPUT_DIR = Path("assets")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_csv(DATA_PATH)

    # Recreate the original development/validation/test split.
    from sklearn.model_selection import train_test_split

    X = df[features]
    y = df["Class"]

    X_dev, _, y_dev, _ = train_test_split(
        X,
        y,
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

    # Use validation samples, never the reserved test set.
    sample = X_val.sample(
        n=min(300, len(X_val)),
        random_state=42,
    )

    print("Creating SHAP explainer...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample)

    print("Generating SHAP summary plot...")

    shap.summary_plot(
        shap_values.values,
        sample,
        show=False,
        max_display=15,
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "shap_summary.png"

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"SHAP summary saved to: {output}")

    # Global feature importance
    importance = pd.DataFrame({
        "Feature": features,
        "Mean_Absolute_SHAP": np.abs(
            shap_values.values
        ).mean(axis=0),
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )

    importance.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False,
    )

    print("\nTop 10 important features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
