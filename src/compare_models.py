from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


df = pd.read_csv("data/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Use exactly the same splits as the previous experiment.
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

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negative_count / positive_count,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    ),
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...", flush=True)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, probabilities
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    threshold = float(thresholds[best_index])

    predictions = (probabilities >= threshold).astype(int)

    result = {
        "Model": name,
        "PR-AUC": average_precision_score(y_val, probabilities),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(y_val, predictions, zero_division=0),
        "Threshold": threshold,
    }

    results.append(result)

    if name == "XGBoost":
        Path("models").mkdir(exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "features": X.columns.tolist(),
            },
            "models/xgboost_fraud_detector.joblib",
        )
        print("XGBoost model saved successfully.")

    print(
        f"{name}: "
        f"PR-AUC={result['PR-AUC']:.4f}, "
        f"F1={result['F1']:.4f}, "
        f"Threshold={threshold:.10f}",
        flush=True,
    )


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "PR-AUC", ascending=False
)

Path("models").mkdir(exist_ok=True)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False,
)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nBest model by validation PR-AUC:")
print(results_df.iloc[0]["Model"])

print("\nTest set remains untouched.")
