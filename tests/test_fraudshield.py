from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


MODEL_PATH = Path("models/xgboost_fraud_detector.joblib")
DATA_PATH = Path("data/creditcard.csv")
METRICS_PATH = Path("assets/final_test_metrics.csv")


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_data(bundle):
    df = pd.read_csv(DATA_PATH)

    X = df[bundle["features"]]
    y = df["Class"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    return X_test, y_test


def test_model_exists():
    assert MODEL_PATH.exists()


def test_model_bundle(bundle):
    assert "model" in bundle
    assert "threshold" in bundle
    assert "features" in bundle

    assert 0 <= bundle["threshold"] <= 1
    assert len(bundle["features"]) == 30


def test_dataset_structure(bundle):
    df = pd.read_csv(DATA_PATH)

    assert "Class" in df.columns
    assert set(bundle["features"]).issubset(df.columns)
    assert set(df["Class"].unique()).issubset({0, 1})


def test_predictions_are_valid(bundle, test_data):
    X_test, _ = test_data

    probabilities = bundle["model"].predict_proba(
        X_test.iloc[:100]
    )[:, 1]

    assert len(probabilities) == 100
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_final_metrics_reproducible(bundle, test_data):
    X_test, y_test = test_data

    probabilities = bundle["model"].predict_proba(X_test)[:, 1]
    predictions = (
        probabilities >= bundle["threshold"]
    ).astype(int)

    expected = pd.read_csv(METRICS_PATH).iloc[0]

    actual = {
        "PR-AUC": average_precision_score(y_test, probabilities),
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

    for metric, value in actual.items():
        assert value == pytest.approx(
            expected[metric],
            abs=1e-6,
        )
