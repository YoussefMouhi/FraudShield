# FraudShield AI
### Financial Fraud Detection & Risk Analysis

FraudShield AI is an end-to-end machine-learning project for
detecting potentially fraudulent credit card transactions.

The project combines supervised classification, unsupervised
anomaly detection, explainable AI, and an interactive Streamlit
dashboard.

## Project Overview

Financial fraud detection presents a challenging classification
problem because fraudulent transactions are extremely rare.

This project investigates how different machine-learning
approaches perform on highly imbalanced transaction data.

### Key Features

- Credit card fraud classification
- Five machine-learning algorithms
- Fraud risk scoring
- Anomaly detection
- SHAP model explanations
- Interactive Streamlit dashboard
- Model performance comparison
- Automated evaluation tests

## Dataset

The project uses the publicly available Credit Card Fraud
Detection dataset.

Dataset characteristics:

- 284,807 transactions
- 492 fraudulent transactions
- 30 input features
- Highly imbalanced binary classification

Features V1–V28 are anonymized PCA-transformed variables.

The dataset also includes Time, Amount, and Class.

The raw dataset is not included in this repository.

Dataset source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Machine Learning Models

Five models were implemented:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Isolation Forest
5. Neural Network Autoencoder

Supervised models learn from labeled examples.

Unsupervised models identify unusual transaction patterns.

## Validation Results

| Model | PR-AUC | F1 |
|---|---:|---:|
| XGBoost | 0.8214 | 0.8508 |
| Random Forest | 0.7840 | 0.8191 |
| Logistic Regression | 0.6807 | 0.7676 |
| Autoencoder | 0.3016 | 0.4170 |
| Isolation Forest | 0.0883 | 0.1939 |

XGBoost achieved the strongest validation performance.

## Final XGBoost Test Results

The selected XGBoost model was evaluated using a
previously reserved test partition.

| Metric | Score |
|---|---:|
| PR-AUC | 0.8604 |
| ROC-AUC | 0.9785 |
| Precision | 0.9070 |
| Recall | 0.7959 |
| F1-score | 0.8478 |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,856 | 8 |
| Actual Fraud | 20 | 78 |

The model identified 78 of 98 fraudulent transactions,
with 8 false alarms.

### Evaluation Notes

The dataset was divided using stratified random sampling.

The model selection and decision threshold were determined
using validation data.

An earlier exploratory Logistic Regression baseline inspected
the same test partition. Consequently, the test set should not
be described as completely untouched throughout development.

Threshold-dependent validation metrics were calculated using
thresholds selected on the validation set itself.

A chronological evaluation would be an important extension
for real-world fraud detection.

## Explainable AI

SHAP is used to explain XGBoost predictions.

The project includes:

- Global feature importance
- SHAP summary visualization
- Individual transaction explanations

Because most features are anonymized PCA components,
their original financial meaning is unavailable.

## Interactive Dashboard

The Streamlit application provides:

- Dataset overview
- Transaction-level fraud predictions
- Explainability visualizations
- Model comparison

## Installation

Python 3.12 is recommended.

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd FraudShield_AI
```

Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Credit Card Fraud Detection dataset from Kaggle
and place `creditcard.csv` inside the `data/` directory.

Launch the dashboard:

```bash
python -m streamlit run app.py
```

## Testing

Run:

```bash
python -m pytest tests/ -v
```

The evaluation tests require the dataset, trained model,
and saved metrics.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- SHAP
- Streamlit
- Plotly
- Matplotlib
- Pytest

## Limitations

- The dataset contains anonymized features.
- The evaluation uses a stratified random split.
- Fraud patterns may change over time.
- Results do not establish production readiness.
- Real-world deployment requires monitoring, security,
  calibration, and operational validation.

## Future Improvements

- Chronological fraud evaluation
- Cost-sensitive threshold optimization
- Model monitoring and drift detection
- Real-time prediction API
- Experiment tracking
- Cloud deployment

## Author

Youssef MOUHI

Computer Engineering | Data Science & Artificial Intelligence
