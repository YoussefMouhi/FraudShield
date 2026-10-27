from pathlib import Path

import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="assets/fraudshield_icon.png",
    layout="wide",
)


MODEL_PATH = Path("models/xgboost_fraud_detector.joblib")
DATA_PATH = Path("data/creditcard.csv")

col_logo, col_title = st.columns([1, 10], vertical_alignment="center")

with col_logo:
    st.image("assets/fraudshield_icon.png", width=55)

with col_title:
    st.title("FraudShield AI")

if not MODEL_PATH.exists():
    st.error("Model not found. Train and save XGBoost first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

overview, analysis, explainability, comparison = st.tabs([
    "📊 Overview",
    "🔍 Transaction Analysis",
    "🧠 Explainability",
    "📈 Model Comparison",
])

with overview:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    fraud_count = int(df["Class"].sum())
    total = len(df)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraudulent Transactions", f"{fraud_count:,}")
    c3.metric("Legitimate Transactions", f"{total - fraud_count:,}")
    c4.metric("Fraud Rate", f"{100 * fraud_count / total:.3f}%")

    counts = df["Class"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Class"] = counts["Class"].map({
        0: "Legitimate",
        1: "Fraud",
    })

    fig = px.bar(
        counts,
        x="Class",
        y="Count",
        color="Class",
        title="Transaction Class Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

with analysis:
    st.subheader("Transaction Risk Analysis")
    st.write(
        "Select a transaction from the dataset to inspect "
        "its model-generated fraud risk score."
    )

    transaction_index = st.number_input(
        "Transaction index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    transaction = df.iloc[[transaction_index]][features]

    if st.button("Analyze Transaction", type="primary"):
        probability = float(model.predict_proba(transaction)[0, 1])
        prediction = int(probability >= threshold)

        c1, c2 = st.columns(2)

        c1.metric("Model Risk Score", f"{probability * 100:.2f}%")
        c2.metric(
            "Classification",
            "Suspicious" if prediction else "Not Flagged",
        )

        if prediction:
            st.error("⚠️ Transaction flagged for further review.")
        else:
            st.success("Transaction not flagged by the current model.")

        st.caption(
            "The risk score is a model output, not a calibrated "
            "real-world fraud probability."
        )

        st.write("Actual dataset label:", int(df.iloc[transaction_index]["Class"]))

        with st.expander("Transaction Features"):
            st.dataframe(transaction.T, use_container_width=True)

        st.subheader("Why did the model make this prediction?")

        with st.spinner("Calculating SHAP explanations..."):
            explainer = shap.TreeExplainer(model)
            explanation = explainer(transaction)

            fig, ax = plt.subplots(figsize=(10, 7))

            shap.plots.waterfall(
                explanation[0],
                max_display=15,
                show=False,
            )

            st.pyplot(plt.gcf())
            plt.close("all")

        st.caption(
            "SHAP explains contributions to the model output. "
            "The anonymized PCA features do not have known "
            "real-world financial meanings."
        )

with explainability:
    st.subheader("SHAP Model Explainability")

    shap_path = Path("assets/shap_summary.png")

    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True)
    else:
        st.warning("SHAP visualization not found.")

    importance_path = Path("assets/shap_feature_importance.csv")

    if importance_path.exists():
        importance = pd.read_csv(importance_path)

        fig = px.bar(
            importance.head(15),
            x="Mean_Absolute_SHAP",
            y="Feature",
            orientation="h",
            title="Top 15 Features by Mean Absolute SHAP Value",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "V1–V28 are anonymized PCA components. Their original "
        "financial meanings are not available."
    )

with comparison:
    st.subheader("Machine Learning Model Comparison")

    comparison_path = Path("models/model_comparison.csv")

    if comparison_path.exists():
        results = pd.read_csv(comparison_path)

        st.dataframe(results, use_container_width=True)

        fig = px.bar(
            results,
            x="Model",
            y="PR-AUC",
            color="Model",
            title="Validation PR-AUC by Model",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Model comparison results are from validation data. "
            "Final held-out test evaluation is pending."
        )
    else:
        st.warning("Model comparison results not found.")

st.divider()
st.caption(
    "FraudShield AI | Machine Learning Portfolio Project | "
    "Research and demonstration only"
)
