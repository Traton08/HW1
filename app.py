import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="COVID-19 Mortality Prediction", layout="wide")

st.title("COVID-19 Mortality Prediction — End-to-End App")

tabs = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

# -------------------------
# Tab 1 — Executive Summary
# -------------------------
with tabs[0]:
    st.write("""
Dataset and prediction task
This project uses a COVID-19 patient outcomes dataset with 10,000 patient records and 17 total columns. Each row represents one patient and includes demographic information such as age and sex, clinical indicators of severity like hospitalized and pneumonia, COVID status whether positive or negative, and multiple preexisting conditions such as diabetes, hypertension, obesity, asthma etc. The prediction target is death with a binary label that a 1 indicates the patient died and 0 indicates the patient survived.

Why this matters (so what)
Predicting a patient’s risk of death is useful because it helps healthcare teams understand who might need extra attention or resources. Even when doctors and nurses are very experienced, having a consistent prediction model can catch the high risk patients earlier and show which factors are contributing to their risk. It also supports good decision making, especially when the team is busy or short staffed.

Approach and key findings
I started by looking through the dataset using basic descriptive analytics to understand the target variable, spot any patterns in the features, and check how different variables were related. After that, I trained several models using a fixed train/test split (random_state=42). I included Logistic Regression as a baseline, along with a Decision Tree, Random Forest, & XGBoost. To compare them, I looked at each test set metrics like F1 and ROC AUC. I used cross validation to tune the hyperparameters and choose the best versions for the tree-based models.

For explainability, I used SHAP on the best performing tree-based model to see which features had the biggest impact on the mortality predictions. The SHAP results showed that age and severity related factors, especially hospitalized and pneumonia, were the strongest contributors to higher predicted risk. Others like diabetes and hypertension also added risk in many cases.
""")

# -------------------------
# Tab 2 — Descriptive Analytics
# -------------------------
with tabs[1]:
    st.subheader("Descriptive Analytics")
    st.write("Key EDA plots saved from the notebook:")

    eda_files = [
        ("reports/target_distribution.png", "Target Distribution"),
        ("reports/eda1.png", "EDA Plot 1"),
        ("reports/eda2.png", "EDA Plot 2"),
        ("reports/eda3.png", "EDA Plot 3"),
        ("reports/eda4.png", "EDA Plot 4"),
        ("reports/corr_heatmap.png", "Correlation Heatmap"),
    ]
    shown_any = False
    for path, caption in eda_files:
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
            shown_any = True
    if not shown_any:
        st.warning("EDA images not found in reports/. Upload your saved EDA PNGs if required.")

# -------------------------
# Tab 3 — Model Performance
# -------------------------
with tabs[2]:
    st.subheader("Model Performance")

    if os.path.exists("reports/model_results.csv"):
        results_df = pd.read_csv("reports/model_results.csv")
        st.write("### Model Comparison Table")
        st.dataframe(results_df, use_container_width=True)

        if "Model" in results_df.columns and "F1" in results_df.columns:
            st.write("### F1 Comparison")
            fig = plt.figure()
            plt.bar(results_df["Model"], results_df["F1"])
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("F1 Score")
            plt.title("Model Comparison (F1)")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("Missing reports/model_results.csv")

    st.write("### ROC Curves")
    roc_files = [
        ("reports/roc_logistic.png", "ROC - Logistic Regression"),
        ("reports/roc_tree.png", "ROC - Decision Tree"),
        ("reports/roc_rf.png", "ROC - Random Forest"),
        ("reports/roc_xgb.png", "ROC - XGBoost"),
    ]
    cols = st.columns(2)
    i = 0
    for path, caption in roc_files:
        if os.path.exists(path):
            with cols[i % 2]:
                st.image(path, caption=caption, use_container_width=True)
            i += 1
        else:
            st.warning(f"Missing {path} — upload it into reports/.")

    st.write("### MLP Training Loss (from notebook)")
    if os.path.exists("reports/mlp_loss.png"):
        st.image("reports/mlp_loss.png", use_container_width=True)

# -------------------------
# Tab 4 — Explainability & Interactive Prediction
# -------------------------
with tabs[3]:
    st.subheader("Explainability (SHAP)")

    for path, caption in [
        ("reports/shap_beeswarm_rf.png", "SHAP Summary (Beeswarm) — Random Forest"),
        ("reports/shap_bar_rf.png", "SHAP Feature Importance (Bar) — Random Forest"),
        ("reports/shap_waterfall_rf.png", "SHAP Waterfall Example — Random Forest"),
    ]:
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)

    st.divider()
    st.subheader("Interactive Prediction (Simple Demo)")
    st.write("This demo lets you adjust a few key features and returns a simple risk score (not the trained model output).")

    age = st.slider("AGE", 0, 110, 50)
    hospitalized = st.selectbox("HOSPITALIZED (0/1)", [0, 1], index=0)
    pneumonia = st.selectbox("PNEUMONIA (0/1)", [0, 1], index=0)
    diabetes = st.selectbox("DIABETES (0/1)", [0, 1], index=0)
    hypertension = st.selectbox("HYPERTENSION (0/1)", [0, 1], index=0)

    # Simple heuristic score
    score = 0
    score += 2 if age >= 65 else 0
    score += 2 if hospitalized == 1 else 0
    score += 2 if pneumonia == 1 else 0
    score += 1 if diabetes == 1 else 0
    score += 1 if hypertension == 1 else 0

    st.write(f"Estimated risk score (0–8): **{score}**")
    st.write("Note: This score is a simple demo for interactivity. Model results are shown in Tabs 2–3 and SHAP in Tab 4.")
