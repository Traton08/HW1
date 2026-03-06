import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="COVID-19 Mortality Prediction", layout="wide")

FEATURES = [
    "SEX", "HOSPITALIZED", "PNEUMONIA", "AGE", "PREGNANT", "DIABETES", "COPD",
    "ASTHMA", "IMMUNOSUPPRESSION", "HYPERTENSION", "OTHER_DISEASE",
    "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO", "COVID_POSITIVE"
]

@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load("models/logistic.joblib"),
        "Decision Tree": joblib.load("models/tree.joblib"),
        "Random Forest": joblib.load("models/rf.joblib"),
        "XGBoost": joblib.load("models/xgb.joblib"),
    }
    results_df = pd.read_csv("reports/model_results.csv")
    best_params = joblib.load("reports/best_params.joblib")
    return models, results_df, best_params

models, results_df, best_params = load_artifacts()

st.title("COVID-19 Mortality Prediction — End-to-End App")

tabs = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

with tabs[0]:
    st.write("""
Dataset and prediction task
This project uses a COVID-19 patient outcomes dataset with 10,000 patient records and 17 total columns. Each row represents one patient and includes demographic information such as age and sex, clinical indicators of severity like hospitalized and pneumonia, COVID status whether positive or negative, and multiple preexisting conditions such as diabetes, hypertension, obesity, asthma, etc. The prediction target is death with a binary label that a 1 indicates the patient died and 0 indicates the patient survived.

Why this matters (so what)
Predicting a patient’s risk of death is useful because it helps healthcare teams understand who might need extra attention or resources. Even when doctors and nurses are very experienced, having a consistent prediction model can catch the high risk patients earlier and show which factors are contributing to their risk. It also supports good decision making, especially when the team is busy or short staffed.

Approach and key findings
I started by looking through the dataset using basic descriptive analytics to understand the target variable, spot any patterns in the features, and check how different variables were related. After that, I trained several models using a fixed train/test split (random_state=42). I included Logistic Regression as a baseline, along with a Decision Tree, Random Forest, and XGBoost. To compare them, I looked at test set metrics like F1 and ROC-AUC. I used cross validation to tune the hyperparameters and choose the best versions for the tree-based models.

For explainability, I used SHAP on the best performing tree-based model to see which features had the biggest impact on the mortality predictions. The SHAP results showed that age and severity related factors, especially hospitalized and pneumonia, were the strongest contributors to higher predicted risk. Others like diabetes and hypertension also added risk in many cases. In the Streamlit app, I’ve included an interactive tool where users can adjust different feature values and watch how the predicted probability changes live.
""")

with tabs[1]:
    st.subheader("Descriptive Analytics")
    st.info("Add your saved EDA plots into the reports/ folder (target + 4 plots + heatmap).")

with tabs[2]:
    st.subheader("Model Performance")

    st.write("### Model Comparison Table")
    st.dataframe(results_df, use_container_width=True)

    st.write("### F1 Comparison")
    fig = plt.figure()
    plt.bar(results_df["Model"], results_df["F1"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("F1 Score")
    plt.title("Model Comparison (F1)")
    plt.tight_layout()
    st.pyplot(fig)

    st.write("### Best Hyperparameters (Grid Search)")
    st.json(best_params)

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

with tabs[3]:
    st.subheader("Explainability (SHAP) — Random Forest")

    for path, caption in [
        ("reports/shap_beeswarm_rf.png", "SHAP Summary (Beeswarm) — Random Forest"),
        ("reports/shap_bar_rf.png", "SHAP Feature Importance (Bar) — Random Forest"),
        ("reports/shap_waterfall_rf.png", "SHAP Waterfall Example — Random Forest"),
    ]:
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)

    st.divider()
    st.subheader("Interactive Prediction")

    chosen_model = st.selectbox("Select model", list(models.keys()), index=2)

    c1, c2, c3 = st.columns(3)
    with c1:
        AGE = st.slider("AGE", 0, 110, 50)
        SEX = st.selectbox("SEX (0/1)", [0, 1], index=0)
        PREGNANT = st.selectbox("PREGNANT (0/1)", [0, 1], index=0)
        COVID_POSITIVE = st.selectbox("COVID_POSITIVE (0/1)", [0, 1], index=1)
    with c2:
        HOSPITALIZED = st.selectbox("HOSPITALIZED (0/1)", [0, 1], index=0)
        PNEUMONIA = st.selectbox("PNEUMONIA (0/1)", [0, 1], index=0)
        DIABETES = st.selectbox("DIABETES (0/1)", [0, 1], index=0)
        HYPERTENSION = st.selectbox("HYPERTENSION (0/1)", [0, 1], index=0)
    with c3:
        OBESITY = st.selectbox("OBESITY (0/1)", [0, 1], index=0)
        COPD = st.selectbox("COPD (0/1)", [0, 1], index=0)
        ASTHMA = st.selectbox("ASTHMA (0/1)", [0, 1], index=0)
        RENAL_CHRONIC = st.selectbox("RENAL_CHRONIC (0/1)", [0, 1], index=0)

    user_row = {f: 0 for f in FEATURES}
    user_row.update({
        "AGE": AGE,
        "SEX": SEX,
        "PREGNANT": PREGNANT,
        "COVID_POSITIVE": COVID_POSITIVE,
        "HOSPITALIZED": HOSPITALIZED,
        "PNEUMONIA": PNEUMONIA,
        "DIABETES": DIABETES,
        "HYPERTENSION": HYPERTENSION,
        "OBESITY": OBESITY,
        "COPD": COPD,
        "ASTHMA": ASTHMA,
        "RENAL_CHRONIC": RENAL_CHRONIC,
    })
    user_df = pd.DataFrame([user_row], columns=FEATURES)

    if st.button("Predict"):
        proba = float(models[chosen_model].predict_proba(user_df)[:, 1][0])
        pred = int(proba >= 0.5)
        st.write(f"Predicted class (1=Death): {pred}")
        st.write(f"Predicted probability: {proba:.3f}")
