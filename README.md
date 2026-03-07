# COVID-19 Mortality Prediction

A machine learning web app that predicts COVID-19 patient mortality risk based on demographic and clinical features.

---

## What's in this repo

| File/Folder | What it is |
|---|---|
| `COVID_Project_Final_Submission (1).ipynb` | Jupyter notebook with all analysis, model training, and EDA code |
| `app.py` | Streamlit web app code |
| `requirements.txt` | List of Python packages needed to run everything |
| `models/` | Saved model files |
| `reports/` | Saved plots and charts used in the app |

---

## How to run the Jupyter notebook

1. Open [Google Colab](https://colab.research.google.com) or Jupyter
2. Upload `COVID_Project_Final_Submission (1).ipynb`
3. Run the first cell to install dependencies
4. Run all cells from top to bottom

---

## How to run the Streamlit app locally

1. Make sure you have Python installed
2. Clone this repo:
   ```
   git clone https://github.com/Traton08/HW1
   cd HW1
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```
5. It will open automatically in your browser

---

## Live App

The app is deployed and accessible here:
🔗 https://your-app-url.streamlit.app

*(Replace with your actual Streamlit URL)*

---

## Project Summary

This project uses a COVID-19 patient outcomes dataset with 10,000 records. The goal is to predict whether a patient will die based on features like age, pre-existing conditions, and hospitalization status. Models trained include Logistic Regression, Decision Tree, Random Forest, and XGBoost. SHAP values are used to explain model predictions.
