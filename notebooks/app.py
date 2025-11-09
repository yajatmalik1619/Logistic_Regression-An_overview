import streamlit as st
from joblib import load
from pathlib import Path
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_model.joblib"

if not MODEL_PATH.exists():
    st.error(f"Model file not found at {MODEL_PATH}. Run training first.")
    st.stop()

model = load(MODEL_PATH)

from src.data_utils import load_breast_cancer_df
df_train = load_breast_cancer_df()
X_train = df_train.drop(columns=["target"])
feature_names = list(X_train.columns) 

feature_defaults = X_train.median()

st.title("Breast Cancer Prediction App")

st.write("Provide values for a small set of features. The rest will use median defaults from the training data.")

demo_features = [
    "mean radius", "mean texture", "mean smoothness", "mean compactness",
    "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"
]

user_inputs = {}
for feat in demo_features:
    q1 = X_train[feat].quantile(0.25)
    q3 = X_train[feat].quantile(0.75)
    default = float(X_train[feat].median())
    min_val = float(max(0, q1 - 3*(q3 - q1)))
    max_val = float(q3 + 3*(q3 - q1))
    user_inputs[feat] = st.sidebar.number_input(feat, min_value=min_val, max_value=max_val, value=default)

if st.button("Predict"):
    row = feature_defaults.copy()

    for feat, val in user_inputs.items():
        row[feat] = float(val)

    X_row = pd.DataFrame([row[feature_names]])  

    prob = model.predict_proba(X_row)[:, 1][0]  
    pred_label = "Benign" if prob > 0.5 else "Malignant"

    st.success(f"Prediction: **{pred_label}**")
    st.info(f"Predicted probability (positive class / benign): {prob:.3f}")
