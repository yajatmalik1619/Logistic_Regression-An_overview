import sys
from pathlib import Path
import os

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  
SRC_DIR = PROJECT_ROOT / "src"

for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.append(p)

import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from src.data_utils import load_breast_cancer_df
    from src.visualize import plot_roc, plot_confusion, plot_coefficients
except Exception as e:
    st.error(f"Failed to import project modules from src/: {e}")
    raise

MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_model.joblib"

if not MODEL_PATH.exists():
    st.error(f"Model file not found at:\n{MODEL_PATH}\n\nPlease run the training script (e.g. `python -m src.train_model`) to create it.")
    st.stop()

model = load(MODEL_PATH)

df_train = load_breast_cancer_df()
X_train = df_train.drop(columns=["target"])
feature_names = list(X_train.columns)
feature_defaults = X_train.median()

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🩺 Breast Cancer Prediction App")
st.write("Provide values for a small set of features. The rest will use median defaults from the training data.")

demo_features = [
    "mean radius", "mean texture", "mean smoothness", "mean compactness",
    "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"
]

st.sidebar.header("Input features (demo subset)")
user_inputs = {}
for feat in demo_features:
    if feat in X_train.columns:
        default = float(feature_defaults[feat])
        q1 = float(X_train[feat].quantile(0.25))
        q3 = float(X_train[feat].quantile(0.75))
        rng = max(1e-3, q3 - q1)
        min_val = float(max(0.0, q1 - 3 * rng))
        max_val = float(q3 + 3 * rng)
        user_inputs[feat] = st.sidebar.number_input(feat, min_value=min_val, max_value=max_val, value=default, format="%.4f")
    else:
        user_inputs[feat] = st.sidebar.number_input(feat, value=0.0)

if st.button("Predict"):
    row = feature_defaults.copy()
    for feat, val in user_inputs.items():
        if feat in row.index:
            row[feat] = float(val)
    try:
        X_row = pd.DataFrame([row[feature_names]])
    except Exception:
        X_row = pd.DataFrame([row]).reindex(columns=feature_names)

    try:
        prob = float(model.predict_proba(X_row)[:, 1][0])  
        pred_label = "Benign" if prob > 0.5 else "Malignant"
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        raise

    st.success(f"Prediction: **{pred_label}**")
    st.info(f"Predicted probability (positive class): {prob:.4f}")

    print(f"[APP PREDICT] label={pred_label}, prob={prob:.4f}")
    try:
        print("Input row (first 8 features):", X_row.iloc[0].head(8).to_dict())
    except Exception:
        print("Input row (full):", X_row.iloc[0].to_dict())

PLOT_DIR = PROJECT_ROOT / "models" / "plots"
st.markdown("---")
st.header("Saved evaluation plots")
if st.checkbox("Show saved evaluation plots (ROC, Confusion, Coeffs)"):
    if (PLOT_DIR / "roc_curve.png").exists():
        st.image(str(PLOT_DIR / "roc_curve.png"), caption="ROC Curve")
    else:
        st.info("ROC plot not found (run predictions script to generate plots).")
    if (PLOT_DIR / "confusion_matrix.png").exists():
        st.image(str(PLOT_DIR / "confusion_matrix.png"), caption="Confusion Matrix")
    if (PLOT_DIR / "coefficients.png").exists():
        st.image(str(PLOT_DIR / "coefficients.png"), caption="Top coefficients")

st.markdown("---")
st.header("Model evaluation (test set)")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X_all = X_train
y_all = df_train["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

try:
    y_prob_test = model.predict_proba(X_te)[:, 1]
    y_pred_test = model.predict(X_te)
    acc = (y_pred_test == y_te).mean()
    auc_val = roc_auc_score(y_te, y_prob_test)
    st.write(f"Accuracy: **{acc:.3f}**  —  ROC AUC: **{auc_val:.3f}**")

    if st.checkbox("Show classification report"):
        st.text(classification_report(y_te, y_pred_test, digits=3))

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Show ROC curve (live)"):
            plt_obj = plot_roc(y_te, y_prob_test, label="LogisticRegression")
            st.pyplot(plt_obj.gcf())
    with col2:
        if st.button("Show Confusion Matrix (live)"):
            plt_obj = plot_confusion(y_te, y_pred_test, labels=["malignant","benign"])
            st.pyplot(plt_obj.gcf())
    with col3:
        if st.button("Show Top coefficients (live)"):
            feature_names_all = list(X_all.columns)
            coefs = model.named_steps["clf"].coef_.ravel()
            plt_obj = plot_coefficients(coefs, feature_names_all, top_n=15)
            st.pyplot(plt_obj.gcf())

except Exception as e:
    st.warning(f"Could not compute test metrics or plots: {e}")


