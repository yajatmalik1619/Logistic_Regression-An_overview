# Logistic_Regression-An_overview
This repository aims to explain as well as show case a few examples and use cases of logistic regression, giving a wide essence of what and how it works.
# Logistic Regression — Multi-Dataset Machine Learning Project

## Overview
This project explores **Logistic Regression** across four well-known datasets:
- Breast Cancer Wisconsin
- Titanic Survival
- Pima Indians Diabetes
- Bank Marketing

The goal is to:
1. Perform complete **EDA** and preprocessing (imputation, encoding, scaling)
2. Train models using:
   - Logistic Regression (from scratch with NumPy)
   - Logistic Regression (using scikit-learn)
3. Compare model performance and interpret coefficients
4. Visualize results and deploy a simple **Streamlit app** for demo predictions.

---

## Folder Structure
src/ → all reusable Python modules
notebooks/ → EDA + modeling notebooks
data/ → raw and processed datasets
models/ → trained models (.joblib files)
demos/ → Streamlit web demo


---

## Setup Instructions (enter in bash)
```bash
git clone https://github.com/yajatmalik1619/Logistic_Regression-An_overview
cd logistic-regression-ml-project
bash setup.sh
```

---


### To run jupyter notebook (in bash)
jupyter lab

### To view demo
streamlit run demos/app.py

### Topics covered
- Data preprocessing (imputation, encoding, scaling)
- Feature engineering
- Logistic regression theory & implementation
- Regularization (L1/L2)
- Model evaluation (Accuracy, ROC, F1)
- Decision boundary visualization
- Streamlit model deployment

