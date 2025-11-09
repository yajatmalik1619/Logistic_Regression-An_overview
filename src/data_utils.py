import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_breast_cancer_df():
    b = load_breast_cancer(as_frame=True)
    df = pd.concat([b.data, b.target.rename("target")], axis=1)
    return df

def load_titanic_csv(path="data/raw/titanic.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Titanic CSV not found at {path}. Put data in data/raw/")
    return pd.read_csv(path)

def load_pima_csv(path="data/raw/pima_diabetes.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pima CSV not found at {path}. Put data in data/raw/")
    return pd.read_csv(path)

def load_bank_marketing_csv(path="data/raw/bank-additional/bank-additional-full.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bank Marketing CSV not found at {path}. Put data in data/raw/")
    return pd.read_csv(path, sep=';')
