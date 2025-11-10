import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_breast_cancer_df():
    b = load_breast_cancer(as_frame=True)
    df = pd.concat([b.data, b.target.rename("target")], axis=1)
    return df

