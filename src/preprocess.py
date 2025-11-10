from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_numeric_transformer(strategy: str = "median"):
    return Pipeline([
        ("imputer", SimpleImputer(strategy=strategy)),
        ("scaler", StandardScaler())
    ])

def build_categorical_transformer(handle_unknown: str = "ignore"):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown=handle_unknown, sparse=False))
    ])

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]):
    transformers = []
    if numeric_features:
        transformers.append(("num", build_numeric_transformer(), numeric_features))
    if categorical_features:
        transformers.append(("cat", build_categorical_transformer(), categorical_features))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor

def split_features_target(df: pd.DataFrame, target_col: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


