import os
from data_utils import load_breast_cancer_df
from preprocess import build_preprocessor, split_features_target
from logistic_sklearn import build_pipeline, grid_search_pipeline, save_model
from sklearn.model_selection import train_test_split

def main():
    df = load_breast_cancer_df()
    X, y = split_features_target(df, target_col="target")

    numeric_features = list(X.select_dtypes(include=["int64", "float64"]).columns)
    categorical_features = [] 

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipe = build_pipeline(preprocessor, solver="liblinear", penalty="l2")

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    gs = grid_search_pipeline(pipe, X_train, y_train, param_grid={"clf__C":[0.01, 0.1, 1.0]}, cv=5, n_jobs=1)
    print("Best params:", gs.best_params_)
    best = gs.best_estimator_

    os.makedirs("models", exist_ok=True)
    save_model(best, "models/breast_cancer_model.joblib")
    print("Saved model to models/breast_cancer_model.joblib")

if __name__ == "__main__":
    main()
