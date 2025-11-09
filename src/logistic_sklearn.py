from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def build_pipeline(preprocessor, solver="liblinear", penalty="l2", C=1.0, max_iter=1000):
    clf = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=max_iter)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])
    return pipe

def grid_search_pipeline(pipe, X, y, param_grid=None, cv=5, scoring="roc_auc", n_jobs=1):
    if param_grid is None:
        param_grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"]
        }
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv_obj, scoring=scoring, n_jobs=n_jobs)
    gs.fit(X, y)
    return gs

def save_model(obj, path):
    dump(obj, path)

def load_model(path):
    return load(path)
