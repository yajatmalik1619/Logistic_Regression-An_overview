import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
OUT = Path("assets")
OUT.mkdir(exist_ok=True)

# 1) Sigmoid function
x = np.linspace(-10, 10, 400)
sig = 1/(1+np.exp(-x))
plt.figure(figsize=(6,3))
plt.plot(x, sig, linewidth=2)
plt.axvline(0, color='k', lw=0.6)
plt.xlabel("z (linear combination)")
plt.ylabel("σ(z) = 1 / (1 + e^{-z})")
plt.title("Sigmoid (Logistic) function")
plt.tight_layout()
plt.savefig(OUT/"sigmoid.png", dpi=150)
plt.close()

# 2) Logistic loss for y=1 and y=0
z = np.linspace(-10, 10, 400)
loss_pos = -np.log(1/(1+np.exp(-z)))   # -log(sigmoid(z))
loss_neg = -np.log(1 - 1/(1+np.exp(-z)))  # -log(1-sigmoid(z))
plt.figure(figsize=(6,3))
plt.plot(z, loss_pos, label="loss when y=1")
plt.plot(z, loss_neg, label="loss when y=0")
plt.ylim(0,10)
plt.xlabel("z")
plt.ylabel("Log-loss")
plt.title("Logistic Loss (Cross-Entropy) for binary labels")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"logistic_loss.png", dpi=150)
plt.close()

# 3) Decision boundary example on 2 features (synthetic)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=400, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, class_sep=1.2, random_state=0)
clf = LogisticRegression().fit(X, y)

# plot
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.6)
plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap="bwr", s=30)
plt.title("Decision boundary of Logistic Regression (2 features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig(OUT/"decision_boundary.png", dpi=150)
plt.close()

# 4) Coefficient bar plot for your trained model (if model exists)
from joblib import load
model_path = Path("notebooks/models/breast_cancer_model.joblib")
if model_path.exists():
    import pandas as pd
    m = load(model_path)
    # assume pipeline: preprocessor -> clf
    try:
        coef = m.named_steps["clf"].coef_.ravel()
    except Exception:
        # if not pipeline with named step, try direct
        coef = m.coef_.ravel()
    # Load feature names from dataset
    from data_utils import load_breast_cancer_df
    df = load_breast_cancer_df()
    feature_names = list(df.drop(columns=["target"]).columns)
    dfc = pd.DataFrame({"feature": feature_names, "coef": coef})
    dfc["abs"] = dfc["coef"].abs()
    dfc = dfc.sort_values("abs", ascending=False).head(15).sort_values("coef")
    plt.figure(figsize=(8,6))
    colors = dfc["coef"].apply(lambda v: "tab:blue" if v>=0 else "tab:orange")
    plt.barh(dfc["feature"], dfc["coef"], color=colors)
    plt.axvline(0, color='k', lw=0.6)
    plt.title("Top 15 coefficients (by absolute value)")
    plt.tight_layout()
    plt.savefig(OUT/"coefficients.png", dpi=150)
    plt.close()
else:
    print("Model not found; skip coefficient plot.")
