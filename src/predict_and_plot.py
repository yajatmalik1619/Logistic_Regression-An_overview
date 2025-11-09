from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "breast_cancer_model.joblib"
PLOT_DIR = ROOT / "models" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found at {MODEL_PATH}. Run scripts/train.py first.")

import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import load_breast_cancer_df
model = load(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")

df = load_breast_cancer_df()
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
try:
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
except Exception:
    pass

print("\n=== Example Predictions (first 10) ===")
sample_X = X_test.reset_index(drop=True).iloc[:10]
sample_y_true = y_test.reset_index(drop=True).iloc[:10]
sample_y_prob = y_prob[:10]
sample_y_pred = y_pred[:10]
for i in range(len(sample_X)):
    print(f"Index {i}: True={int(sample_y_true[i])} Pred={int(sample_y_pred[i])} Prob={sample_y_prob[i]:.3f}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test,y_prob):.3f}")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "roc_curve.png")
plt.close()
print(f"Saved ROC curve to {PLOT_DIR/'roc_curve.png'}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["malignant","benign"], yticklabels=["malignant","benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png")
plt.close()
print(f"Saved confusion matrix to {PLOT_DIR/'confusion_matrix.png'}")

feature_names = list(X.columns) 
coefs = model.named_steps["clf"].coef_.ravel()  
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False).head(15).sort_values("coef")

plt.figure(figsize=(8,6))
colors = coef_df["coef"].apply(lambda v: "tab:blue" if v >= 0 else "tab:orange")
plt.barh(coef_df["feature"], coef_df["coef"], color=colors)
plt.axvline(0, color="k", lw=0.5)
plt.xlabel("Coefficient")
plt.title("Top 15 Coefficients (by absolute value)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "coefficients.png")
plt.close()
print(f"Saved coefficient plot to {PLOT_DIR/'coefficients.png'}")

print("\nDone. Plots saved in:", PLOT_DIR)
