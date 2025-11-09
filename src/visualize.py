import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

sns.set_theme(style="whitegrid")

def plot_roc(y_true, y_prob, label=None, figsize=(6, 6)):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})" if label else f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt

def plot_confusion(y_true, y_pred, labels=None, figsize=(5, 4), cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = ["class 0", "class 1"]
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return plt

def plot_coefficients(coefs, feature_names, top_n=15, figsize=(8, 6)):
    df = pd.DataFrame({"feature": feature_names, "coef": np.asarray(coefs).ravel()})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_n).sort_values("coef")
    colors = df["coef"].apply(lambda v: "tab:blue" if v >= 0 else "tab:orange").tolist()
    plt.figure(figsize=figsize)
    plt.barh(df["feature"], df["coef"], color=colors)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.xlabel("Coefficient (signed)")
    plt.title(f"Top {top_n} coefficients (by absolute value)")
    plt.tight_layout()
    return plt


def plot_precision_recall(y_true, y_prob, figsize=(6, 6)):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(precision, recall)
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    return plt


def save_figure(plt_obj, path, dpi=150):
    plt_obj.savefig(path, dpi=dpi, bbox_inches="tight")
    plt_obj.close()
