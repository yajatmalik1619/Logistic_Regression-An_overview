import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

def plot_target_balance(df, target_col="target"):
    sns.countplot(x=target_col, data=df)
    plt.title("Target class balance")
    plt.show()

def plot_roc(y_true, y_prob, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})" if label else f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_confusion(y_true, y_pred, labels=None):
    cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels)
    cm.ax_.set_title("Confusion Matrix")
    plt.show()
