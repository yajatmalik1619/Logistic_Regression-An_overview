import sys
from pathlib import Path
import pandas as pd
from joblib import load

# Make src importable
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_DIR))

from src.data_utils import load_breast_cancer_df

def compute_range(series):
    """Compute safe and meaningful min/max + default median."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    # Safe min/max range
    min_val = max(0, q1 - 3 * iqr)
    max_val = q3 + 3 * iqr
    default = series.median()

    return float(min_val), float(max_val), float(default)

def main():
    MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_model.joblib"

    if not MODEL_PATH.exists():
        print("Model not found. Please run: python -m src.train_model")
        return

    model = load(MODEL_PATH)
    print("Loaded model successfully.\n")
    # Load dataset to get feature structure
    df = load_breast_cancer_df()
    X = df.drop(columns=["target"])
    feature_names = X.columns
    medians = X.median()
    # Subset of features to prompt
    demo_features = [
        "mean radius",
        "mean texture",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean concave points",
        "mean symmetry",
        "mean fractal dimension"
    ]
    print("Enter feature values.")
    print("Press Enter to use default median.\n")
    user_values = {}

    for feat in demo_features:
        series = X[feat]
        min_val, max_val, default = compute_range(series)

        prompt = (
            f"{feat}\n"
            f"  Range:   [{min_val:.4f}  —  {max_val:.4f}]\n"
            f"  Default:  {default:.4f}\n"
            f"Enter value (or press Enter for default): "
        )

        val = input(prompt).strip()

        if val == "":
            user_values[feat] = default
        else:
            try:
                val_float = float(val)
                if val_float < min_val or val_float > max_val:
                    print("  ⚠ Out of range. Using default.")
                    user_values[feat] = default
                else:
                    user_values[feat] = val_float
            except:
                print("  ⚠ Invalid input. Using default.")
                user_values[feat] = default

        print()  # blank line for formatting

    # Build full row
    row = medians.copy()
    for feat, val in user_values.items():
        row[feat] = val
    X_row = pd.DataFrame([row[feature_names]])

    # Predict
    prob = float(model.predict_proba(X_row)[0][1])
    pred_label = "Benign" if prob >= 0.5 else "Malignant"
    # Final output
    print("\n Prediction Result")
    print("---------------------")
    print(f"Predicted Class       : {pred_label}")
    print(f"Probability (Benign)  : {prob:.4f}")
    print("\nValues used:")
    for feat in demo_features:
        print(f"  {feat:25s} : {X_row.iloc[0][feat]:.4f}")
    print("\nDone.\n")
if __name__ == "__main__":
    main()
