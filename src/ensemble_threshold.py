import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# ============================================================
# ShieldML - Ensemble + F2 Threshold Optimization
# ============================================================

TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")
TEST_PATH = Path("data/processed/test.csv")

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "Class"


# ============================================================
# Load data
# ============================================================

def load_dataset(path):
    df = pd.read_csv(path)

    # These columns were not used during model training
    X = df.drop(columns=[TARGET, "Hour"])
    y = df[TARGET]

    return X, y


# ============================================================
# Load trained models
# ============================================================

def load_models():
    print("\nLoading trained models...")

    rf = joblib.load(MODEL_DIR / "random_forest.joblib")
    xgb = joblib.load(MODEL_DIR / "xgboost.joblib")
    lgbm = joblib.load(MODEL_DIR / "lightgbm.joblib")

    print("Random Forest loaded.")
    print("XGBoost loaded.")
    print("LightGBM loaded.")

    return rf, xgb, lgbm


# ============================================================
# Soft Voting Ensemble
# ============================================================

def ensemble_predict(models, X):
    rf, xgb, lgbm = models

    rf_prob = rf.predict_proba(X)[:, 1]
    xgb_prob = xgb.predict_proba(X)[:, 1]
    lgbm_prob = lgbm.predict_proba(X)[:, 1]

    # Equal-weight soft voting
    ensemble_prob = (
        rf_prob +
        xgb_prob +
        lgbm_prob
    ) / 3.0

    return ensemble_prob


# ============================================================
# Find F2-optimal threshold
# ============================================================

def find_best_threshold(y_true, probabilities):

    thresholds = np.linspace(0.01, 0.99, 197)

    best_threshold = 0.5
    best_f2 = -1.0

    threshold_results = []

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        f2 = fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        threshold_results.append(
            (threshold, f2, precision, recall)
        )

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold

    return best_threshold, best_f2, threshold_results


# ============================================================
# Evaluate final model
# ============================================================

def evaluate_model(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f2 = fbeta_score(
        y_true,
        predictions,
        beta=2,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "confusion_matrix": cm,
    }


# ============================================================
# Save ROC Curve
# ============================================================

def save_roc_curve(y_true, probabilities):

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Ensemble ROC-AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ShieldML - ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "roc_curve.png",
        dpi=200
    )

    plt.close()


# ============================================================
# Save Precision-Recall Curve
# ============================================================

def save_pr_curve(y_true, probabilities):

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("ShieldML - Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "precision_recall_curve.png",
        dpi=200
    )

    plt.close()


# ============================================================
# Save Confusion Matrix
# ============================================================

def save_confusion_matrix(cm):

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title("ShieldML - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        [0, 1],
        ["Legitimate", "Fraud"]
    )

    plt.yticks(
        [0, 1],
        ["Legitimate", "Fraud"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=200
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ShieldML - Ensemble + F2 Threshold Optimization")
    print("=" * 70)

    # --------------------------------------------------------
    # Load validation and test data
    # --------------------------------------------------------

    print("\nLoading validation data...")

    X_val, y_val = load_dataset(VAL_PATH)

    print(
        f"Validation samples: {len(X_val):,}"
    )

    print("\nLoading test data...")

    X_test, y_test = load_dataset(TEST_PATH)

    print(
        f"Test samples: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models = load_models()

    # --------------------------------------------------------
    # Validation ensemble
    # --------------------------------------------------------

    print("\nGenerating validation ensemble probabilities...")

    val_probabilities = ensemble_predict(
        models,
        X_val
    )

    val_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    val_pr_auc = average_precision_score(
        y_val,
        val_probabilities
    )

    print(
        f"Validation ROC-AUC: {val_auc:.4f}"
    )

    print(
        f"Validation PR-AUC : {val_pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Optimize F2 threshold on validation
    # --------------------------------------------------------

    print("\nSearching for F2-optimal threshold...")

    best_threshold, best_val_f2, threshold_results = (
        find_best_threshold(
            y_val,
            val_probabilities
        )
    )

    print(
        f"Best validation threshold: {best_threshold:.4f}"
    )

    print(
        f"Best validation F2       : {best_val_f2:.4f}"
    )

    # --------------------------------------------------------
    # Test ensemble
    # --------------------------------------------------------

    print("\nGenerating test ensemble probabilities...")

    test_probabilities = ensemble_predict(
        models,
        X_test
    )

    # IMPORTANT:
    # The threshold was selected ONLY on validation.
    # The test set is used only for final evaluation.

    test_results = evaluate_model(
        y_test,
        test_probabilities,
        best_threshold
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST PERFORMANCE")
    print("=" * 70)

    print(
        f"ROC-AUC  : {test_results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : {test_results['pr_auc']:.4f}"
    )

    print(
        f"Precision: {test_results['precision']:.4f}"
    )

    print(
        f"Recall   : {test_results['recall']:.4f}"
    )

    print(
        f"F2 Score : {test_results['f2']:.4f}"
    )

    print(
        f"\nF2 Threshold: {best_threshold:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        test_results["confusion_matrix"]
    )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    print("\nSaving performance plots...")

    save_roc_curve(
        y_test,
        test_probabilities
    )

    save_pr_curve(
        y_test,
        test_probabilities
    )

    save_confusion_matrix(
        test_results["confusion_matrix"]
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = REPORT_DIR / "performance_report.txt"

    cm = test_results["confusion_matrix"]

    with open(report_path, "w", encoding="utf-8") as file:

        file.write(
            "ShieldML - Financial Fraud Detection\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            "Model: Equal-Weight Soft Voting Ensemble\n"
        )

        file.write(
            "Base Models: Random Forest, XGBoost, LightGBM\n\n"
        )

        file.write(
            "Threshold Selection\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            "Validation F2-optimal threshold: "
            f"{best_threshold:.4f}\n"
        )

        file.write(
            "Validation F2: "
            f"{best_val_f2:.4f}\n\n"
        )

        file.write(
            "Final Test Performance\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"ROC-AUC: {test_results['roc_auc']:.4f}\n"
        )

        file.write(
            f"PR-AUC: {test_results['pr_auc']:.4f}\n"
        )

        file.write(
            f"Precision: {test_results['precision']:.4f}\n"
        )

        file.write(
            f"Recall: {test_results['recall']:.4f}\n"
        )

        file.write(
            f"F2 Score: {test_results['f2']:.4f}\n\n"
        )

        file.write(
            "Confusion Matrix\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"TN: {cm[0, 0]}\n"
        )

        file.write(
            f"FP: {cm[0, 1]}\n"
        )

        file.write(
            f"FN: {cm[1, 0]}\n"
        )

        file.write(
            f"TP: {cm[1, 1]}\n"
        )

        file.write(
            "\nFeature Engineering Note\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            "The supplied credit-card dataset does not contain "
            "merchant ID, device fingerprint, or location fields. "
            "Therefore, transaction-history behavioral proxy "
            "features were used instead of fabricating unavailable "
            "attributes.\n"
        )

    print(
        f"\nPerformance report saved to: {report_path}"
    )

    print("\nGenerated files:")

    print("reports/roc_curve.png")
    print("reports/precision_recall_curve.png")
    print("reports/confusion_matrix.png")
    print("reports/performance_report.txt")

    print("\n" + "=" * 70)
    print("Ensemble evaluation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()