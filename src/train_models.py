import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MLFLOW_URI = "sqlite:///mlflow.db"

TARGET = "Class"


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)

    # Remove target and raw Hour helper column
    drop_columns = [TARGET, "Hour"]

    X_train = train.drop(columns=drop_columns)
    y_train = train[TARGET]

    X_val = val.drop(columns=drop_columns)
    y_val = val[TARGET]

    return X_train, y_train, X_val, y_val


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(name, model, X_val, y_val):

    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(
        y_val,
        probabilities
    )

    ap = average_precision_score(
        y_val,
        probabilities
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    print(f"\n{name}")
    print("-" * 50)

    print(f"ROC-AUC : {auc:.4f}")
    print(f"PR-AUC  : {ap:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1      : {f1:.4f}")

    return {
        "roc_auc": auc,
        "pr_auc": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("ShieldML - Model Training")
    print("=" * 70)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    X_train, y_train, X_val, y_val = load_data()

    print("\nOriginal training distribution:")
    print(y_train.value_counts())

    # -----------------------------------------------------
    # SMOTE
    # -----------------------------------------------------

    print("\nApplying SMOTE to training data only...")

    smote = SMOTE(
        sampling_strategy=0.2,
        random_state=42
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

    # -----------------------------------------------------
    # MLflow
    # -----------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_URI
    )

    mlflow.set_experiment(
        "ShieldML-Fraud-Detection"
    )

    results = {}

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    with mlflow.start_run(
        run_name="LogisticRegression_Baseline"
    ):

        print("\nTraining Logistic Regression...")

        lr = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ])

        lr.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            "Logistic Regression",
            lr,
            X_val,
            y_val
        )

        mlflow.log_params({
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000
        })

        mlflow.log_metrics(
            metrics
        )

        results["LogisticRegression"] = metrics

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    with mlflow.start_run(
        run_name="RandomForest"
    ):

        print("\nTraining Random Forest...")

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        rf.fit(
            X_train_smote,
            y_train_smote
        )

        metrics = evaluate_model(
            "Random Forest",
            rf,
            X_val,
            y_val
        )

        mlflow.log_params({
            "model": "RandomForest",
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "training_method": "SMOTE"
        })

        mlflow.log_metrics(
            metrics
        )

        # MLflow trusted type for Random Forest
        mlflow.sklearn.log_model(
            rf,
            name="random_forest",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        joblib.dump(
            rf,
            MODEL_DIR / "random_forest.joblib"
        )

        results["RandomForest"] = metrics

    # =====================================================
    # XGBOOST
    # =====================================================

    with mlflow.start_run(
        run_name="XGBoost"
    ):

        print("\nTraining XGBoost...")

        scale_pos_weight = (
            (y_train == 0).sum()
            /
            (y_train == 1).sum()
        )

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1
        )

        xgb.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            "XGBoost",
            xgb,
            X_val,
            y_val
        )

        mlflow.log_params({
            "model": "XGBoost",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight
        })

        mlflow.log_metrics(
            metrics
        )

        # MLflow trusted types for XGBoost
        mlflow.sklearn.log_model(
            xgb,
            name="xgboost",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )

        joblib.dump(
            xgb,
            MODEL_DIR / "xgboost.joblib"
        )

        results["XGBoost"] = metrics

    # =====================================================
    # LIGHTGBM
    # =====================================================

    with mlflow.start_run(
        run_name="LightGBM"
    ):

        print("\nTraining LightGBM...")

        lgbm = LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        lgbm.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            "LightGBM",
            lgbm,
            X_val,
            y_val
        )

        mlflow.log_params({
            "model": "LightGBM",
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "class_weight": "balanced"
        })

        mlflow.log_metrics(
            metrics
        )

        # MLflow trusted types for LightGBM
        mlflow.sklearn.log_model(
            lgbm,
            name="lightgbm",
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier"
            ]
        )

        joblib.dump(
            lgbm,
            MODEL_DIR / "lightgbm.joblib"
        )

        results["LightGBM"] = metrics

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for name, metrics in results.items():

        print(
            f"{name:20s} "
            f"AUC={metrics['roc_auc']:.4f} "
            f"PR-AUC={metrics['pr_auc']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

    # -----------------------------------------------------
    # Best models by different metrics
    # -----------------------------------------------------

    best_auc = max(
        results,
        key=lambda x: results[x]["roc_auc"]
    )

    best_pr_auc = max(
        results,
        key=lambda x: results[x]["pr_auc"]
    )

    best_f1 = max(
        results,
        key=lambda x: results[x]["f1"]
    )

    print("\n" + "=" * 70)
    print("BEST VALIDATION MODELS")
    print("=" * 70)

    print(
        f"Best ROC-AUC : {best_auc} "
        f"({results[best_auc]['roc_auc']:.4f})"
    )

    print(
        f"Best PR-AUC  : {best_pr_auc} "
        f"({results[best_pr_auc]['pr_auc']:.4f})"
    )

    print(
        f"Best F1      : {best_f1} "
        f"({results[best_f1]['f1']:.4f})"
    )

    print("\nTraining completed successfully.")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()