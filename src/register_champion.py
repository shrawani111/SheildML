import mlflow
import mlflow.sklearn
import joblib
import os

print("=" * 70)
print("ShieldML - Register Champion Model")
print("=" * 70)

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ShieldML-Fraud-Detection")

# Model path
model_path = "models/xgboost.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

print("\nLoading XGBoost model...")
model = joblib.load(model_path)
print("XGBoost model loaded.")

# Champion metrics
metrics = {
    "validation_roc_auc": 0.9899,
    "validation_pr_auc": 0.8581,
    "test_roc_auc": 0.9744,
    "test_pr_auc": 0.7645,
    "test_precision": 0.6000,
    "test_recall": 0.7500,
    "test_f2": 0.7143,
    "fraud_threshold": 0.1850,
    "api_p95_latency_ms": 41.26
}

with mlflow.start_run(run_name="ShieldML-XGBoost-Registration") as run:

    run_id = run.info.run_id

    # Parameters
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("model_role", "Champion")
    mlflow.log_param("ensemble_context", "RF + XGBoost + LightGBM")
    mlflow.log_param("threshold_selection", "Validation F2 optimization")

    # Metrics
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    # Artifacts
    artifact_files = [
        "reports/performance_report.txt",
        "reports/latency_benchmark.txt",
        "reports/roc_curve.png",
        "reports/precision_recall_curve.png",
        "reports/confusion_matrix.png",
        "reports/shap_summary.png",
        "reports/shap_bar.png",
    ]

    for file in artifact_files:
        if os.path.exists(file):
            mlflow.log_artifact(file)

    print("\nLogging XGBoost model to MLflow...")

    # XGBoost models contain trusted XGBoost classes.
    # MLflow/skops requires these types to be explicitly trusted.
    trusted_types = [
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier"
    ]

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="champion_model",
        skops_trusted_types=trusted_types
    )

    print("Model logged successfully.")
    print(f"Run ID: {run_id}")

    # Register model
    print("\nRegistering model...")

    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name="ShieldML-Champion"
    )

    print("\n" + "=" * 70)
    print("MODEL REGISTERED SUCCESSFULLY")
    print("=" * 70)
    print(f"Model name    : ShieldML-Champion")
    print(f"Model version : {registered_model.version}")
    print(f"Run ID        : {run_id}")
    print(f"Test ROC-AUC  : {metrics['test_roc_auc']}")
    print(f"Test PR-AUC   : {metrics['test_pr_auc']}")
    print(f"Test F2       : {metrics['test_f2']}")
    print(f"API p95       : {metrics['api_p95_latency_ms']} ms")

print("\nChampion registration completed.")