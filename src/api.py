# src/api.py

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path("models")

FRAUD_THRESHOLD = 0.1850

FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Hour_Sin",
    "Hour_Cos",
    "Day",
    "Amount_Log",
    "Amount_Rolling_Mean_10",
    "Amount_Rolling_Std_10",
    "Amount_Deviation",
    "Time_Since_Previous",
    "Previous_1H_Transaction_Count",
]


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="ShieldML Fraud Detection API",
    description=(
        "Real-time financial fraud detection using an "
        "equal-weight RF + XGBoost + LightGBM ensemble."
    ),
    version="1.2.0",
)


# ============================================================
# Load Models at Startup
# ============================================================

print("Loading ShieldML ensemble models...")

rf_model = joblib.load(MODEL_DIR / "random_forest.joblib")
xgb_model = joblib.load(MODEL_DIR / "xgboost.joblib")
lgbm_model = joblib.load(MODEL_DIR / "lightgbm.joblib")

# ------------------------------------------------------------
# Serving optimization
# ------------------------------------------------------------
# The trained Random Forest uses 200 trees and was originally
# configured with n_jobs=-1.
#
# For single-transaction API inference, using all CPU threads
# introduces significant thread-management overhead.
#
# Setting n_jobs=1 does NOT retrain or change the trees.
# It only changes how the existing model performs prediction.
# ------------------------------------------------------------

rf_model.set_params(n_jobs=1)

print("Random Forest loaded successfully.")
print("Random Forest serving threads: 1")

print("XGBoost loaded successfully.")
print("LightGBM loaded successfully.")
print("ShieldML ensemble loaded successfully.")


# ============================================================
# Thread Pool
# ============================================================

# Run the three model predictions concurrently.
executor = ThreadPoolExecutor(max_workers=3)


# ============================================================
# Request Schema
# ============================================================

class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float

    Hour_Sin: float
    Hour_Cos: float
    Day: float

    Amount_Log: float
    Amount_Rolling_Mean_10: float
    Amount_Rolling_Std_10: float
    Amount_Deviation: float
    Time_Since_Previous: float
    Previous_1H_Transaction_Count: float


# ============================================================
# Ensemble Prediction
# ============================================================

def ensemble_predict(dataframe: pd.DataFrame) -> float:
    """
    Generate fraud probability using the equal-weight
    RF + XGBoost + LightGBM ensemble.
    """

    # Submit all three predictions concurrently.
    rf_future = executor.submit(
        rf_model.predict_proba,
        dataframe
    )

    xgb_future = executor.submit(
        xgb_model.predict_proba,
        dataframe
    )

    lgbm_future = executor.submit(
        lgbm_model.predict_proba,
        dataframe
    )

    # Retrieve probabilities.
    rf_probability = rf_future.result()[0][1]
    xgb_probability = xgb_future.result()[0][1]
    lgbm_probability = lgbm_future.result()[0][1]

    # Equal-weight soft voting.
    ensemble_probability = (
        rf_probability
        + xgb_probability
        + lgbm_probability
    ) / 3.0

    return float(ensemble_probability)


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "model": "RF + XGBoost + LightGBM Ensemble",
        "threshold": FRAUD_THRESHOLD,
        "rf_serving_threads": 1,
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):
    """
    Predict whether a financial transaction is fraudulent.
    """

    start_time = time.perf_counter()

    # Convert request to dictionary.
    transaction_data = transaction.model_dump()

    # Ensure exact feature order.
    input_data = pd.DataFrame(
        [transaction_data],
        columns=FEATURE_COLUMNS
    )

    # Ensemble prediction.
    fraud_probability = ensemble_predict(input_data)

    # Apply validation-selected F2 threshold.
    prediction = int(
        fraud_probability >= FRAUD_THRESHOLD
    )

    # Human-readable label.
    label = (
        "FRAUD"
        if prediction == 1
        else "LEGITIMATE"
    )

    # Measure complete inference time.
    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "prediction": prediction,
        "label": label,
        "fraud_probability": round(
            fraud_probability,
            6
        ),
        "threshold": FRAUD_THRESHOLD,
        "model": "RF + XGBoost + LightGBM Ensemble",
        "inference_time_ms": round(
            inference_time_ms,
            3
        ),
    }


# ============================================================
# Metrics Endpoint
# ============================================================

@app.get("/metrics")
def metrics():
    """
    Return the evaluated test-set performance of the
    deployed ensemble.
    """

    return {
        "model": "RF + XGBoost + LightGBM Ensemble",
        "threshold": FRAUD_THRESHOLD,

        "test_roc_auc": 0.9744,
        "test_pr_auc": 0.7645,
        "test_precision": 0.6000,
        "test_recall": 0.7500,
        "test_f2": 0.7143,

        "status": "model_metrics_available",
    }


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "name": "ShieldML Fraud Detection API",
        "version": "1.2.0",
        "model": "RF + XGBoost + LightGBM Ensemble",
        "threshold": FRAUD_THRESHOLD,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "prediction_endpoint": "/predict",
    }