# ShieldML — Model Card

## 1. Model Overview

**Project:** ShieldML — Real-Time Financial Fraud Detection System

**Deployed Model:** RF + XGBoost + LightGBM Equal-Weight Soft-Voting Ensemble

**Registered Champion Model:** XGBoost

**Registered Model:** `ShieldML-Champion`

**Model Version:** 1

**MLflow Run ID:** `1b27f6023fc849bdb89359fc407c1dbc`

ShieldML is a machine learning-based financial fraud detection system designed to identify potentially fraudulent credit card transactions using transaction-level numerical and behavior-derived features.

The system combines data preprocessing, leakage-safe feature engineering, class-imbalance handling, multiple machine learning models, ensemble prediction, threshold optimization, explainability, MLflow tracking, and FastAPI serving.

The deployed API uses an equal-weight soft-voting ensemble of Random Forest, XGBoost, and LightGBM models.

---

## 2. Intended Use

ShieldML is intended to support real-time financial transaction fraud screening.

The system produces:

* Fraud probability
* Fraud or legitimate prediction
* Fraud decision threshold
* Model and prediction information
* Inference latency

SHAP-based explanations are generated separately through the model explainability pipeline.

The system is designed as a decision-support component rather than an autonomous replacement for human fraud investigation.

---

## 3. Dataset

The project uses a credit card transaction dataset containing:

* `Time`
* `V1` to `V28`
* `Amount`
* `Class`

The target variable is:

* `Class = 0` → Legitimate transaction
* `Class = 1` → Fraudulent transaction

The original dataset contains approximately 284,807 transactions.

After duplicate removal, the data was sorted chronologically and divided into:

* Training: 198,608 samples
* Validation: 42,559 samples
* Test: 42,559 samples

Fraud rates:

* Training: approximately 0.1843%
* Validation: approximately 0.1292%
* Test: approximately 0.1222%

The chronological split was used to better represent a future-transaction prediction scenario and reduce the risk of information from future transactions influencing earlier predictions.

---

## 4. Data Processing

The ShieldML data processing pipeline performs:

1. Loading and schema processing of the transaction dataset.
2. Removal of exact duplicate transactions.
3. Chronological sorting using the `Time` feature.
4. Time-based feature generation.
5. Transaction amount transformation.
6. Historical transaction feature generation.
7. Chronological train/validation/test splitting.

The split is chronological rather than randomly shuffled to better represent a real-world future-transaction prediction scenario.

Historical features are calculated using information from previous transactions only.

---

## 5. Feature Engineering

ShieldML uses the original transaction variables together with leakage-safe derived features.

### Original Features

* `Time`
* `V1` to `V28`
* `Amount`

### Derived Features

* `Hour`
* `Hour_Sin`
* `Hour_Cos`
* `Day`
* `Amount_Log`
* `Amount_Rolling_Mean_10`
* `Amount_Rolling_Std_10`
* `Amount_Deviation`
* `Time_Since_Previous`
* `Previous_1H_Transaction_Count`

The model-serving feature set contains the engineered numerical features used during training.

Historical rolling features are calculated using previous transactions only. The current transaction is excluded using a shift operation to reduce the risk of target or data leakage.

Examples include:

* Previous transaction timing
* Previous transaction count within one hour
* Rolling transaction amount statistics
* Deviation from recent transaction amount behavior

---

## 6. Important Dataset Limitation

The available dataset does not contain explicit:

* Merchant ID
* Device fingerprint
* Geographic location
* Customer/account identifier
* IP address

Therefore, ShieldML does not fabricate these fields.

Instead, transaction-history proxy features such as rolling amount statistics, time since the previous transaction, and recent transaction counts are used.

This means the current system demonstrates fraud detection using the available transaction-level dataset rather than a complete production banking fraud-data environment.

For production deployment, additional customer, merchant, device, geographic, and transaction-context signals could improve behavioral fraud detection.

---

## 7. Class Imbalance Handling

Fraudulent transactions represent a very small percentage of the dataset.

ShieldML uses different imbalance strategies for the evaluated models:

### Random Forest

SMOTE is applied to the training data to increase representation of the minority class.

### XGBoost

Cost-sensitive learning is used through class weighting.

### LightGBM

Class weighting is used to account for the imbalanced fraud class.

SMOTE is applied only to the training data.

Validation and test datasets remain untouched so that evaluation reflects the original class distribution.

No validation or test samples are used for SMOTE.

---

## 8. Models Evaluated

Three tree-based models were trained and evaluated.

### Random Forest

Validation performance:

* ROC-AUC: 0.9833
* PR-AUC: 0.8512
* Precision: 0.7586
* Recall: 0.8000
* F1: 0.7788

### XGBoost

Validation performance:

* ROC-AUC: 0.9899
* PR-AUC: 0.8581
* Precision: 0.8571
* Recall: 0.7636
* F1: 0.8077

### LightGBM

Validation performance:

* ROC-AUC: 0.9817
* PR-AUC: 0.8565
* Precision: 0.9556
* Recall: 0.7818
* F1: 0.8600

An equal-weight soft-voting ensemble was then created using the three trained models.

---

## 9. Champion Model and Deployed Model

The registered MLflow champion is the XGBoost model.

XGBoost was selected as the registered champion because it achieved the strongest individual validation ROC-AUC and PR-AUC among the evaluated base models.

### Registered Champion

* Registered Model: `ShieldML-Champion`
* Version: 1
* Model: XGBoost
* MLflow Run ID: `1b27f6023fc849bdb89359fc407c1dbc`

### Deployed FastAPI Model

The current FastAPI `/predict` endpoint does **not** use the registered XGBoost artifact alone.

Instead, the deployed API uses:

* Random Forest
* XGBoost
* LightGBM

with equal weights.

The ensemble probability is calculated as:

```text
Ensemble Probability =
(Random Forest Probability
+ XGBoost Probability
+ LightGBM Probability) / 3
```

The ensemble was selected for API serving because it combines predictions from the three evaluated tree-based models and was directly evaluated on the held-out test set.

The distinction between the registered MLflow champion and the deployed ensemble is intentionally documented for transparency.

---

## 10. Ensemble Performance

The RF + XGBoost + LightGBM equal-weight soft-voting ensemble achieved the following results.

### Validation Performance

* ROC-AUC: 0.9836
* PR-AUC: 0.8621

### Test Performance

* ROC-AUC: 0.9744
* PR-AUC: 0.7645
* Precision: 0.6000
* Recall: 0.7500
* F2 Score: 0.7143

The test ROC-AUC is above the project target of 0.90.

### Test Confusion Matrix

```text
                 Predicted
                 Legit   Fraud

Actual Legit     42481     26
Actual Fraud        13     39
```

This means:

* True Negatives: 42,481
* False Positives: 26
* False Negatives: 13
* True Positives: 39

The model detected 39 of the 52 fraud cases in the held-out test set.

---

## 11. Threshold Optimization

The default classification threshold of 0.50 was not used as the final fraud decision threshold.

Because fraud detection is highly imbalanced and missing fraudulent transactions can be costly, the F2 score was used for threshold optimization.

The threshold was selected using the validation dataset.

### Selected Threshold

```text
F2-Optimal Threshold = 0.1850
```

### Validation Result at Selected Threshold

* F2 Score: 0.8451

The selected threshold was then kept fixed and applied to the unseen test set.

### Test Result

* Precision: 0.6000
* Recall: 0.7500
* F2 Score: 0.7143

The test set was not used to select the threshold.

---

## 12. Explainability

ShieldML includes SHAP-based model explainability.

The XGBoost model is analyzed using:

```text
shap.TreeExplainer
```

A sample of 1,000 test transactions was used for the SHAP analysis.

The SHAP analysis generated:

* Global SHAP summary plot
* Global SHAP feature-importance bar chart
* Three local SHAP waterfall examples
* SHAP feature-importance CSV report
* SHAP analysis report

### Top Features by Mean Absolute SHAP Value

The most influential features identified in the SHAP analysis included:

1. `V4`
2. `V14`
3. `V12`
4. `V11`
5. `V10`
6. `Time`
7. `V1`
8. `V8`
9. `V3`
10. `V19`

SHAP explanations are used to understand model behavior and identify which input features contribute most strongly to individual predictions.

The SHAP analysis is performed on the XGBoost model rather than the complete three-model ensemble.

---

## 13. FastAPI Serving

ShieldML provides a FastAPI service for real-time prediction.

### Available Endpoints

#### `GET /`

Provides basic API information.

#### `GET /health`

Returns the health status, deployed model information, and fraud threshold.

#### `POST /predict`

Accepts transaction features and returns:

* Prediction
* Fraud label
* Fraud probability
* Fraud threshold
* Model information
* Inference time

#### `GET /metrics`

Returns the evaluated test performance of the deployed ensemble.

The models are loaded once when the FastAPI application starts rather than being loaded for every request.

This reduces unnecessary model-loading overhead during inference.

---

## 14. Serving Optimization

The Random Forest model was originally configured with multiple processing threads.

During API latency benchmarking, excessive thread-management overhead was observed for individual local requests.

For serving, the Random Forest model is configured with:

```text
n_jobs = 1
```

This is a serving-time optimization only.

It does not retrain the model and does not change the learned decision trees.

XGBoost and LightGBM are also loaded once at application startup.

The three model predictions are executed concurrently using a thread pool, after which their probabilities are averaged.

---

## 15. API Latency Benchmark

The final latency benchmark was performed against the local FastAPI `/predict` endpoint.

Benchmark configuration:

* Endpoint: `POST /predict`
* Requests: 100
* Warm-up: 1 request
* Model: RF + XGBoost + LightGBM ensemble
* Environment: Local FastAPI/Uvicorn server
* Random Forest serving threads: 1

### Final Results

| Metric          |   Result |
| --------------- | -------: |
| Average latency | 38.39 ms |
| P95 latency     | 41.26 ms |
| Minimum latency | 35.36 ms |
| Maximum latency | 67.05 ms |
| Target P95      |  < 50 ms |
| Status          |   PASSED |

The final average latency and P95 latency are both below the project's 50 ms serving target.

The maximum individual request was higher than 50 ms, but the P95 latency remained below the target.

---

## 16. MLflow Tracking

MLflow is used to track model training and experiment information.

The project uses the MLflow experiment:

```text
ShieldML-Fraud-Detection
```

Tracked information includes model performance and training-related parameters.

The registered model is:

```text
ShieldML-Champion
```

Current registered version:

```text
Version 1
```

The registered artifact is the XGBoost champion model.

The deployed FastAPI ensemble is evaluated separately using the held-out test set.

This distinction is maintained to avoid incorrectly representing the registered XGBoost artifact as the deployed three-model ensemble.

---

## 17. Limitations and Failure Modes

ShieldML has several limitations that should be considered before production deployment.

### Dataset Limitations

The dataset does not contain:

* Merchant information
* Customer identity
* Device information
* Geographic information
* IP address
* Authentication information

Therefore, the model cannot directly learn merchant-specific, customer-specific, device-specific, or location-specific fraud patterns.

### Historical Proxy Limitations

The engineered historical features are based on the available transaction sequence.

They should not be interpreted as complete customer behavioral profiles.

### False Positives

The model can classify legitimate transactions as fraudulent.

This can cause unnecessary transaction reviews or customer friction.

### False Negatives

The model can fail to detect fraudulent transactions.

Fraud patterns may change over time, and previously unseen fraud behavior may not be represented in the training data.

### Distribution Shift

Fraud patterns can change because of:

* New attack strategies
* Changes in payment behavior
* New transaction channels
* Changes in customer behavior
* Changes in fraud prevention systems

Therefore, model performance may degrade after deployment.

### Threshold Sensitivity

The threshold of 0.1850 was optimized for the validation dataset.

A different operational cost structure may require a different threshold.

### Local Latency Measurement

The reported latency was measured on a local FastAPI/Uvicorn environment.

Actual production latency can differ depending on:

* Hardware
* Network conditions
* Deployment platform
* Concurrent request volume
* Infrastructure configuration

---

## 18. Ethical Considerations

Fraud detection systems can affect legitimate users when transactions are incorrectly flagged.

ShieldML should therefore be used as a decision-support system rather than as an unquestioned automated decision maker.

Production deployment should include:

* Human review for high-impact decisions
* Monitoring of false-positive rates
* Monitoring of false-negative rates
* Periodic model evaluation
* Monitoring for data drift
* Secure handling of transaction data
* Appropriate access controls
* Protection of customer privacy

The model should not be used to make unrelated decisions about individuals.

---

## 19. Reproducibility

The project contains a reproducible machine learning pipeline covering:

```text
Raw Dataset
    ↓
Data Preparation
    ↓
Feature Engineering
    ↓
Chronological Split
    ↓
Imbalance Handling
    ↓
Model Training
    ↓
RF + XGBoost + LightGBM
    ↓
Soft-Voting Ensemble
    ↓
F2 Threshold Optimization
    ↓
Test Evaluation
    ↓
SHAP Explainability
    ↓
FastAPI Serving
```

The project stores trained model artifacts, reports, plots, and configuration files in the project structure.

The main trained model artifacts are:

```text
models/random_forest.joblib
models/xgboost.joblib
models/lightgbm.joblib
```

The main reports include:

```text
reports/performance_report.txt
reports/latency_benchmark.txt
reports/shap_report.txt
reports/shap_feature_importance.csv
```

---

## 20. Final Status

ShieldML currently provides:

* Leakage-safe chronological data processing
* Behavioral transaction feature engineering
* Class-imbalance handling
* Random Forest model
* XGBoost model
* LightGBM model
* Equal-weight soft-voting ensemble
* F2-based threshold optimization
* Test ROC-AUC above 0.90
* SHAP TreeExplainer analysis
* Global and local SHAP visualizations
* MLflow experiment tracking
* Registered XGBoost champion model
* FastAPI real-time prediction service
* Health and metrics endpoints
* Sub-50 ms average and P95 API latency
* Performance and latency reports
* Documented limitations and failure modes

### Final Key Metrics

| Metric              |      Result |
| ------------------- | ----------: |
| Test ROC-AUC        |      0.9744 |
| Test PR-AUC         |      0.7645 |
| Test Precision      |      0.6000 |
| Test Recall         |      0.7500 |
| Test F2             |      0.7143 |
| Fraud Threshold     |      0.1850 |
| API Average Latency |    38.39 ms |
| API P95 Latency     |    41.26 ms |
| API Target          | < 50 ms P95 |
| Target Status       |      PASSED |

**Model Card Status:** Complete
