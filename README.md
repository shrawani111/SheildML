# ShieldML - Real-Time Financial Fraud Detection System

ShieldML is an end-to-end machine learning system for detecting potentially fraudulent financial transactions.

The project combines leakage-safe feature engineering, class-imbalance handling, multiple tree-based machine learning models, equal-weight ensemble evaluation, F2 threshold optimization, SHAP explainability, MLflow experiment tracking, model registration, and low-latency FastAPI serving.

---

## Project Objective

The objective of ShieldML is to develop a fraud detection pipeline capable of:

* Detecting fraudulent financial transactions
* Handling highly imbalanced transaction data
* Using historical transaction behavior without data leakage
* Optimizing the fraud decision threshold
* Explaining predictions using SHAP
* Tracking experiments using MLflow
* Registering the champion model
* Serving predictions through FastAPI
* Maintaining low inference latency

---

## System Architecture

```mermaid
flowchart TD

    A[Raw Transaction CSV] --> B[Data Cleaning]
    B --> C[Chronological Sorting]
    C --> D[Leakage-Safe Feature Engineering]

    D --> E[Time-Based Train / Validation / Test Split]

    E --> F[Imbalance Handling]

    F --> G[Random Forest]
    F --> H[XGBoost]
    F --> I[LightGBM]

    G --> J[Equal-Weight Soft Voting Ensemble]
    H --> J
    I --> J

    J --> K[F2 Threshold Optimization]
    K --> L[Test Evaluation]

    H --> M[SHAP TreeExplainer]
    M --> N[Global + Local Explanations]

    G --> O[MLflow]
    H --> O
    I --> O
    J --> O

    O --> P[Champion Model Registry]

    J --> Q[FastAPI]
    Q --> R[/health]
    Q --> S[/predict]
    Q --> T[/metrics]

    Q --> U[Latency Benchmark]
```

---

## Dataset

ShieldML uses a credit-card transaction dataset containing:

* `Time`
* `V1` to `V28`
* `Amount`
* `Class`

where:

* `Class = 0` â†’ legitimate transaction
* `Class = 1` â†’ fraudulent transaction

The dataset is highly imbalanced, making fraud detection an imbalanced classification problem.

### Dataset Limitation

The available dataset does not contain:

* Merchant ID
* Customer ID
* Device fingerprint
* Geographic location
* IP address
* Explicit transaction category

Therefore, ShieldML does not fabricate these unavailable attributes.

Instead, historical transaction behavior is represented through leakage-safe temporal and amount-based features.

---

## Data Preparation

The preprocessing pipeline:

1. Loads the raw transaction data.
2. Removes exact duplicate records.
3. Sorts transactions chronologically using `Time`.
4. Generates leakage-safe behavioral features.
5. Creates chronological train, validation, and test datasets.

### Dataset Split

| Dataset    | Samples | Fraud Rate |
| ---------- | ------: | ---------: |
| Training   | 198,608 |    0.1843% |
| Validation |  42,559 |    0.1292% |
| Test       |  42,559 |    0.1222% |

The chronological split helps prevent future information from influencing past transactions.

---

## Feature Engineering

ShieldML creates temporal and transaction-behavior features including:

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

Historical rolling features use shifted values so that the current transaction does not influence its own historical statistics.

This helps reduce target and temporal leakage.

---

## Imbalance Handling

Fraudulent transactions represent only a very small fraction of the dataset.

ShieldML uses different imbalance strategies for the trained models:

### Random Forest

Random Forest uses:

```text
class_weight = balanced
```

and SMOTE is applied to the training data.

### XGBoost

XGBoost uses class weighting through:

```text
scale_pos_weight
```

### LightGBM

LightGBM uses class weighting through:

```text
class_weight = balanced
```

Validation and test datasets are not SMOTE-resampled.

---

## Machine Learning Models

ShieldML trains three tree-based models:

### 1. Random Forest

Configuration includes:

* 200 estimators
* Maximum depth = 12
* `class_weight = balanced`
* `random_state = 42`

### 2. XGBoost

XGBoost is trained using class-weighted learning to address the severe class imbalance.

### 3. LightGBM

LightGBM is trained using balanced class weighting.

---

## Validation Performance

| Model               |    ROC-AUC |     PR-AUC |  Precision | Recall |         F1 |
| ------------------- | ---------: | ---------: | ---------: | -----: | ---------: |
| Logistic Regression |     0.9842 |     0.8421 |     0.0319 | 0.9273 |     0.0617 |
| Random Forest       |     0.9833 |     0.8512 |     0.7586 | 0.8000 |     0.7788 |
| XGBoost             | **0.9899** | **0.8581** |     0.8571 | 0.7636 |     0.8077 |
| LightGBM            |     0.9817 |     0.8565 | **0.9556** | 0.7818 | **0.8600** |

XGBoost achieved the strongest validation ROC-AUC and PR-AUC, while LightGBM achieved the strongest validation F1 score.

---

## Ensemble Model

The deployed fraud detector uses an equal-weight soft voting ensemble consisting of:

* Random Forest
* XGBoost
* LightGBM

The probability from each model is averaged:

```text
Ensemble Probability =
    (RF Probability
    + XGBoost Probability
    + LightGBM Probability) / 3
```

No probability calibration is claimed in the deployed implementation.

---

## F2 Threshold Optimization

The default classification threshold of `0.50` is not ideal for highly imbalanced fraud detection.

ShieldML searches validation probabilities for the threshold that maximizes the F2 score.

The selected threshold is:

```text
F2 Threshold = 0.1850
```

The threshold is selected using the validation set and then kept fixed for final test evaluation.

---

## Final Test Performance

The final equal-weight ensemble achieved:

| Metric             | Test Result |
| ------------------ | ----------: |
| ROC-AUC            |  **0.9744** |
| PR-AUC             |  **0.7645** |
| Precision          |  **0.6000** |
| Recall             |  **0.7500** |
| F2 Score           |  **0.7143** |
| Decision Threshold |  **0.1850** |

### Test Confusion Matrix

```text
                 Predicted
                 Legit   Fraud

Actual Legit     42481     26
Actual Fraud        13     39
```

The model detected 39 of the 52 fraud cases in the test set.

---

## SHAP Explainability

ShieldML uses:

```text
SHAP TreeExplainer
```

for model explainability.

SHAP analysis was performed using the XGBoost model.

### Top Global Features

| Rank | Feature |
| ---: | ------- |
|    1 | V4      |
|    2 | V14     |
|    3 | V12     |
|    4 | V11     |
|    5 | V10     |
|    6 | Time    |
|    7 | V1      |
|    8 | V8      |
|    9 | V3      |
|   10 | V19     |

The project generates:

* SHAP global summary plot
* SHAP feature-importance bar chart
* Three local SHAP waterfall examples
* SHAP feature-importance CSV
* SHAP analysis report

Generated files are stored in:

```text
reports/
```

---

## MLflow

MLflow is used for experiment tracking and model registration.

Tracked components include:

* Training parameters
* Validation metrics
* Model artifacts
* Performance metrics
* Experiment runs

The registered model is:

```text
ShieldML-Champion
```

The current registered champion version is:

```text
Version 1
```

The registered champion corresponds to the XGBoost model artifact used during model registration.

The deployed FastAPI service uses the independently evaluated RF + XGBoost + LightGBM ensemble.

---

## FastAPI Serving

ShieldML provides a FastAPI application for real-time inference.

Start the API with:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

The API loads all three trained models at startup.

### Available Endpoints

| Endpoint   | Method | Purpose                           |
| ---------- | ------ | --------------------------------- |
| `/`        | GET    | API information                   |
| `/health`  | GET    | Service health check              |
| `/predict` | POST   | Fraud prediction                  |
| `/metrics` | GET    | Test performance metrics          |
| `/docs`    | GET    | Interactive Swagger documentation |

---

## Prediction Response

The `/predict` endpoint returns:

```json
{
    "prediction": 0,
    "label": "LEGITIMATE",
    "fraud_probability": 0.003487,
    "threshold": 0.185,
    "model": "RF + XGBoost + LightGBM Ensemble",
    "inference_time_ms": 38.0
}
```

For a fraud prediction:

```text
prediction = 1
label = FRAUD
```

---

## API Latency

A warm 100-request benchmark was performed against the local FastAPI `/predict` endpoint.

### Final Benchmark

| Metric      |         Result |
| ----------- | -------------: |
| Requests    |            100 |
| Average     |   **38.39 ms** |
| P95         |   **41.26 ms** |
| Minimum     |       35.36 ms |
| Maximum     |       67.05 ms |
| Requirement | **<50 ms P95** |
| Status      |     **PASSED** |

### Serving Optimization

The trained Random Forest originally used:

```text
n_jobs = -1
```

For single-transaction API inference, this caused substantial CPU thread-management overhead.

The API serving configuration uses:

```text
n_jobs = 1
```

for the Random Forest.

The model was not retrained, and the 200 trained trees were retained.

### Before Optimization

```text
Average: 75.07 ms
P95:     110.65 ms
```

### After Optimization

```text
Average: 38.39 ms
P95:      41.26 ms
```

Therefore, the ShieldML `/predict` endpoint satisfies the required sub-50 ms P95 latency target under the tested local benchmark.

The complete benchmark is documented in:

```text
reports/latency_benchmark.txt
```

---

## Project Structure

```text
ShieldML/
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â””â”€â”€ processed/
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ random_forest.joblib
â”‚   â”œâ”€â”€ xgboost.joblib
â”‚   â””â”€â”€ lightgbm.joblib
â”‚
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ confusion_matrix.png
â”‚   â”œâ”€â”€ precision_recall_curve.png
â”‚   â”œâ”€â”€ roc_curve.png
â”‚   â”œâ”€â”€ shap_summary.png
â”‚   â”œâ”€â”€ shap_bar.png
â”‚   â”œâ”€â”€ shap_waterfall_1.png
â”‚   â”œâ”€â”€ shap_waterfall_2.png
â”‚   â”œâ”€â”€ shap_waterfall_3.png
â”‚   â”œâ”€â”€ shap_feature_importance.csv
â”‚   â”œâ”€â”€ shap_report.txt
â”‚   â”œâ”€â”€ performance_report.txt
â”‚   â””â”€â”€ latency_benchmark.txt
â”‚
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ model_card.md
â”‚   â””â”€â”€ architecture.md
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_preparation.py
â”‚   â”œâ”€â”€ train_models.py
â”‚   â”œâ”€â”€ ensemble_threshold.py
â”‚   â”œâ”€â”€ shap_analysis.py
â”‚   â”œâ”€â”€ register_champion.py
â”‚   â””â”€â”€ api.py
â”‚
â”œâ”€â”€ .gitignore
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ mlflow.db
â””â”€â”€ README.md
```

---

## Installation

Create and activate the virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Training Pipeline

### 1. Prepare the data

```powershell
python src\data_preparation.py
```

### 2. Train the models

```powershell
python src\train_models.py
```

### 3. Generate ensemble results and optimize threshold

```powershell
python src\ensemble_threshold.py
```

### 4. Generate SHAP explanations

```powershell
python src\shap_analysis.py
```

### 5. Register the champion model

```powershell
python src\register_champion.py
```

### 6. Start the API

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

---

## MLflow Dashboard

Start MLflow with:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the displayed MLflow URL in a browser.

The MLflow experiment is:

```text
ShieldML-Fraud-Detection
```

The registered model is:

```text
ShieldML-Champion
```

---

## Generated Reports

The project generates:

* ROC curve
* Precision-Recall curve
* Confusion matrix
* Performance report
* SHAP summary plot
* SHAP feature-importance plot
* Three SHAP waterfall plots
* SHAP feature-importance CSV
* SHAP report
* FastAPI latency benchmark

---

## Limitations

### Dataset Features

The dataset does not contain merchant, customer, device, IP, or geographic information.

Therefore, the system cannot directly model:

* Merchant-level fraud patterns
* Device-level behavioral patterns
* Geographic anomalies
* Customer identity behavior

Historical transaction features are used as behavioral proxies where possible.

### Model Limitations

Fraud patterns can change over time.

Potential failure modes include:

* Previously unseen fraud patterns
* Concept drift
* False positives on unusual legitimate transactions
* False negatives on sophisticated fraud
* Reduced performance on substantially different datasets

The selected threshold is optimized for the available validation data and may require recalibration when the production data distribution changes.

### Latency

The reported latency was measured on the local development environment using a warm FastAPI service and 100 sequential requests. Production cloud latency may differ depending on hardware, network conditions, and deployment configuration.

---

## Ethical Considerations

Fraud detection systems should be used as decision-support systems rather than unquestionable automated decision makers.

The model should be monitored for:

* False positives
* False negatives
* Data drift
* Performance degradation
* Changes in fraud patterns

Human review may be appropriate for high-impact financial decisions.

---

## Key Results

ShieldML successfully demonstrates:

* Leakage-safe temporal feature engineering
* Chronological train/validation/test splitting
* Severe class-imbalance handling
* Random Forest, XGBoost, and LightGBM models
* Equal-weight soft voting ensemble
* Validation-based F2 threshold optimization
* Test ROC-AUC above 0.90
* SHAP TreeExplainer interpretability
* MLflow experiment tracking
* Champion model registration
* FastAPI real-time inference
* **Average API latency of 38.39 ms**
* **P95 API latency of 41.26 ms**
* **Sub-50 ms P95 serving requirement achieved**

---

## Project Status

**ShieldML â€” Implementation Complete**

Core machine learning, ensemble evaluation, explainability, MLflow tracking, model registration, FastAPI serving, and latency benchmarking have been implemented.

Final verified ensemble test ROC-AUC:

```text
0.9744
```

Final verified F2 score:

```text
0.7143
```

Final verified API P95 latency:

```text
41.26 ms
```
