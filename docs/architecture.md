\# ShieldML Architecture



```mermaid

flowchart TD



&#x20;   A\[Raw Credit Card Transaction CSV] --> B\[Data Cleaning]



&#x20;   B --> C\[Chronological Sorting]



&#x20;   C --> D\[Leakage-Safe Feature Engineering]



&#x20;   D --> D1\[Time Features]

&#x20;   D --> D2\[Amount Features]

&#x20;   D --> D3\[Transaction History Features]



&#x20;   D1 --> E\[Time-Based Train / Validation / Test Split]

&#x20;   D2 --> E

&#x20;   D3 --> E



&#x20;   E --> F\[Training Data]



&#x20;   F --> G\[Imbalance Handling]



&#x20;   G --> G1\[SMOTE for Random Forest]

&#x20;   G --> G2\[Class Weighting for XGBoost]

&#x20;   G --> G3\[Class Weighting for LightGBM]



&#x20;   G1 --> H\[Random Forest]

&#x20;   G2 --> I\[XGBoost]

&#x20;   G3 --> J\[LightGBM]



&#x20;   H --> K\[Soft Voting Ensemble]

&#x20;   I --> K

&#x20;   J --> K



&#x20;   K --> L\[Validation F2 Threshold Optimization]



&#x20;   L --> M\[Final Fraud Threshold]



&#x20;   M --> N\[Test Evaluation]



&#x20;   I --> O\[SHAP TreeExplainer]

&#x20;   O --> P\[Global SHAP Analysis]

&#x20;   O --> Q\[Local SHAP Waterfall Explanations]



&#x20;   I --> R\[MLflow Tracking]

&#x20;   H --> R

&#x20;   J --> R

&#x20;   K --> R



&#x20;   R --> S\[Registered Champion Model]



&#x20;   S --> T\[FastAPI]



&#x20;   T --> U\[/health]

&#x20;   T --> V\[/predict]

&#x20;   T --> W\[/metrics]



&#x20;   V --> X\[Fraud Probability]

&#x20;   V --> Y\[Fraud / Legitimate Decision]



&#x20;   T --> Z\[Latency Benchmark]

