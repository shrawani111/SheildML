import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# ShieldML - SHAP Explainability
# ============================================================

MODEL_PATH = Path("models/xgboost.joblib")
TEST_PATH = Path("data/processed/test.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

TARGET = "Class"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ShieldML - SHAP Explainability Analysis")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading XGBoost model...")

    model = joblib.load(MODEL_PATH)

    print("XGBoost model loaded successfully.")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test data...")

    test = pd.read_csv(TEST_PATH)

    X_test = test.drop(columns=[TARGET, "Hour"])
    y_test = test[TARGET]

    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_test.shape[1]}")

    # --------------------------------------------------------
    # Select representative SHAP sample
    # --------------------------------------------------------

    # Use a fixed random state for reproducibility.
    # 1,000 observations keeps SHAP computation manageable.

    sample_size = min(1000, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    print(
        f"\nSHAP sample size: {len(X_sample):,}"
    )

    # --------------------------------------------------------
    # Create TreeExplainer
    # --------------------------------------------------------

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    print("TreeExplainer created successfully.")

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    print("\nCalculating SHAP values...")

    shap_values = explainer.shap_values(X_sample)

    # XGBoost binary classification normally returns
    # a 2-dimensional array.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    print(
        f"SHAP values shape: {shap_values.shape}"
    )

    # --------------------------------------------------------
    # Global Feature Importance - Beeswarm
    # --------------------------------------------------------

    print("\nGenerating global SHAP summary plot...")

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=15
    )

    plt.title(
        "ShieldML - SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "shap_summary.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved: reports/shap_summary.png"
    )

    # --------------------------------------------------------
    # Global Feature Importance - Bar
    # --------------------------------------------------------

    print("\nGenerating SHAP bar plot...")

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=15
    )

    plt.title(
        "ShieldML - Mean Absolute SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "shap_bar.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved: reports/shap_bar.png"
    )

    # --------------------------------------------------------
    # Find three fraud examples
    # --------------------------------------------------------

    print("\nSelecting local fraud examples...")

    fraud_indices = test[
        test[TARGET] == 1
    ].index

    # Use the first three fraud records that are
    # available in the test set.

    selected_indices = list(
        fraud_indices[:3]
    )

    # Make sure the selected records exist
    # inside our SHAP sample.

    selected_rows = X_sample[
        X_sample.index.isin(selected_indices)
    ]

    # If sampling did not include the first three frauds,
    # use three fraud observations directly.

    if len(selected_rows) < 3:

        fraud_test = test[
            test[TARGET] == 1
        ].head(3)

        selected_rows = fraud_test.drop(
            columns=[TARGET, "Hour"]
        )

    # --------------------------------------------------------
    # Generate local waterfall plots
    # --------------------------------------------------------

    print(
        f"Generating {len(selected_rows)} local explanations..."
    )

    for i, (index, row) in enumerate(
        selected_rows.iterrows(),
        start=1
    ):

        row_df = row.to_frame().T

        row_shap = explainer.shap_values(
            row_df
        )

        if isinstance(row_shap, list):
            row_shap = row_shap[1]

        row_shap = np.asarray(
            row_shap
        ).flatten()

        # Obtain the model's expected/base value.
        expected_value = explainer.expected_value

        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[-1]

        expected_value = float(
            np.asarray(expected_value).flatten()[0]
        )

        # SHAP waterfall explanation
        explanation = shap.Explanation(
            values=row_shap,
            base_values=expected_value,
            data=row.values,
            feature_names=X_sample.columns.tolist()
        )

        plt.figure(figsize=(10, 8))

        shap.plots.waterfall(
            explanation,
            max_display=12,
            show=False
        )

        plt.title(
            f"ShieldML - Local Fraud Explanation {i}"
        )

        plt.tight_layout()

        output_path = (
            REPORT_DIR /
            f"shap_waterfall_{i}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Save numerical feature importance
    # --------------------------------------------------------

    print(
        "\nCalculating numerical SHAP importance..."
    )

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Absolute_SHAP": mean_abs_shap
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance_path = (
        REPORT_DIR /
        "shap_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    # --------------------------------------------------------
    # Print top features
    # --------------------------------------------------------

    print("\nTop SHAP Features:")
    print("-" * 60)

    for _, row in importance.head(15).iterrows():

        print(
            f"{row['Feature']:35s} "
            f"{row['Mean_Absolute_SHAP']:.6f}"
        )

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    report_path = (
        REPORT_DIR /
        "shap_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ShieldML - SHAP Explainability Report\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            "Model: XGBoost\n"
        )

        file.write(
            "Explainer: SHAP TreeExplainer\n"
        )

        file.write(
            f"SHAP sample size: {len(X_sample):,}\n\n"
        )

        file.write(
            "Top Features by Mean Absolute SHAP Value\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for _, row in importance.head(15).iterrows():

            file.write(
                f"{row['Feature']:35s} "
                f"{row['Mean_Absolute_SHAP']:.6f}\n"
            )

        file.write(
            "\nGenerated SHAP Artifacts\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            "shap_summary.png\n"
        )

        file.write(
            "shap_bar.png\n"
        )

        file.write(
            "shap_waterfall_1.png\n"
        )

        file.write(
            "shap_waterfall_2.png\n"
        )

        file.write(
            "shap_waterfall_3.png\n"
        )

        file.write(
            "shap_feature_importance.csv\n"
        )

    print(
        f"\nSHAP report saved to: {report_path}"
    )

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()