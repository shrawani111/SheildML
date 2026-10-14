import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path("data/creditcard.csv")
PROCESSED_DIR = Path("data/processed")


def create_time_features(df):
    """Create leakage-safe time features from transaction timestamp."""

    df = df.copy()

    # Dataset Time is measured in seconds from the first transaction.
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)

    # Cyclical encoding of hour of day
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # Relative day
    df["Day"] = (df["Time"] // 86400).astype(int)

    return df


def create_amount_features(df):
    """Create transaction amount features."""

    df = df.copy()

    # Log transformation reduces extreme amount skew.
    df["Amount_Log"] = np.log1p(df["Amount"])

    # Rolling amount statistics are calculated using only previous
    # transactions because the current transaction is shifted out.
    df["Amount_Rolling_Mean_10"] = (
        df["Amount"]
        .shift(1)
        .rolling(window=10, min_periods=1)
        .mean()
    )

    df["Amount_Rolling_Std_10"] = (
        df["Amount"]
        .shift(1)
        .rolling(window=10, min_periods=2)
        .std()
        .fillna(0)
    )

    # Difference from historical transaction amount
    df["Amount_Deviation"] = (
        df["Amount"] - df["Amount_Rolling_Mean_10"]
    )

    return df


def create_behavioral_features(df):
    """Create transaction-behavior proxy features."""

    df = df.copy()

    # Time since previous transaction
    df["Time_Since_Previous"] = df["Time"].diff().fillna(0)

    # Number of transactions observed in previous 1 hour.
    # This uses only transactions before the current transaction.
    time_values = df["Time"].values

    previous_1h_count = np.zeros(len(df), dtype=np.int32)

    left = 0

    for i in range(len(df)):
        current_time = time_values[i]

        while left < i and time_values[left] < current_time - 3600:
            left += 1

        previous_1h_count[i] = i - left

    df["Previous_1H_Transaction_Count"] = previous_1h_count

    return df


def main():

    print("=" * 70)
    print("ShieldML - Data Preparation")
    print("=" * 70)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading dataset...")
    df = pd.read_csv(RAW_PATH)

    print(f"Original shape: {df.shape}")

    # Sort chronologically.
    df = df.sort_values("Time").reset_index(drop=True)

    # Remove exact duplicate transactions.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    print(f"Duplicate rows removed: {removed}")
    print(f"Shape after duplicate removal: {df.shape}")

    # Feature engineering
    print("\nCreating time features...")
    df = create_time_features(df)

    print("Creating amount features...")
    df = create_amount_features(df)

    print("Creating behavioral features...")
    df = create_behavioral_features(df)

    # Remove rows with undefined rolling statistics.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Time-based split.
    # 70% train, 15% validation, 15% test.
    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("\nTime-based split:")
    print(f"Train:      {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Test:       {test_df.shape}")

    print("\nFraud distribution:")
    print(
        f"Train fraud rate:      {train_df['Class'].mean() * 100:.4f}%"
    )
    print(
        f"Validation fraud rate: {val_df['Class'].mean() * 100:.4f}%"
    )
    print(
        f"Test fraud rate:       {test_df['Class'].mean() * 100:.4f}%"
    )

    # Save
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("\nSaved:")
    print("  data/processed/train.csv")
    print("  data/processed/validation.csv")
    print("  data/processed/test.csv")

    print("\nFeature columns:")
    print(df.columns.tolist())

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()