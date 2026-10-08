import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/creditcard.csv")


def main():
    print("=" * 60)
    print("ShieldML - Dataset Inspection")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nFraud percentage:")
    fraud_percentage = df["Class"].mean() * 100
    print(f"{fraud_percentage:.4f}%")

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()