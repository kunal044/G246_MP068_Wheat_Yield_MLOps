# Import required libraries
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

# Define project root directory
ROOT = Path(__file__).resolve().parents[2]

# Define raw dataset paths
TRAIN_PATH = ROOT / "data/raw/train.csv"
LATER_PATH = ROOT / "data/raw/later.csv"
HOLDOUT_PATH = ROOT / "data/raw/holdout_features.csv"

# Define target column
TARGET = "yield_qtl_per_acre"

# Numerical input features
NUMERIC_FEATURES = [
    "rainfall_mm",
    "fertilizer_kg",
    "sowing_delay_days",
    "irrigation_count",
    "soil_ph",
    "farm_acres"
]

# Categorical input features
CATEGORICAL_FEATURES = [
    "seed_variety",
    "tehsil"
]


# Load all three datasets
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    return train, later, holdout


# Calculate original dataset fingerprint
def calculate_hash():
    return hashlib.sha256(
        TRAIN_PATH.read_bytes()
    ).hexdigest()


# Detect exact duplicate records
def detect_duplicates(train):
    count = int(train.duplicated().sum())

    return {
        "present": count > 0,
        "count": count if count > 0 else None
    }
# Detect columns that are unavailable at prediction time
def detect_leakage(train, holdout):

    # Exclude the target from unavailable columns
    unavailable = (
        set(train.columns)
        - set(holdout.columns)
        - {TARGET}
    )

    return {
        "present": bool(unavailable),
        "columns": sorted(unavailable)
    }

# Detect the verified sentinel value in farm size
def detect_sentinel(train):

    column = "farm_acres"
    value = 999

    # Calculate the number of affected rows
    count = int((train[column] == value).sum())

    return {
        "present": count > 0,
        "column": column if count > 0 else None,
        "value": value if count > 0 else None,
        "count": count if count > 0 else None
    }    
# Detect abnormally large target values
def detect_target_typos(train):

    # Threshold selected from the observed distribution gap
    threshold = 100

    # Count suspicious target values
    count = int((train[TARGET] > threshold).sum())

    return {
        "present": count > 0,
        "count": count if count > 0 else None
    }
# Compare categorical levels with the clean dataset
def detect_spelling_variants(train, later):

    findings = {}

    for column in CATEGORICAL_FEATURES:

        # Extract unique categorical values
        train_levels = set(train[column].dropna())
        later_levels = set(later[column].dropna())

        # Identify unexpected categories
        unexpected = train_levels - later_levels

        findings[column] = sorted(unexpected)

    return findings

# Investigate possible unit differences across tehsils
def detect_unit_mixup(train):

    # Create a temporary copy of the dataset
    audit = train.copy()

    # Exclude the identified sentinel value
    audit["farm_acres"] = audit["farm_acres"].replace(
        999, np.nan
    )

    # Calculate group-wise numerical statistics
    statistics = audit.groupby("tehsil")[
        NUMERIC_FEATURES
    ].agg(["min", "mean", "max"])

    return statistics.round(2).T

# Investigate prolonged constant sensor readings
def detect_stuck_sensor(train):

    # Remove duplicate records for accurate analysis
    ordered = train.drop_duplicates().copy()

    # Convert dates and sort chronologically
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values(
        ["date", "record_id"]
    )

    findings = []

    for column in NUMERIC_FEATURES:

        # Identify consecutive identical readings
        runs = ordered[column].ne(
            ordered[column].shift()
        ).cumsum()

        # Summarize each constant run
        summary = ordered.groupby(runs).agg(
            value=(column, "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=(column, "size")
        )

        # Calculate duration of each run
        summary["days"] = (
            summary["end_date"] - summary["start_date"]
        ).dt.days + 1

        # Identify suspicious prolonged readings
        suspicious = summary[
            (summary["days"] >= 7) &
            (summary["rows"] >= 7)
        ]

        for _, row in suspicious.iterrows():

            findings.append({
                "column": column,
                "value": row["value"],
                "start_date": str(row["start_date"].date()),
                "end_date": str(row["end_date"].date())
            })

    return findings

    
    
# Main execution function
def main():

    # Load all datasets
    train, later, holdout = load_data()

    # Display dataset dimensions
    print("Training shape:", train.shape)
    print("Later shape:", later.shape)
    print("Holdout shape:", holdout.shape)

    # Display dataset fingerprint
    print("\nDataset SHA-256:")
    print(calculate_hash())

    # Run data quality checks
    print("\n1. Data leakage:")
    print(detect_leakage(train, holdout))

    print("\n2. Sentinel values:")
    print(detect_sentinel(train))

    print("\n3. Duplicate rows:")
    print(detect_duplicates(train))

    print("\n4. Target typos:")
    print(detect_target_typos(train))

    print("\n5. Spelling variants:")
    print(detect_spelling_variants(train, later))

        # Investigate unit conversion problems
    print("\n6. Unit mixup investigation:")
    print(detect_unit_mixup(train))

    # Investigate prolonged constant readings
    print("\n7. Stuck sensor investigation:")
    print(detect_stuck_sensor(train))

    # Add calls to unit mixup and stuck sensor
    # after transferring those notebook checks


# Execute the main function
if __name__ == "__main__":
    main()