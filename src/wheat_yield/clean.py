# Import required libraries
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Locate the project root
ROOT = Path(__file__).resolve().parents[2]

# Load settings from params.yaml
with open(ROOT / "params.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# Clean the training dataset
def clean_data(df):

    # Create a copy to protect the original data
    df = df.copy()

    # Remove exact duplicate records
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove the leaking feature
    leakage = config["cleaning"]["leakage_column"]
    df = df.drop(columns=[leakage], errors="ignore")

    # Replace invalid farm sizes with missing values
    column = config["cleaning"]["sentinel_column"]
    value = config["cleaning"]["sentinel_value"]

    df[column] = df[column].replace(value, np.nan)

    # Correct target typing errors
    target = config["model"]["target"]
    threshold = config["cleaning"]["target_typo_threshold"]
    factor = config["cleaning"]["target_correction_factor"]

    mask = df[target] > threshold

    df.loc[mask, target] = df.loc[mask, target] / factor

    # Return the cleaned dataframe
    return df