# Data processing libraries
import pandas as pd
import yaml
import hashlib
from pathlib import Path

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Experiment tracking
import mlflow
import mlflow.sklearn

# Locate project root
ROOT = Path(__file__).resolve().parents[2]

# Load configuration settings
with open(ROOT / "params.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Extract feature names and target
NUMERIC_FEATURES = config["model"]["numeric_features"]
CATEGORICAL_FEATURES = config["model"]["categorical_features"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = config["model"]["target"]

# Build preprocessing and model pipeline
def build_pipeline(model):

    # Replace missing numerical values with the median
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Handle missing categories and encode text features
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Apply separate preprocessing to each feature type
    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer,
         CATEGORICAL_FEATURES)
    ])

    # Combine preprocessing and regression model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


# Prepare training and validation datasets
def prepare_data():

    # Load the cleaned training dataset
    df = pd.read_csv(
        ROOT / config["data"]["cleaned_train"]
    )

    # Select only approved prediction features
    X = df[FEATURES]

    # Select the prediction target
    y = df[TARGET]

    # Split into training and validation sets
    return train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )


# Calculate the original training dataset SHA-256
def calculate_data_hash():

    raw_path = ROOT / config["data"]["train"]

    return hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()


