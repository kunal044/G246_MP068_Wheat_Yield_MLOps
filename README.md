# Wheat Yield per Acre (Nagpur)

## Project Information

- Subject: SCSE3040 MLOps
- Group ID: G246
- Problem ID: MP068
- Batch: 56
- Faculty: Dr. Upasna Joshi

## Team Members

| Enrollment Number | Name |
|---|---|
| S24CSEU1439 | Amaskapoor |
| S24CSEU1479 | Kunal |
| S24CSEU1531 | Sarthak Shukla |
| S24CSEU1451 | Charchit Kanchav |

## Project Objective

Predict wheat yield per acre using agricultural
and environmental features.

## Machine Learning Task

- Task: Regression
- Target: yield_qtl_per_acre
- Evaluation metric: MAE

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Git and DVC
- MLflow
- FastAPI
- Pytest
- Docker
- GitHub Actions
- Kubernetes

## Data Quality Analysis

Four data problems were identified:

- Data leakage: yield_from_mandi_receipt
- Sentinel values: 69 rows
- Duplicate rows: 42
- Target typos: 12

## Run Data Analysis

    python src/wheat_yield/analyze.py

## Project Stages

- Stage 1: Data analysis and version control
- Stage 2: Cleaning and model training
- Stage 3: API, deployment and logging
- Stage 4: Monitoring and retraining
