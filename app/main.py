# Import required libraries
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from app.logger import get_prediction_logger


# Locate the project root directory
ROOT = Path(__file__).resolve().parents[1]


# Load the trained champion model
model = joblib.load(ROOT / "models" / "model.joblib")


# Current deployed model version
MODEL_VERSION = 1


# Initialize structured JSON prediction logger
logger = get_prediction_logger()


# Initialize FastAPI application
app = FastAPI(
    title="Wheat Yield Prediction API",
    description="MP068 - Wheat Yield per Acre (Nagpur)",
    version="1.0"
)


# Define the prediction request schema
class PredictionRequest(BaseModel):

    # Reject unexpected input fields
    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float
    fertilizer_kg: float

    # Sowing delay must not exceed 65 days
    sowing_delay_days: int = Field(..., le=65)

    irrigation_count: int
    soil_ph: float

    # Farm size cannot be negative
    farm_acres: float = Field(..., ge=0)

    seed_variety: str
    tehsil: str


# Health endpoint for Docker and Kubernetes
@app.get("/health")
def health():

    return {"status": "ok"}


# Wheat yield prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert validated input into a Pandas DataFrame
    input_df = pd.DataFrame([request.model_dump()])

    # Record inference start time
    start = perf_counter()

    # Generate wheat yield prediction
    prediction = float(model.predict(input_df)[0])

    # Calculate inference latency in milliseconds
    latency_ms = (perf_counter() - start) * 1000

    # Generate one structured JSON log per successful prediction
    logger.info(
        "prediction",
        extra={
            "prediction_fields": {
                "event": "prediction",
                "model_version": MODEL_VERSION,
                "prediction": prediction,
                "latency_ms": latency_ms
            }
        }
    )

    # Return the predicted wheat yield
    return {"prediction": prediction}