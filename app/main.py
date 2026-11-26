# Import API and validation libraries
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

# Import model loading and data processing libraries
import joblib
import pandas as pd
from pathlib import Path

# Locate the project directory
ROOT = Path(__file__).resolve().parents[1]

# Load the trained champion model
model = joblib.load(ROOT / "models/model.joblib")

# Initialize FastAPI
app = FastAPI(
    title="Wheat Yield Prediction API",
    version="1.0"
)


# Define the eight required input features
class PredictionRequest(BaseModel):

    # Reject additional unexpected fields
    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float
    fertilizer_kg: float

    # Reject sowing delays above 65 days
    sowing_delay_days: int = Field(..., le=65)

    irrigation_count: int
    soil_ph: float

    # Reject negative farm sizes
    farm_acres: float = Field(..., ge=0)

    seed_variety: str
    tehsil: str


# Health endpoint for checking service availability
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert request into a single-row dataframe
    input_df = pd.DataFrame([request.model_dump()])

    # Generate prediction
    prediction = model.predict(input_df)[0]

    # Return prediction as a JSON-compatible float
    return {"prediction": float(prediction)}