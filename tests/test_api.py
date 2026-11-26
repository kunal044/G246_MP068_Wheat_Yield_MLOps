# Import testing libraries
import math
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI application
from app.main import app

# Create a test client without starting the server
client = TestClient(app)


# Create synthetic input data for testing
@pytest.fixture
def valid_input():
    return {
        "rainfall_mm": 750,
        "fertilizer_kg": 85,
        "sowing_delay_days": 15,
        "irrigation_count": 4,
        "soil_ph": 6.8,
        "farm_acres": 5.5,
        "seed_variety": "HD-2967",
        "tehsil": "Tehsil North"
    }


# Test 1: Verify health endpoint
def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# Test 2: Verify successful prediction
def test_valid_prediction(valid_input):

    response = client.post(
        "/predict", json=valid_input
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert isinstance(result["prediction"], (int, float))
    assert math.isfinite(result["prediction"])


# Test 3: Reject invalid sowing delay
def test_invalid_sowing_delay(valid_input):

    valid_input["sowing_delay_days"] = 66

    response = client.post(
        "/predict", json=valid_input
    )

    assert response.status_code == 422


# Test 4: Reject negative farm size
def test_negative_farm_size(valid_input):

    valid_input["farm_acres"] = -1

    response = client.post(
        "/predict", json=valid_input
    )

    assert response.status_code == 422


# Test 5: Reject missing required features
def test_missing_feature(valid_input):

    del valid_input["rainfall_mm"]

    response = client.post(
        "/predict", json=valid_input
    )

    assert response.status_code == 422


# Test 6: Verify maximum allowed sowing delay
def test_sowing_delay_boundary(valid_input):

    valid_input["sowing_delay_days"] = 65

    response = client.post(
        "/predict", json=valid_input
    )

    assert response.status_code == 200


# Test 7: Verify minimum allowed farm size
def test_farm_size_boundary(valid_input):

    valid_input["farm_acres"] = 0

    response = client.post(
        "/predict", json=valid_input
    )

    assert response.status_code == 200

    