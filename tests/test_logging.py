import io
import json
import math

from fastapi.testclient import TestClient
from app.logger import get_prediction_logger
from app.main import app, logger


def test_prediction_log(monkeypatch):

    # Synthetic input for testing
    payload = {
        "rainfall_mm": 750,
        "fertilizer_kg": 85,
        "sowing_delay_days": 15,
        "irrigation_count": 4,
        "soil_ph": 6.8,
        "farm_acres": 5.5,
        "seed_variety": "HD-2967",
        "tehsil": "Tehsil North"
    }

    # Capture stdout
    stream = io.StringIO()
    monkeypatch.setattr("sys.stdout", stream)
    monkeypatch.setattr(logger, "handlers", [])

    # Ensure logger does not create duplicate handlers
    assert get_prediction_logger() is get_prediction_logger() is logger

    handler, = logger.handlers
    assert handler.stream is stream
    assert not logger.propagate

    with TestClient(app) as client:

        # Valid prediction request
        response = client.post("/predict", json=payload)

        # Test invalid farm size
        assert client.post(
            "/predict",
            json={**payload, "farm_acres": -1}
        ).status_code == 422

        # Test invalid sowing delay
        assert client.post(
            "/predict",
            json={**payload, "sowing_delay_days": 66}
        ).status_code == 422

    assert response.status_code == 200

    # Verify exactly one JSON log
    line, = stream.getvalue().splitlines()
    event = json.loads(line)

    # Verify required fields in our logging schema
    assert set(event) == {
        "event",
        "model_version",
        "prediction",
        "latency_ms"
    }

    assert event["event"] == "prediction"

    assert type(event["model_version"]) is int
    assert event["model_version"] == 1

    assert type(event["prediction"]) is float
    assert math.isfinite(event["prediction"])

    assert type(event["latency_ms"]) is float
    assert 0 <= event["latency_ms"] < math.inf

    # Compare logged prediction with API response
    assert response.json() == {
        "prediction": event["prediction"]
    }