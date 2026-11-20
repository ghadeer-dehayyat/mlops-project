from fastapi.testclient import TestClient
from src.app.main import app

VALID_PAYLOAD = {
    "age": 55.0,
    "sex": 1.0,
    "cp": 2.0,
    "trestbps": 130.0,
    "chol": 240.0,
    "fbs": 0.0,
    "restecg": 1.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 1.2,
    "slope": 1.0,
    "ca": 0.0,
    "thal": 2.0,
}


def test_reject_non_numeric_age():
    bad = {**VALID_PAYLOAD, "age": "not-a-number"}
    with TestClient(app) as client:
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


def test_reject_missing_required_field():
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "chol"}
    with TestClient(app) as client:
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


def test_reject_out_of_range_age():
    bad = {**VALID_PAYLOAD, "age": 180.0}
    with TestClient(app) as client:
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
