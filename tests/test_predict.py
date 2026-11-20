from fastapi.testclient import TestClient
from src.app.main import app

VALID_PAYLOAD = {
    "age": 63.0,
    "sex": 1.0,
    "cp": 1.0,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1.0,
    "restecg": 2.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 2.3,
    "slope": 3.0,
    "ca": 0.0,
    "thal": 6.0,
}


def test_predict_valid_payload_returns_200():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200


def test_predict_response_schema_and_values():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert "prediction" in data and "result" in data
        assert data["prediction"] in (0, 1)
        assert data["result"] in (
            "Heart disease detected",
            "No heart disease detected",
        )


def test_predict_handles_unloaded_model_gracefully():
    with TestClient(app) as client:
        saved_model = app.state.model
        app.state.model = None
        try:
            response = client.post("/predict", json=VALID_PAYLOAD)
            assert response.status_code == 503
        finally:
            app.state.model = saved_model
