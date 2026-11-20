from fastapi.testclient import TestClient
from src.app.main import app


def test_health_endpoint_returns_200():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_health_endpoint_reports_healthy_and_model_loaded():
    with TestClient(app) as client:
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_swagger_docs_endpoint_available():
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200
