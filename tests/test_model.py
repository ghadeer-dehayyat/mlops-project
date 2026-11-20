from pathlib import Path
from src.app.model import DEFAULT_MODEL_PATH, load_heart_model, predict_heart_disease


def test_model_file_exists_and_loads():
    model = load_heart_model(DEFAULT_MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict")


def test_model_loader_raises_on_missing_file():
    missing_path = Path("models/does_not_exist.joblib")
    try:
        load_heart_model(missing_path)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True


def test_predict_helper_returns_valid_tuple():
    model = load_heart_model(DEFAULT_MODEL_PATH)
    sample = {
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
    pred, label = predict_heart_disease(model, sample)
    assert pred in (0, 1)
    assert isinstance(label, str) and len(label) > 0
