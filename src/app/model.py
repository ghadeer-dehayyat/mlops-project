from pathlib import Path
from typing import Any
import joblib
import pandas as pd

FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

DEFAULT_MODEL_PATH = Path("models/model.joblib")


def load_heart_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def predict_heart_disease(model: Any, payload_dict: dict[str, float]) -> tuple[int, str]:
    ordered_row = {col: payload_dict[col] for col in FEATURE_ORDER}
    df = pd.DataFrame([ordered_row], columns=FEATURE_ORDER)
    raw_pred = model.predict(df)[0]
    pred_int = int(raw_pred)
    label = "Heart disease detected" if pred_int == 1 else "No heart disease detected"
    return pred_int, label
