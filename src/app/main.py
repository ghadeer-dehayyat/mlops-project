from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.app.model import DEFAULT_MODEL_PATH, load_heart_model, predict_heart_disease
from src.app.schemas import HeartDiseaseInput, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_heart_model(DEFAULT_MODEL_PATH)
        app.state.model_error = None
    except Exception as exc:
        app.state.model = None
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational FastAPI service for heart disease prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    model = getattr(app.state, "model", None)
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "detail": getattr(app.state, "model_error", "Model not loaded"),
        }
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartDiseaseInput):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        pred_int, label = predict_heart_disease(model, payload.model_dump())
        return PredictionResponse(prediction=pred_int, result=label)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(exc)}")
