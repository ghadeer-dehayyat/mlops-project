from pydantic import BaseModel, Field


class HeartDiseaseInput(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: float = Field(..., ge=0, le=4, description="Chest pain type")
    trestbps: float = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: float = Field(..., ge=0, le=2, description="Resting electrocardiographic results")
    thalach: float = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=-5, le=10, description="ST depression induced by exercise")
    slope: float = Field(..., ge=0, le=3, description="Slope of the peak exercise ST segment")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: float = Field(..., ge=0, le=6, description="Thalassemia")


class PredictionResponse(BaseModel):
    prediction: int
    result: str
