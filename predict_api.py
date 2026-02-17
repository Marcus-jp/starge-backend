from fastapi import APIRouter
from pydantic import BaseModel
from utils.predict import all_symptoms, predict_disease  # your predict.py functions

router = APIRouter()

class PredictionRequest(BaseModel):
    symptoms: list

@router.get("/symptoms")
def get_symptoms():
    return {"symptoms": all_symptoms}

@router.post("/predict")
def predict(request: PredictionRequest):
    selected = request.symptoms
    if not selected:
        return {"prediction": "No symptoms selected"}
    try:
        result = predict_disease(selected)
        return {"prediction": result}
    except Exception as e:
        return {"prediction": f"Error: {e}"}
