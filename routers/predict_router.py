from fastapi import APIRouter
from pydantic import BaseModel
from utils.predict import all_symptoms, enhanced_symptoms, predict_disease

router = APIRouter()

class PredictionRequest(BaseModel):
    symptoms: list

class EnhancedPredictionRequest(BaseModel):
    symptoms: list  # List of selected symptom names
    enhanced_features: dict  # Dict with feature column names and numeric values

@router.get("/symptoms")
def get_symptoms():
    """
    Get all 40 basic symptoms for basic mode
    Returns: {"symptoms": ["Fever", "Headache", ...]}
    """
    return {"symptoms": all_symptoms}

@router.get("/enhanced-symptoms")
def get_enhanced_symptoms():
    """
    Get enhanced symptoms structure for enhanced mode
    Returns: {
        "enhanced_symptoms": {
            "Fever Pattern": {
                "column": "fever_pattern",
                "options": ["No fever", "Continuous fever", ...]
            },
            ...
        }
    }
    """
    return {"enhanced_symptoms": enhanced_symptoms}

@router.post("/predict")
def predict(request: PredictionRequest):
    """
    Basic prediction using only 40 basic symptoms
    Request: {"symptoms": ["Fever", "Headache", ...]}
    Returns: {
        "predictions": [
            {
                "disease": "Malaria",
                "confidence": 85.5,
                "risk": "HIGH",
                "about": "...",
                "action": "..."
            },
            ...
        ]
    }
    """
    selected = request.symptoms
    if not selected:
        return {"predictions": []}
    
    try:
        result = predict_disease(selected, use_enhanced=False)
        return result
    except Exception as e:
        return {"error": f"Error: {e}"}

@router.post("/predict-enhanced")
def predict_enhanced(request: EnhancedPredictionRequest):
    """
    Enhanced prediction using basic symptoms + detailed follow-up answers
    Request: {
        "symptoms": ["Fever", "Headache", ...],
        "enhanced_features": {
            "fever_pattern": 2,
            "pain_location": 1,
            "rash_type": 0,
            "stool_character": 1,
            "urine_character": 0
        }
    }
    Returns: {
        "predictions": [
            {
                "disease": "Malaria",
                "confidence": 92.3,
                "risk": "HIGH",
                "about": "...",
                "action": "..."
            },
            ...
        ]
    }
    """
    selected = request.symptoms
    enhanced_features = request.enhanced_features
    
    if not selected:
        return {"predictions": []}
    
    try:
        result = predict_disease(
            selected,
            use_enhanced=True,
            enhanced_features_input=enhanced_features
        )
        return result
    except Exception as e:
        return {"error": f"Error: {e}"}