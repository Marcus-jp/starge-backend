# predict.py - for FastAPI backend (NO SKLEARN)
import joblib
import numpy as np
import os

# ========== DISEASE INFORMATION ==========
disease_info = {
    "Malaria": {"risk": "HIGH", "about": "Mosquito-borne disease common in Kenya.", "action": "• Get malaria test\n• Use mosquito nets"},
    "Typhoid": {"risk": "HIGH", "about": "Bacterial infection from contaminated food/water.", "action": "• Seek medical attention\n• Ensure safe water"},
    "Pneumonia": {"risk": "HIGH", "about": "Serious lung infection.", "action": "• Seek medical care\n• Rest and fluids"},
    "Tuberculosis": {"risk": "HIGH", "about": "Chronic lung infection.", "action": "• Medical consultation\n• Long-term treatment"},
    "HIV_AIDS": {"risk": "HIGH", "about": "Immunodeficiency virus.", "action": "• Get HIV test\n• Medical management"},
    "COVID-19": {"risk": "HIGH", "about": "Respiratory viral disease.", "action": "• Self-isolate\n• Get tested"},
    "Dengue": {"risk": "HIGH", "about": "Mosquito-borne viral disease.", "action": "• Prevent mosquito bites\n• Rest and hydrate"},
    "Cholera": {"risk": "HIGH", "about": "Severe diarrheal disease.", "action": "• Oral rehydration\n• Seek medical care"},
    "UTI": {"risk": "MEDIUM", "about": "Urinary Tract Infection.", "action": "• Drink water\n• May need antibiotics"},
    "Diabetes": {"risk": "MEDIUM", "about": "Affects blood sugar regulation.", "action": "• Monitor blood sugar\n• Healthy diet"},
    "Hypertension": {"risk": "MEDIUM", "about": "High blood pressure.", "action": "• Check blood pressure\n• Reduce salt intake"},
    "Asthma": {"risk": "MEDIUM", "about": "Chronic respiratory condition.", "action": "• Use inhaler\n• Avoid triggers"},
    "Gastroenteritis": {"risk": "MEDIUM", "about": "Stomach flu.", "action": "• Rest and rehydrate\n• BRAT diet"},
    "Hepatitis": {"risk": "HIGH", "about": "Liver inflammation.", "action": "• Avoid alcohol\n• Healthy diet"},
    "Measles": {"risk": "HIGH", "about": "Highly contagious viral infection.", "action": "• Isolate\n• Vaccinate contacts"},
    "Chickenpox": {"risk": "MEDIUM", "about": "Viral disease with itchy rash.", "action": "• Avoid scratching\n• Calamine lotion"},
    "Bronchitis": {"risk": "MEDIUM", "about": "Inflammation of bronchial tubes.", "action": "• Rest\n• Drink fluids"},
    "Rheumatic_Fever": {"risk": "HIGH", "about": "Inflammatory disease.", "action": "• Antibiotic treatment\n• Rest"},
    "Brucellosis": {"risk": "MEDIUM", "about": "Bacterial infection from animals.", "action": "• Antibiotic treatment"},
    "Leptospirosis": {"risk": "MEDIUM", "about": "Bacterial infection from animal urine.", "action": "• Antibiotic treatment"}
}

# ========== BASIC SYMPTOMS (40 symptoms) ==========
basic_symptoms = [
    "Fever", "Headache", "Cough", "Fatigue", "Nausea",
    "Vomiting", "Diarrhea", "Abdominal Pain", "Body Aches",
    "Chills", "Sweating", "Loss Of Appetite", "Weight Loss",
    "Shortness Of Breath", "Chest Pain", "Sore Throat",
    "Runny Nose", "Rash", "Itchy Skin", "Jaundice (Yellow Skin)",
    "Dark Urine", "Pale Stools", "Muscle Pain", "Joint Pain",
    "Swollen Joints", "Frequent Urination", "Burning Urination",
    "Blood In Urine", "Excessive Thirst", "Blurred Vision",
    "Dizziness", "Confusion", "Seizures", "Stiff Neck",
    "Sensitivity To Light", "Dehydration", "Swollen Glands",
    "Night Sweats", "Coughing Blood", "Wheezing"
]

# ========== ENHANCED SYMPTOMS (5 detailed follow-up questions) ==========
# These map to the columns in your enhanced dataset
enhanced_symptoms = {
    "Fever Pattern": {
        "column": "fever_pattern",
        "options": ["No fever", "Continuous fever", "Cyclical fever (comes & goes)", "Sudden high fever"]
    },
    "Pain Location": {
        "column": "pain_location",
        "options": ["No significant pain", "Joint pain", "Muscle pain", "Abdominal pain", "Chest pain", "Headache"]
    },
    "Rash Type": {
        "column": "rash_type",
        "options": ["No rash", "Blisters", "Flat red rash", "Rose spots", "Petechial spots"]
    },
    "Stool Character": {
        "column": "stool_character",
        "options": ["Normal", "Watery diarrhea", "Rice-water stools", "Pale/clay-colored", "Bloody"]
    },
    "Urine Changes": {
        "column": "urine_character",
        "options": ["Normal", "Dark urine", "Blood in urine", "Frequent urination", "Burning sensation"]
    }
}

# Mapping enhanced features values for model input
enhanced_mapping = {
    "No fever": 0, "Continuous fever": 1, "Cyclical fever (comes & goes)": 2, "Sudden high fever": 3,
    "No significant pain": 0, "Joint pain": 1, "Muscle pain": 2, "Abdominal pain": 3, "Chest pain": 4, "Headache": 5,
    "No rash": 0, "Blisters": 1, "Flat red rash": 2, "Rose spots": 3, "Petechial spots": 4,
    "Normal": 0, "Watery diarrhea": 1, "Rice-water stools": 2, "Pale/clay-colored": 3, "Bloody": 4,
    "Normal": 0, "Dark urine": 1, "Blood in urine": 2, "Frequent urination": 3, "Burning sensation": 4
}

# ========== MODEL LOADING ==========
def load_model(use_enhanced=False):
    """Load model, encoder, and feature names"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        if use_enhanced:
            model_file = os.path.join(base_dir, "starge_model_enhanced.pkl")
            encoder_file = os.path.join(base_dir, "starge_model_enhanced_encoder.pkl")
            features_file = os.path.join(base_dir, "starge_model_enhanced_features.pkl")
        else:
            model_file = os.path.join(base_dir, "starge_model_basic.pkl")
            encoder_file = os.path.join(base_dir, "starge_model_basic_encoder.pkl")
            features_file = os.path.join(base_dir, "starge_model_basic_features.pkl")
        
        model = joblib.load(model_file)
        encoder = joblib.load(encoder_file)
        features = joblib.load(features_file)
        return model, encoder, features
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, []

# ========== ALL SYMPTOMS ==========
all_symptoms = basic_symptoms  # For frontend API /symptoms endpoint

# ========== PREDICTION FUNCTION ==========
def predict_disease(selected_symptoms, use_enhanced=False, enhanced_features_input=None):
    """
    selected_symptoms: list of symptom names (strings)
    use_enhanced: bool
    enhanced_features_input: dict with enhanced features
    
    Returns: dict with "predictions" key containing list of predicted diseases
    """
    model, encoder, feature_names = load_model(use_enhanced)
    if model is None:
        return {"error": "Model not loaded"}

    # Map GUI names to feature names
    symptom_mapping = {s: s.lower().replace(" ", "_").replace("(", "").replace(")", "") for s in basic_symptoms}

    input_data = []
    for feature in feature_names:
        # check basic symptoms
        val = 0
        for s in selected_symptoms:
            mapped = symptom_mapping.get(s)
            if mapped == feature:
                val = 1
                break
        # check enhanced
        if enhanced_features_input and feature in enhanced_features_input:
            val = enhanced_features_input[feature]
        input_data.append(val)
    
    # Convert to numpy array instead of pandas DataFrame
    input_array = np.array([input_data])
    
    try:
        probs = model.predict_proba(input_array)[0]
        top_3_idx = np.argsort(probs)[::-1][:3]
        predictions = []
        
        for i in top_3_idx:
            # Use encoder.classes_ instead of inverse_transform (no sklearn needed!)
            disease = encoder.classes_[i]
            confidence = float(probs[i] * 100)
            info = disease_info.get(disease, {})
            
            predictions.append({
                "disease": disease,
                "confidence": confidence,
                "risk": info.get("risk", "UNKNOWN"),
                "about": info.get("about", ""),
                "action": info.get("action", "")
            })
        
        return {"predictions": predictions}
    except Exception as e:
        return {"error": f"Prediction error: {e}"}