"""
ML Model loading and prediction functionality.
"""
import os
import joblib
import numpy as np
from pathlib import Path
from django.conf import settings

# Global variables to store loaded model and scaler
_model = None
_scaler = None


def load_model():
    """Load the trained XGBoost model and scaler from joblib files."""
    global _model, _scaler
    
    if _model is not None and _scaler is not None:
        return _model, _scaler
    
    try:
        model_path = settings.MODEL_DIR / 'self_training_xgb_model.joblib'
        scaler_path = settings.MODEL_DIR / 'self_training_xgb_scaler.joblib'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        
        return _model, _scaler
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def predict(features):
    """
    Make a prediction using the loaded model.
    
    Args:
        features: numpy array or list of 27 features in the correct order:
            ['age', 'physic', 'mentality_aggression', 'mentality_interceptions',
             'power_stamina', 'power_strength', 'defending_marking_awareness',
             'power_jumping', 'defending_standing_tackle', 'defending_sliding_tackle',
             'attacking_heading_accuracy', 'mentality_composure', 'movement_reactions',
             'skill_long_passing', 'skill_dribbling', 'skill_fk_accuracy',
             'skill_ball_control', 'attacking_crossing', 'power_shot_power',
             'attacking_finishing', 'skill_curve', 'movement_balance',
             'attacking_volleys', 'power_long_shots', 'mentality_vision',
             'mentality_penalties', 'movement_agility']
    
    Returns:
        dict: {
            'prediction': int (0 or 1),
            'probability': float (0-1),
            'probability_class_1': float
        }
    """
    model, scaler = load_model()
    
    # Convert to numpy array and ensure correct shape
    features_array = np.array(features).reshape(1, -1)
    
    # Check feature count
    if features_array.shape[1] != 27:
        raise ValueError(f"Expected 27 features, got {features_array.shape[1]}")
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'probability': float(probabilities[1]),  # Probability of class 1 (big potential)
        'probability_class_1': float(probabilities[1]),
        'probability_class_0': float(probabilities[0]),
    }

