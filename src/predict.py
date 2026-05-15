import pandas as pd
import joblib

def load_artifacts(model_path: str, pipeline_path: str) -> tuple:
    """
    Load serialized model and preprocessing pipeline from disk.
    
    Parameters:
        model_path (str): Path to the saved model artifact.
        pipeline_path (str): Path to the saved pipeline artifact.
        
    Returns:
        tuple: (fitted_model, fitted_pipeline)
    """
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline

def predict(new_data: pd.DataFrame, model, pipeline) -> pd.Series:
    """
    Generate predictions on new, unseen data using loaded artifacts.
    
    Parameters:
        new_data (pd.DataFrame): Raw new features.
        model: Fitted machine learning model.
        pipeline: Fitted preprocessing pipeline.
        
    Returns:
        pd.Series: Array of predictions.
    """
    # 1. Ensure the new data goes through the EXACT same transformations
    processed_data = pipeline.transform(new_data)
    
    # 2. Make predictions
    predictions = model.predict(processed_data)
    
    return pd.Series(predictions, name="Predicted_Performance")
