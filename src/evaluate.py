import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute evaluation metrics on test data.
    
    Parameters:
        model: Fitted machine learning model.
        X_test (pd.DataFrame): Processed test features.
        y_test (pd.Series): True labels.
        
    Returns:
        dict: Dictionary containing precision, recall, f1, and accuracy.
    """
    predictions = model.predict(X_test)
    
    # We use macro average to handle multi-class classification imbalances
    return {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='macro', zero_division=0),
        'recall': recall_score(y_test, predictions, average='macro', zero_division=0),
        'f1': f1_score(y_test, predictions, average='macro', zero_division=0)
    }
