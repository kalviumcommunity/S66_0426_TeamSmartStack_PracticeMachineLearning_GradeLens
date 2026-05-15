import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    """
    Fit a Random Forest model on processed training data.
    
    Parameters:
        X_train (pd.DataFrame): Processed feature dataset.
        y_train (pd.Series): Target labels.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed.
        
    Returns:
        RandomForestClassifier: The fitted model artifact.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model
