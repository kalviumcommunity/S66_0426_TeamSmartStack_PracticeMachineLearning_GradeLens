import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Construct a scikit-learn preprocessing pipeline for encoding and scaling.
    
    Parameters:
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
        
    Returns:
        ColumnTransformer: An unfitted scikit-learn preprocessing pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor
