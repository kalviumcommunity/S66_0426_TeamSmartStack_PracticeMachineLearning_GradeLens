import os
import joblib

from src.config import Config
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import load_artifacts, predict

def main():
    print("Starting ML Pipeline...")
    
    # 1. Load Data
    print(f"Loading data from {Config.DATA_RAW_PATH}")
    df = load_data(Config.DATA_RAW_PATH)
    
    # 2. Clean Data
    print("Cleaning data...")
    df_clean = clean_data(df)
    
    # 3. Split Data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(
        df_clean, 
        target_column=Config.TARGET_COLUMN,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )
    
    # 4. Feature Engineering
    print("Building and fitting preprocessing pipeline...")
    pipeline = build_preprocessing_pipeline(Config.NUMERICAL_COLS, Config.CATEGORICAL_COLS)
    
    # Notice we ONLY fit on training data
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # 5. Train Model
    print("Training Random Forest model...")
    model = train_model(
        X_train_processed, 
        y_train, 
        n_estimators=Config.N_ESTIMATORS, 
        random_state=Config.RANDOM_STATE
    )
    
    # 6. Evaluate Model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test_processed, y_test)
    print("Evaluation Metrics on Test Set:")
    for metric, value in metrics.items():
        print(f"  - {metric.capitalize()}: {value:.4f}")
        
    # 7. Persistence
    print("Saving artifacts...")
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    joblib.dump(model, Config.MODEL_PATH)
    joblib.dump(pipeline, Config.PIPELINE_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")
    print(f"Pipeline saved to {Config.PIPELINE_PATH}")

    # 8. Test Prediction loading
    print("Testing artifact loading and prediction...")
    loaded_model, loaded_pipeline = load_artifacts(Config.MODEL_PATH, Config.PIPELINE_PATH)
    
    # Grab the first row of X_test for dummy prediction
    sample_data = X_test.iloc[[0]] 
    pred = predict(sample_data, loaded_model, loaded_pipeline)
    print(f"Sample prediction: {pred.values[0]}")

if __name__ == "__main__":
    main()
