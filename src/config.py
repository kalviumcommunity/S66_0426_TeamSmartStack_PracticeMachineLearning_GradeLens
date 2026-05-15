import os

class Config:
    # File Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'student_data.csv')
    DATA_PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'student_data_clean.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
    PIPELINE_PATH = os.path.join(BASE_DIR, 'models', 'preprocessing_pipeline.pkl')
    
    # Target and Features
    TARGET_COLUMN = 'Performance_Category'
    NUMERICAL_COLS = ['Attendance_Percent', 'Peer_Score_Given', 'Peer_Score_Received', 'Assignments_Avg']
    CATEGORICAL_COLS = ['Department']
    
    # Model configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_ESTIMATORS = 100
