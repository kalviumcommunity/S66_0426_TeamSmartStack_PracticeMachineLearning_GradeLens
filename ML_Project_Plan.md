# ML Project Plan: GradeLens (TeamSmartStack)

## 1. Problem Statement & Solution Overview
**The Problem:** Instructors and educational institutions struggle to identify at-risk students and "free-riders" in group projects early enough to intervene. Currently, evaluations happen at the end of the semester, when it is too late to offer help or correct team dynamics.
**The Solution:** GradeLens uses machine learning to predict a student's final performance category (e.g., High Performing, Average, At-Risk) based on mid-semester data such as attendance, assignment scores, and peer review feedback. 
**Why ML?** The relationship between a student's peer review scores, submission delays, and attendance is non-linear. Simple rule-based systems fail to capture nuanced behaviors (e.g., a student with high attendance but poor peer scores might be struggling with teamwork, while low attendance with high assignment scores might indicate a different profile). 
**Target Variable:** `Performance_Category` (Multi-class: High, Average, At-Risk).

## 2. Dataset Definition & Assessment

| Attribute | Details |
| :--- | :--- |
| **Dataset Name / Source** | Student Performance & Peer Review Dataset (Internal/Synthetic) |
| **Number of Rows** | ~2,500 students/records |
| **Number of Features** | 12 features (Attendance %, Avg Submission Delay, Peer Score Given, Peer Score Received, Midterm Score, etc.) |
| **Target Variable** | `Performance_Category` |
| **Task Type** | Multi-class Classification |
| **Class Balance** | Imbalanced: 65% Average, 20% High Performing, 15% At-Risk |
| **Missing Data** | `Peer Score Given/Received` has ~5% missing values (students who didn't submit reviews). |
| **Known Limitations** | Data may be specific to a single course structure; peer reviews can be highly subjective. |

**Assessment Checklist Answers:**
* **Enough Data?** Yes, 2500 rows is sufficient for traditional ML classification.
* **Handling Imbalance:** We will use class weighting (`class_weight='balanced'`) during training and evaluate using F1-score instead of accuracy.
* **Missing Values:** Missing peer reviews will be imputed with the median score or flagged with a "missing" indicator feature.
* **Data Leakage:** We will ensure the train/test split happens *before* imputation and scaling.

## 3. Scope & Boundaries
**✅ In Scope:**
* Exploratory Data Analysis (EDA) and data cleaning.
* Feature engineering (calculating derived features like "Review Discrepancy").
* Baseline model (Logistic Regression).
* Primary model (Random Forest Classifier) with hyperparameter tuning.
* Streamlit web dashboard MVP for uploading new data and viewing predictions.
* Saved model artifact (`model.pkl`).
* Evaluation report (Precision, Recall, F1).

**❌ Out of Scope:**
* Deep learning / Neural Networks.
* Real-time automated data ingestion from Canvas/Blackboard APIs.
* Automated continuous retraining pipelines (MLOps).
* Cloud deployment (AWS/GCP) — the MVP dashboard will run locally.

## 4. Roles & Responsibilities
*(Note: Assign team members to these roles before starting)*

| Role | Team Member | Key Responsibilities |
| :--- | :--- | :--- |
| **Data Lead** | *[Name]* | Dataset acquisition, EDA, handling missing values, creating data quality report. |
| **Feature Engineering Lead** | *[Name]* | Designing feature transformations, encoding categorical data, standardizing numerical features. |
| **Modeling Lead** | *[Name]* | Baseline & Primary model training, hyperparameter tuning, exporting `model.pkl`. |
| **Evaluation Lead** | *[Name]* | Designing evaluation metrics (F1, Confusion Matrix), auditing pipeline for data leakage. |
| **Integration Lead** | *[Name]* | Building the Streamlit dashboard, loading the model, writing the final `README.md`. |

## 5. Sprint Timeline (4 Weeks)

| Week | Focus Area | Milestones & Deliverables |
| :--- | :--- | :--- |
| **Week 1** | Setup, Data, & Exploration | Dataset loaded, EDA completed, missing values handled, project repository structure (`src/`, `data/`) finalized. |
| **Week 2** | Feature Engineering & Baseline | Preprocessing pipeline implemented (Scalers/Encoders), Baseline Logistic Regression trained, Baseline metrics recorded. |
| **Week 3** | Modeling & Evaluation | Random Forest trained, hyperparameters tuned (GridSearchCV), evaluation performed on test set without leakage. |
| **Week 4** | MVP Completion & Dashboard | Final model saved as `.pkl`, Streamlit `app.py` built to load model, README completed, final demo recorded. |

**Experiment Tracking & Reproducibility Plan:**
* All experiments will be logged in a shared spreadsheet (recording Model Type, Hyperparameters, F1-Score).
* `random_state=42` will be used across all splits and models.
* Dependencies will be pinned in a `requirements.txt` file.

## 6. MVP (Minimum Viable Product)
Our MVP is a functional, reproducible pipeline that takes raw student data, processes it, trains a model, and serves predictions via a basic Streamlit app.
* **Data Pipeline:** Code to load CSV, impute missing peer reviews, and split data.
* **Feature Engineering:** `scikit-learn` Pipeline object that scales numerical features.
* **Modeling:** A tuned Random Forest model saved as `model.pkl`.
* **Evaluation:** A classification report showing Precision, Recall, and F1-scores for the held-out test set.
* **Integration:** A Streamlit dashboard where an instructor can upload a CSV of new student data and see a table of predicted `Performance_Category` labels.

## 7. Functional Requirements
1. The preprocessing script must successfully handle missing peer review data without crashing.
2. The Streamlit dashboard must be able to load the serialized `model.pkl` file from disk.
3. The dashboard must accept a `.csv` file upload containing new student records and output predictions for each row.
4. The evaluation script must output Precision, Recall, and F1-scores (not just accuracy) due to class imbalance.

## 8. Non-Functional Requirements
1. **Reproducibility:** A new developer must be able to clone the repo, run `pip install -r requirements.txt`, and run the training script to get the exact same evaluation metrics.
2. **Correctness:** The test dataset must be strictly separated from the training dataset before any fitting of scalers or imputers occurs (Zero Data Leakage).
3. **Efficiency:** Model training and hyperparameter tuning must complete in under 10 minutes on a standard laptop.

## 9. Success Metrics
* [ ] The pipeline runs end-to-end (from raw data to `model.pkl`) without manual intervention.
* [ ] The Random Forest model outperforms the Baseline Logistic Regression by at least 5% on the macro F1-score.
* [ ] The test set is verified to be 100% unseen during training and feature engineering.
* [ ] The Streamlit dashboard launches locally and successfully returns predictions for a dummy CSV file.
* [ ] The `README.md` clearly explains how to set up the environment and run the app.

## 10. Risks & Mitigation

| Risk | Impact | Mitigation Plan |
| :--- | :--- | :--- |
| **Severe Class Imbalance ("At-Risk" is rare)** | Model simply predicts "Average" for everyone and gets high accuracy but is useless. | Use `class_weight='balanced'` in the model. Strictly evaluate using macro F1-score and confusion matrices. |
| **Data Leakage during Scaling** | Evaluation results are artificially high, model fails in production. | Ensure train/test split happens first. Use `sklearn.pipeline.Pipeline` to strictly isolate fit/transform steps. |
| **Streamlit Integration Issues** | We have a working model but no UI to show it off for the demo. | The Integration Lead will start building a dummy Streamlit app in Week 2, before the final model is ready, to ensure the UI works. |
| **Subjective Peer Reviews** | Features based on peer reviews are too noisy to be predictive. | Experiment with removing peer reviews in Week 3 to see if attendance and assignment scores alone provide a stronger signal. |
