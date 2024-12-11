# DSW_INTERNSHIP_LOAN_PROJECT

Problem Statement
A Non-Banking Financial Company (NBFC) specializing in small loans aims to enhance its loan approval process by predicting loan repayment behavior. Specifically, the NBFC seeks to build a machine learning-based classification model to identify potential loan defaulters and non-defaulters. The task is to develop a robust training pipeline, evaluate multiple models, and select the most suitable one based on performance metrics.

Tasks and Implementation
1. Exploratory Data Analysis (EDA)
Objective: Understand the data, identify key patterns, and uncover potential preprocessing needs.

Deliverable:

A Jupyter Notebook named eda.ipynb that:
Provides descriptive statistics of the dataset.
Visualizes important features such as cibil_score, loan_amnt, and int_rate.
Investigates correlations between features and the target variable (loan_status).
Includes markdown explanations for each chart or analysis.
Key Steps:

Use pandas, matplotlib, and seaborn for data visualization.
Highlight trends in default rates across sub_grade, purpose, and emp_length.
Handle missing values and outliers as part of initial data analysis.

2. Modelling
Objective: Build an object-oriented training pipeline for loan default prediction.

Deliverable:

A Python script named model_.py containing:
LoanDefaultModel class (already provided by your script):
load: Reads train_data.xlsx and test_data.xlsx.
preprocess: Encodes categorical features, scales numerical features, and handles missing data.
train: Trains models (LogisticRegression and RandomForestClassifier).
test: Evaluates models on the test set with metrics like accuracy, confusion matrix, and classification report.
predict: Enables inference for new data points.
Multiple models:
Logistic Regression (for simplicity and interpretability).
Random Forest (for its ability to capture non-linear relationships).
3. Model Selection
Objective: Evaluate and compare models based on performance metrics and choose the best-performing one.

Deliverable:
A Jupyter Notebook named model_selection.ipynb that:
Loads the models trained in model_.py.
Performs model evaluation on validation data (test_data.xlsx).
Compares models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
Documents why the final model was chosen, considering both performance and interpretability.
Optionally includes hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

4. Deployment Preparation
Objective: Save models and preprocessing artifacts for production use.

Deliverable:
Save trained models (.pkl files) and preprocessing artifacts (like LabelEncoders and StandardScaler) using pickle or joblib.

Summary of Workflow
EDA:

Analyze distributions and correlations.
Visualize class imbalance (defaulters vs. non-defaulters).
Training Pipeline:

Use provided train_data.xlsx to train two models.
Perform preprocessing with scaling and encoding.
Train and save LogisticRegression and RandomForestClassifier.
Testing and Validation:

Evaluate trained models on test_data.xlsx.
Summarize metrics in model_selection.ipynb.
Model Selection:

Choose the best-performing model based on evaluation.
Justify selection in a written summary.
Final Deliverables:

EDA notebook: eda.ipynb
Training pipeline script: model_.py
Model selection notebook: model_selection.ipynb
Saved models and artifacts (.pkl files).

Key Notes
Use inline comments in Python scripts for better readability.
Ensure code adheres to Python 3.7+ standards.
Follow coding conventions such as PEP 8.
Ensure the submitted notebooks are run beforehand and include output cells.
Provide detailed markdown cells in notebooks explaining the results and decisions.
