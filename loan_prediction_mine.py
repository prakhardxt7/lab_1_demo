import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
import mlflow
import mlflow.sklearn
import os
from matplotlib import pyplot as plt

# Load dataset
df = pd.read_csv('train.csv')

# Split features and target
X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df['Loan_Status']

# Split numerical and categorical features
nums = X.select_dtypes(include=np.number).columns.tolist()
cats = X.select_dtypes(include='object').columns.tolist()

# Data preprocessing
# 1. Handle missing values
for col in nums:
    X[col].fillna(X[col].median(), inplace=True)

for col in cats:
    X[col].fillna(X[col].mode()[0], inplace=True)

# 2. Outlier handling
low = X[nums].quantile(0.05)
high = X[nums].quantile(0.95)
X[nums] = X[nums].clip(lower=low, upper=high, axis=1)

# 3. Feature transformations
X['LoanAmount'] = np.log(X['LoanAmount'] + 1)  # Avoid log(0)
X['TotalIncome'] = np.log(X['ApplicantIncome'] + X['CoapplicantIncome'] + 1)
X.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)

# 4. Encoding
le = LabelEncoder()
for col in cats:
    X[col] = le.fit_transform(X[col])

y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter grids
rf_params = {
    'max_depth': [5, 10],
    'max_features': [5, 7],
    'min_samples_split': [2, 5],
    'n_estimators': [100, 200]
}
gb_params = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}
lr_params = {
    'C': [1],  # Default value of C
    'max_iter': [500]  # Increased max_iter for more iterations
}

dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

# Initialize RandomizedSearchCV for all models
models_with_params = [
    ("RF", RandomForestClassifier(), rf_params),
    ("GB", GradientBoostingClassifier(), gb_params),
    ("DT", DecisionTreeClassifier(), dt_params),
    ("LR", LogisticRegression(solver='liblinear'), lr_params)
]

# Find best parameters
best_params = {}
for name, model, params in models_with_params:
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=10,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=2,  # Reduce parallelism to avoid memory issues
        random_state=42
    )
    try:
        search.fit(X_train_scaled, y_train)  # Fit with scaled data
        best_params[name] = search.best_params_
    except Exception as e:
        print(f"Error during training {name}: {e}")

# Log models and metrics to MLflow
def mlflow_logging(model, X_train, X_test, y_train, y_test, name):
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")  # Ensure the tracking URI is set
    mlflow.set_experiment("Loan_Prediction")  # Set the experiment name consistently # hoohaahoohaa
    
    with mlflow.start_run(run_name=name):  # Specify meaningful run name # hoohaahoohaa
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1_Score", f1)

        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
            mlflow.log_metric("AUC", auc)

            # Log ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            # Absolute path for saving ROC curve
            artifact_path = os.path.join("E:/Praxis-Learning/Final_Projects/MLOP_LOAN_Prac/MLFlow_Model", "roc_curve.png")
            plt.savefig(artifact_path)
            mlflow.log_artifact(artifact_path)

        mlflow.sklearn.log_model(model, name)
        # Log hyperparameters for better tracking
        mlflow.log_param("Model_Type", name)
        mlflow.log_param("Best_Parameters", best_params[name])

# Train and log each model
for name, model, params in models_with_params:
    if name in best_params:  # Skip if search failed
        final_model = model.set_params(**best_params[name])
        mlflow_logging(final_model, X_train, X_test, y_train, y_test, name)  # hoohaahoohaa
