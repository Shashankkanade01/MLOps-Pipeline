import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("data/dataset.csv")

# Split features and target (assuming last column is target)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set the experiment name
mlflow.set_experiment("MLOps-Pipeline-Experiment")

with mlflow.start_run():
    # Model parameters
    n_estimators = 100
    random_state = 42

    # Train the model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ✅ TODO 1: Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # ✅ TODO 2: Save model locally and log to MLflow model registry
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.joblib"
    joblib.dump(model, model_path)

    # Log model to MLflow (will go to MLflow tracking directory)
    mlflow.sklearn.log_model(model, "model", registered_model_name="mlops_pipeline_model")

    print(f"✅ Model trained and logged to MLflow. R2: {r2:.3f}, MSE: {mse:.3f}")

