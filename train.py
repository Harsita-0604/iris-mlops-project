import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Enable MLflow
mlflow.set_experiment("Iris_Classification")

with mlflow.start_run():

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("n_estimators", 100)

    # Log metric
    mlflow.log_metric("accuracy", acc)

    # Save model
    model_path = "models/model.joblib"
    joblib.dump(model, model_path)

    # Log model in MLflow
    mlflow.sklearn.log_model(model, "model")

    print(f"Model trained with accuracy: {acc}")