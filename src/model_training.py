import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_processing import load_data, clean_data, preprocess_data, split_data
from feature_engineering import encode_categorical


def train():
    print("Starting training pipeline...")

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    df = load_data("data/raw")
    df = clean_data(df)
    df = preprocess_data(df)

    # -----------------------------
    # Feature engineering
    # -----------------------------
    df = encode_categorical(df)

    # -----------------------------
    # Split dataset
    # -----------------------------
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # -----------------------------
    # Model setup
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # -----------------------------
    # MLflow tracking
    # -----------------------------
    with mlflow.start_run():

        print("Training model...")
        model.fit(X_train, y_train)

        print("Evaluating model...")
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        print(f"Accuracy: {acc}")

        # -----------------------------
        # Log parameters
        # -----------------------------
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        # -----------------------------
        # Log metrics
        # -----------------------------
        mlflow.log_metric("accuracy", acc)

        # -----------------------------
        # Log model artifact
        # -----------------------------
        mlflow.sklearn.log_model(model, "model")

    print("Training complete.")


if __name__ == "__main__":
    train()