import pandas as pd
import mlflow.sklearn


def load_model(model_uri="runs:/latest/model"):
    """
    Load trained model from MLflow
    """
    print("Loading model from MLflow")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict(model, input_data):
    """
    Run inference on new data
    """
    print("Running inference")

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)

    return prediction[0]


if __name__ == "__main__":
    # Example usage
    model = load_model()

    if model:
        sample_input = {
            "age": 35,
            "farm_size_hectares": 5,
            "avg_fertilizer_kg": 200,
            "total_purchases": 10,
            "months_since_first_purchase": 12,
            "support_tickets": 2
        }

        result = predict(model, sample_input)
        print(f"Prediction: {result}")