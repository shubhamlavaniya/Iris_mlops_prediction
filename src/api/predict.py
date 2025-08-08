# src/api/predict.py

import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
import logging
import mlflow
import os

# ---- Ensure Logs Directory Exists ----
if not os.path.exists("logs"):
    os.makedirs("logs")

# ---- Setup Logging ----

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/prediction.log"),  # Save logs to a file
        logging.StreamHandler()                      # Also print logs to console
    ]
)

#### Uncomment if you want to set up CloudWatch logging
# # Set logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # CloudWatch-compatible log format
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# # Log to stdout (Docker/EC2 will forward this to CloudWatch if agent is set up)
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)


# ---- Model Feature Order ----
expected_columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "Petal_ratio", 
    "Sepal_area",
    "Petal_area"
]


# ---- Load Model from MLflow Registry ----
def load_production_model(model_name: str):

    mlflow.set_tracking_uri("file:/app/mlruns")
    logging.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Production":
            #model_uri = mv.source
            model_uri = f"runs:/{mv.run_id}/model"
            logging.info(f"Loaded production model: {model_uri}")
            #model = mlflow.pyfunc.load_model(model_uri)
            model = mlflow.pyfunc.load_model("file:/app/mlruns/739693317334162311/9a987bf0d97b495f8258454bd7a8a31d/artifacts/model")

            return model
    raise Exception(f"No model in Production stage for '{model_name}'.")

# ---- Preprocess Raw Input into Full Feature Set ----
def preprocess_input(input_data: dict):
    try:
        sepal_length = input_data["sepal_length"]
        sepal_width = input_data["sepal_width"]
        petal_length = input_data["petal_length"]
        petal_width = input_data["petal_width"]

        processed = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
            "Petal_area": petal_length * petal_width,
            "Petal_ratio": petal_length / petal_width if petal_width != 0 else 0,
            "Sepal_area": sepal_length * sepal_width,
        }

        logging.info(f"Processed input features: {processed}")
        return processed

    except KeyError as e:
        logging.error(f"Missing required input field: {e}")
        raise

# ---- Make Prediction ----
def predict(model, input_data: dict):
    logging.info(f"Received raw input: {input_data}")

    try:
        processed_data = preprocess_input(input_data)
        input_df = pd.DataFrame(
            [[processed_data[col] for col in expected_columns]],
            columns=expected_columns
        )

        predictions = model.predict(input_df)
        logging.info(f"Prediction output: {predictions[0]}")

        return predictions[0]

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise
