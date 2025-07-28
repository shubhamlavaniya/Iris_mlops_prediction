# src/api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from src.api.predict import load_production_model, predict
from src.utils import load_config
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI(title="Iris Classifier API")

# Initialize Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Load production model on startup
# Load model config
config = load_config("src/config/model_config.yaml")
model_name = config["model"]["registry_name"]
model = load_production_model(model_name)


# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

@app.post("/predict")
def make_prediction(features: IrisFeatures):
    input_data = features.dict()
    prediction = predict(model, input_data)
    return {"prediction": int(prediction)}
