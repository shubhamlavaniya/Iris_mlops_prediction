# src/api/app.py (Updated)
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.predict import load_production_model, predict
from src.utils import load_config
from prometheus_fastapi_instrumentator import Instrumentator
from src.logging import log_prediction, init_db, log_error

app = FastAPI(title="Iris Classifier API")
init_db()
Instrumentator().instrument(app).expose(app)
config = load_config("src/config/model_config.yaml")
model_name = config["model"]["registry_name"]
model = load_production_model(model_name)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

# --- NEW EXCEPTION HANDLER ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Extract details from the exception
    error_detail = exc.errors()
    # Log the error using your log_error function
    log_error(
        data={"input_data": exc.body},
        error_message=f"Validation Error: {error_detail}",
        version=model_name
    )
    # Return a 422 HTTP response to the client
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": error_detail},
    )

@app.post("/predict")
def make_prediction(features: IrisFeatures):
    input_data = features.model_dump()
    
    try:
        prediction = predict(model, input_data)
        log_prediction(input_data, str(prediction), model_name)
        return {"prediction": int(prediction)}
    
    except Exception as e:
        # This block will now only catch other, unexpected server errors (500s)
        log_error(input_data, str(e), model_name)
        raise HTTPException(status_code=500, detail="Prediction failed.")