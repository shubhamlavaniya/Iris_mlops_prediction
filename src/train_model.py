# This script is used to train and tune machine learning models using Optuna for hyperparameter optimization.
# It loads the Iris dataset, splits it into training and validation sets, and optimizes multiple models.
# The best model is registered in MLflow and promoted to production.

import pandas as pd
import optuna
import importlib
import yaml
import mlflow
import mlflow.sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from mlflow.tracking import MlflowClient
from utils import load_config

# Load config file
config = load_config("src/config/model_config.yaml")
models_config = config["models"]
target_column = config["model"]["target_column"]

# Load dataset from config
df = pd.read_csv(config["data"]["processed"])
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data: train -> train + val, test 
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"]
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# Track best model across all candidates
best_model_name = None
best_model_score = -1  # Initialize with a low score
best_inference_time = float('inf') # Initialize with a high value for comparison
best_model_instance = None
best_model_params = {}
best_model_module_class = ""
best_run_id = None

# Set MLflow experiment
mlflow.set_experiment(config["experiment"]["name"])

# Loop through each model
for model_name, model_info in models_config.items():
    print(f"\n Tuning model: {model_name}")
    
    def objective(trial):
        trial_params = {}
        search_space = model_info.get("optuna_search_space", {})
        for param_name, param_def in search_space.items():
            if param_def["type"] == "int":
                trial_params[param_name] = trial.suggest_int(param_name, param_def["low"], param_def["high"])
            elif param_def["type"] == "float":
                trial_params[param_name] = trial.suggest_float(param_name, param_def["low"], param_def["high"], log=param_def.get("log", False))
            elif param_def["type"] == "categorical":
                trial_params[param_name] = trial.suggest_categorical(param_name, param_def["choices"])

        # Dynamic model loading
        module_name, class_name = model_info["class"].rsplit(".", 1)
        ModelClass = getattr(importlib.import_module(module_name), class_name)
        model = ModelClass(**trial_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        # Use macro F1-score for optimization
        f1_macro = f1_score(y_val, preds, average='macro')
        return f1_macro

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    # Best from this model
    model_best_params = study.best_params
    model_best_score = study.best_value
    print(f"{model_name} best F1-score: {model_best_score:.4f} with params: {model_best_params}")

    # Train final model with best params
    module_name, class_name = model_info["class"].rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_name), class_name)
    final_model = ModelClass(**model_best_params)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{model_name}_run") as run:
        final_model.fit(X_train_full, y_train_full)
        
        # --- INFERENCE TIME MEASUREMENT ---
        start_inference_time = time.time()
        preds = final_model.predict(X_test)
        inference_time = time.time() - start_inference_time
        
        # Calculate and log all key metrics
        test_acc = accuracy_score(y_test, preds)
        test_f1_macro = f1_score(y_test, preds, average='macro')
        test_precision_macro = precision_score(y_test, preds, average='macro', zero_division=0)
        test_recall_macro = recall_score(y_test, preds, average='macro')

        mlflow.log_params(model_best_params)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1_macro)
        mlflow.log_metric("test_precision_macro", test_precision_macro)
        mlflow.log_metric("test_recall_macro", test_recall_macro)
        mlflow.log_metric("inference_time", inference_time) # Log the new metric
        mlflow.sklearn.log_model(final_model, "model")

        print(f"Logged {model_name} to MLflow with metrics: Accuracy={test_acc:.4f}, F1={test_f1_macro:.4f}, Inference Time={inference_time:.4f}s")

        # --- LOGIC FOR BEST MODEL SELECTION ---
        # If the new model's F1-score is higher
        if test_f1_macro > best_model_score:
            best_model_score = test_f1_macro
            best_model_name = model_name
            best_inference_time = inference_time
            best_model_instance = final_model
            best_model_params = model_best_params
            best_model_module_class = model_info["class"]
            best_run_id = run.info.run_id
        # OR if the F1-scores are tied, but the new model is faster
        elif test_f1_macro == best_model_score and inference_time < best_inference_time:
            best_model_score = test_f1_macro
            best_model_name = model_name
            best_inference_time = inference_time
            best_model_instance = final_model
            best_model_params = model_best_params
            best_model_module_class = model_info["class"]
            best_run_id = run.info.run_id


# Register best model
if best_model_instance:
    print(f"\nBest model overall: {best_model_name} with F1-score {best_model_score:.4f} and Inference Time {best_inference_time:.4f}s")
    model_uri = f"runs:/{best_run_id}/model"

    BEST_MODEL_REGISTRY_NAME = "iris_best_model"

    result = mlflow.register_model(model_uri=model_uri, name=BEST_MODEL_REGISTRY_NAME)

    # Promote to Production 
    client = MlflowClient()
    client.transition_model_version_stage(
        name=BEST_MODEL_REGISTRY_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Model '{BEST_MODEL_REGISTRY_NAME}' version {result.version} promoted to Production")
else:
    print("No valid model found.")