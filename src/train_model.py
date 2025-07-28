import pandas as pd
import optuna
import importlib
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from utils import load_config

# Load config
config = load_config("src/config/model_config.yaml")
models_config = config["models"]
target_column = config["model"]["target_column"]

# Load dataset
df = pd.read_csv(config["data"]["processed"])
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data: train -> train + val, test
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"])
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

# Track best model across all candidates
best_model_name = None
best_model_score = 0
best_model_instance = None
best_model_params = {}
best_model_module_class = ""



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
        acc = accuracy_score(y_val, preds)
        return acc

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    # Best from this model
    model_best_params = study.best_params
    model_best_score = study.best_value
    print(f"{model_name} best accuracy: {model_best_score:.4f} with params: {model_best_params}")

    # Train final model with best params
    module_name, class_name = model_info["class"].rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_name), class_name)
    final_model = ModelClass(**model_best_params)


    # Log to MLflow
    with mlflow.start_run(run_name=f"{model_name}_run") as run:
        final_model.fit(X_train_full, y_train_full)
        preds = final_model.predict(X_test)
        test_acc = accuracy_score(y_test, preds)

        mlflow.log_params(model_best_params)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(final_model, "model")

        print(f"Logged {model_name} to MLflow with test accuracy: {test_acc:.4f}")

        # Track best model globally
        if test_acc > best_model_score:
            best_model_score = test_acc
            best_model_name = model_name
            best_model_instance = final_model
            best_model_params = model_best_params
            best_model_module_class = model_info["class"]
            best_run_id = run.info.run_id

# Register best model
if best_model_instance:
    print(f"\nBest model overall: {best_model_name} with accuracy {best_model_score:.4f}")
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
