# This script is developed to load configuration files and sample hyperparameters for model tuning.
# It includes functions to load YAML configuration files and sample hyperparameters using Optuna.

import yaml

def load_config(path: str) -> dict:
    """Loads a YAML configuration file from the given path."""
    with open(path, "r") as file:
        return yaml.safe_load(file)


def sample_hyperparameters(trial, search_space):
    params = {}
    for param_name, param_def in search_space.items():
        if param_def["type"] == "int":
            params[param_name] = trial.suggest_int(param_name, param_def["low"], param_def["high"])
        elif param_def["type"] == "float":
            params[param_name] = trial.suggest_float(param_name, param_def["low"], param_def["high"], log=param_def.get("log", False))
        elif param_def["type"] == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_def["choices"])
    return params
