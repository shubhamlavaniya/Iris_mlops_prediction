# This script is developed to build and return machine learning models based on the provided model name and parameters.
# It includes functions to create instances of various models like Logistic Regression, Random Forest, Decision Tree, and SVC.

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

def get_model(model_name, model_params):
    """
    Given the model name and parameters, returns an instance of the model.
    """
    logger.info(f"Creating model: {model_name} with parameters: {model_params}")
    
    if model_name == "logistic_regression":
        model = LogisticRegression(**model_params)
    elif model_name == "random_forest":
        model = RandomForestClassifier(**model_params)
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(**model_params)
    elif model_name == "svc":
        model = SVC(**model_params)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    return model
