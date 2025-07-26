# src/preprocess.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_iris():
    logger.info("Loading Iris dataset")
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])

    # Clean column names
    df.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in df.columns]

    # Feature Engineering
    logger.info("Performing feature engineering...")
    df['Petal_ratio'] = df['petal_length'] / df['petal_width']
    df['Sepal_area'] = df['sepal_length'] * df['sepal_width']
    df['Petal_area'] = df['petal_length'] * df['petal_width']

    # Drop target_name before scaling (keep only numeric features)
    features = df.drop(columns=['target', 'target_name'])

    # Scaling
    logger.info("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

    # Add target back
    df_scaled['target'] = df['target']

    # Save to processed folder
    output_path = "data/processed"
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, "iris_cleaned.csv")
    
    logger.info(f"Saving cleaned and scaled data to {file_path}")
    df_scaled.to_csv(file_path, index=False)

if __name__ == "__main__":
    load_and_preprocess_iris()
