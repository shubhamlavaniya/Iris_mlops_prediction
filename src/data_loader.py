import os
import pandas as pd
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Function to read configuration from YAML file
def read_config(config_path: str = "src/config/model_config.yaml") -> dict:
    """Reads the YAML config file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Config loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

# Load data based on config

def load_data(config: dict, processed: bool = False) -> pd.DataFrame:
    """
    Loads data based on config.
    If processed is True, loads the processed file, else loads the raw file.
    """
    try:
        if processed:
            data_path = config["data"]["processed"]
            logger.info("Loading processed data...")
        else:
            data_path = config["data"]["raw"]
            logger.info("Loading raw data...")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found at {data_path}")

        df = pd.read_csv(data_path)
        logger.info(f"Data loaded. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# Optional test: run directly to check if working

if __name__ == "__main__":
    config = read_config()
    df = load_data(config, processed=False)
    print(df.head())
