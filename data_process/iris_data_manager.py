import pandas as pd
import logging
import os
import json
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json" #os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])


@singleton
class IrisDatasetManager:
    def __init__(self):
        pass

    def load_and_split_data(self):
        iris = datasets.load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target
        return train_test_split(iris_df, test_size=0.2, random_state=42)

    def save_data(self, data, path):
        data.to_csv(path, index=False)


# Main Execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting script...")

    iris_manager = IrisDatasetManager()

    logger.info("Loading and splitting Iris dataset...")
    train_set, inference_set = iris_manager.load_and_split_data()

    logger.info(f"Saving training set to {TRAIN_PATH}...")
    iris_manager.save_data(train_set, TRAIN_PATH)

    logger.info(f"Saving inference set to {INFERENCE_PATH}...")
    iris_manager.save_data(inference_set, INFERENCE_PATH)

    logger.info("Script completed successfully.")
