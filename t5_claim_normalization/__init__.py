import os
import requests

DATA_DIR = os.path.abspath("./data")

TRAIN_DATA_URL = "https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/raw/main/task2/data/train/train-eng.csv"
DEV_DATA_URL = "https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/raw/main/task2/data/dev/dev-eng.csv"
TEST_DATA_URL = "https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/raw/main/task2/data/test/test-eng.csv"

TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train-eng.csv")
DEV_CSV_PATH = os.path.join(DATA_DIR, "dev-eng.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test-eng.csv")

def download_data(file_path, url):
    if not os.path.exists(file_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Dataset saved to {file_path}")

def ensure_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_data(TRAIN_CSV_PATH, TRAIN_DATA_URL)
    download_data(DEV_CSV_PATH, DEV_DATA_URL)
    download_data(TEST_CSV_PATH, TEST_DATA_URL)

ensure_datasets()

from .run_training import main as train
from .evaluate_meteor import main as evaluate
from .model_test import main as test_model
