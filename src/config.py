# src/config.py
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_PATH = DATA_DIR / "preprocessed_data.csv"
SPLITS_DIR = DATA_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Data & Splitting
TARGET_COL = "no_show"
RANDOM_STATE = 42
TEST_SIZE = 0.20      # Strict hold-out
VAL_RATIO = 0.20      # From training set for early stopping / Optuna
CV_FOLDS = 5

# Optuna
OPTUNA_TRIALS_RF = 80
OPTUNA_TRIALS_XGB = 100

# Tracking
MLFLOW_EXP_NAME = "NoShow_Predictor"
