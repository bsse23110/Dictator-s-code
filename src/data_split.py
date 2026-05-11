# src/data_split.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import config as cfg

def load_and_split():
    print("\n[Phase 1.2] Creating reproducible train/val/test splits...")
    df = pd.read_csv(cfg.PREPROCESSED_PATH)
    X = df.drop(columns=[cfg.TARGET_COL])
    y = df[cfg.TARGET_COL]

    # 1. Hold-out TEST (20%) → NEVER touched during tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )

    # 2. VALIDATION (20% of train) → Used for early stopping & Optuna pruning
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=cfg.VAL_RATIO, random_state=cfg.RANDOM_STATE, stratify=y_train
    )

    # Save indices for 100% reproducibility across runs
    indices = {"train": X_train_final.index, "val": X_val.index, "test": X_test.index}
    split_path = os.path.join(cfg.SPLITS_DIR, "indices.pkl")
    with open(split_path, "wb") as f:
        pickle.dump(indices, f)

    print(f"  ✅ Train: {X_train_final.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return (X_train_final, y_train_final), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    load_and_split()
