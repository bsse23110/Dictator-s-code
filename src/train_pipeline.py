"""
train_pipeline.py
-----------------
End-to-end pipeline runner: preprocessing → splits → tuning → evaluation.

Usage:
    python src/train_pipeline.py [path/to/KaggleV2-May-2016.csv]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from preprocessing      import load_and_clean
from feature_engineering import engineer_features
from data_split          import load_and_split
from leakage_check       import detect_leakage
import config as cfg
import pandas as pd


def main(data_path: str = "data/KaggleV2-May-2016.csv"):
    print("\n" + "=" * 60)
    print("  Hospital No-Show Predictor — Full Pipeline")
    print("=" * 60)

    # Phase 1: Load, clean, engineer features --------------------------
    print("\n[Phase 1] Loading + cleaning + feature engineering...")
    df_clean = load_and_clean(data_path)
    feats, _ = engineer_features(df_clean, is_train=True)
    feats.to_csv(cfg.PREPROCESSED_PATH, index=False)
    print(f"  Saved: {cfg.PREPROCESSED_PATH}")

    # Phase 2: Split + leakage check -----------------------------------
    print("\n[Phase 2] Creating splits and running leakage check...")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_and_split()
    safe = detect_leakage(feats.drop(columns=[cfg.TARGET_COL]), feats[cfg.TARGET_COL])
    print(f"  Safe features: {len(safe)}")

    print("\n" + "=" * 60)
    print("  Pipeline setup complete.")
    print("  Next: run src/tune_rf.py and src/tune_xgb.py for tuning.")
    print("  After tuning: run src/evaluate.py for final test results.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/KaggleV2-May-2016.csv"
    main(path)
