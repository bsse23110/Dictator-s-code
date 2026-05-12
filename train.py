"""
train.py
--------
Loads the dataset, runs the preprocessing + feature engineering pipeline,
splits the data, then trains and evaluates three baseline classifiers:

  1. Logistic Regression
  2. Random Forest
  3. XGBoost

Outputs a clean results table with AUC-ROC, F1, Precision, and Recall
for each model.

Usage
-----
    python train.py [path/to/KaggleV2-May-2016.csv]

If no path is given, defaults to data/KaggleV2-May-2016.csv
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# XGBoost with GPU support via tree_method='gpu_hist'
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing      import load_and_clean
from feature_engineering import engineer_features


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE  = 42
TEST_SIZE     = 0.20          # 80 / 20 split
TARGET_COL    = "no_show"

# GPU flag — True to enable CUDA for XGBoost (requires xgboost built with GPU)
USE_GPU = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return a dict of evaluation metrics for one model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model":     name,
        "AUC-ROC":   round(roc_auc_score(y_test, y_proba),    4),
        "F1":        round(f1_score(y_test, y_pred),           4),
        "Precision": round(precision_score(y_test, y_pred),    4),
        "Recall":    round(recall_score(y_test, y_pred),       4),
    }


def print_evaluation(name: str, metrics: dict, y_test, y_pred):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  AUC-ROC  : {metrics['AUC-ROC']}")
    print(f"  F1       : {metrics['F1']}")
    print(f"  Precision: {metrics['Precision']}")
    print(f"  Recall   : {metrics['Recall']}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Showed Up", "No-Show"]))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(data_path: str = "data/KaggleV2-May-2016.csv"):
    print("\n" + "="*60)
    print("  Hospital No-Show Predictor — Baseline Training")
    print("="*60)

    # 1. Load & clean ---------------------------------------------------
    print("\n[1/5] Loading and cleaning data...")
    df_clean = load_and_clean(data_path)

    # 2. Split BEFORE feature engineering (avoids target leakage) -------
    print("\n[2/5] Splitting data (BEFORE feature engineering)...")
    df_train, df_test = train_test_split(
        df_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df_clean[TARGET_COL]
    )
    print(f"  Train size : {df_train.shape[0]:,}  ({df_train[TARGET_COL].mean():.2%} no-show)")
    print(f"  Test  size : {df_test.shape[0]:,}  ({df_test[TARGET_COL].mean():.2%} no-show)")

    # 3. Feature engineering — fit on train, transform on test ----------
    print("\n[3/5] Engineering features...")
    feats_train, encoder = engineer_features(df_train, is_train=True)
    feats_test, _        = engineer_features(df_test, neighbourhood_encoder=encoder, is_train=False)
    print(f"  Train features shape: {feats_train.shape}")
    print(f"  Test  features shape: {feats_test.shape}")

    # Also engineer full dataset for saving (used by src/ tuning pipeline)
    # Note: neighbourhood encoder is fitted on full data here — the src/
    # pipeline re-splits and uses leakage_check.py to validate.
    feats_full, _ = engineer_features(df_clean, is_train=True)

    os.makedirs("data", exist_ok=True)
    out_path = "data/preprocessed_data.csv"
    feats_full.to_csv(out_path, index=False)
    print(f"[train] Full preprocessed data saved to: {out_path}")

    # Separate X and y -----------------------------------------------
    X_train = feats_train.drop(columns=[TARGET_COL])
    y_train = feats_train[TARGET_COL]
    X_test  = feats_test.drop(columns=[TARGET_COL])
    y_test  = feats_test[TARGET_COL]
    print(f"  Features   : {X_train.shape[1]}")

    # 4. Scale (needed for Logistic Regression) ------------------------
    print("\n[4/5] Scaling features...")
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 5. Train & evaluate models ---------------------------------------
    print("\n[5/5] Training models...\n")
    results = []

    # ---- Logistic Regression ----------------------------------------
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr.fit(X_train_sc, y_train)
    m = evaluate("Logistic Regression", lr, X_test_sc, y_test)
    results.append(m)
    print_evaluation("Logistic Regression", m, y_test, lr.predict(X_test_sc))

    # ---- Random Forest ----------------------------------------------
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    m = evaluate("Random Forest", rf, X_test, y_test)
    results.append(m)
    print_evaluation("Random Forest", m, y_test, rf.predict(X_test))

    # ---- XGBoost ----------------------------------------------------
    print("Training XGBoost...")
    neg, pos     = (y_train == 0).sum(), (y_train == 1).sum()
    scale_weight = neg / pos

    # Use GPU if available, fall back to CPU
    if USE_GPU:
        try:
            import xgboost as xgb
            # Quick check: try creating a tiny DMatrix on GPU
            _ = xgb.DMatrix(np.array([[0.0]]), label=np.array([0]))
            _ = xgb.train({"tree_method": "gpu_hist"}, _)
            tree_method = "gpu_hist"
            device = "cuda"
            print("  [XGBoost] GPU (CUDA) enabled.")
        except Exception:
            tree_method = "hist"
            device = "cpu"
            print("  [XGBoost] GPU not available — falling back to CPU.")
    else:
        tree_method = "hist"
        device = "cpu"
        print("  [XGBoost] GPU disabled (USE_GPU = False). Running on CPU.")

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
        tree_method=tree_method,
        device=device,
    )
    xgb_model.fit(X_train, y_train)
    m = evaluate("XGBoost", xgb_model, X_test, y_test)
    results.append(m)
    print_evaluation("XGBoost", m, y_test, xgb_model.predict(X_test))

    # 6. Summary table -------------------------------------------------
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    summary = pd.DataFrame(results).set_index("model")
    print(summary.to_string())
    best = summary["AUC-ROC"].idxmax()
    print(f"\n  Best model by AUC-ROC: {best} ({summary.loc[best, 'AUC-ROC']})")
    print("="*60 + "\n")

    # 7. Feature importance (Random Forest) ----------------------------
    feat_names = X_train.columns.tolist()
    importances = pd.Series(rf.feature_importances_, index=feat_names)
    print("Top 10 features (Random Forest importance):")
    print(importances.sort_values(ascending=False).head(10).to_string())
    print()

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/KaggleV2-May-2016.csv"
    main(path)
