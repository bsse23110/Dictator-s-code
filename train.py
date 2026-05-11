"""
train.py
--------
Loads the dataset, runs the preprocessing + feature engineering pipeline,
splits the data, then trains and evaluates three baseline classifiers:

  1. Logistic Regression
  2. Random Forest
  3. XGBoost

Outputs a clean results table with AUC-ROC, F1, Precision, and Recall
for each model, ready for Rafey to pick up and tune with Optuna + MLflow.

Usage
-----
    python src/train.py [path/to/KaggleV2-May-2016.csv]

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

# Columns that should NOT be scaled (binary flags, ordinals, encoded rates
# are fine to pass through a scaler, but keeping it explicit helps clarity)
# Actually we scale everything for LR; tree models don't need it but it
# doesn't hurt either, so we use the same matrix for all three.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return a dict of evaluation metrics for one model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     name,
        "AUC-ROC":   round(roc_auc_score(y_test, y_proba),    4),
        "F1":        round(f1_score(y_test, y_pred),           4),
        "Precision": round(precision_score(y_test, y_pred),    4),
        "Recall":    round(recall_score(y_test, y_pred),       4),
    }
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  AUC-ROC  : {metrics['AUC-ROC']}")
    print(f"  F1       : {metrics['F1']}")
    print(f"  Precision: {metrics['Precision']}")
    print(f"  Recall   : {metrics['Recall']}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Showed Up", "No-Show"]))
    return metrics


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

    # 2. Feature engineering -------------------------------------------
    print("\n[2/5] Engineering features...")
    feats, _ = engineer_features(df_clean, is_train=True)

    # 3. Train / test split --------------------------------------------
    print("\n[3/5] Splitting data...")
    X = feats.drop(columns=[TARGET_COL])
    y = feats[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train size : {X_train.shape[0]:,}  ({y_train.mean():.2%} no-show)")
    print(f"  Test  size : {X_test.shape[0]:,}  ({y_test.mean():.2%} no-show)")
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
        class_weight="balanced",   # handle class imbalance
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr.fit(X_train_sc, y_train)
    results.append(evaluate("Logistic Regression", lr, X_test_sc, y_test))

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
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    # ---- XGBoost ----------------------------------------------------
    print("Training XGBoost...")
    # scale_pos_weight handles imbalance for XGBoost
    neg, pos     = (y_train == 0).sum(), (y_train == 1).sum()
    scale_weight = neg / pos

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate("XGBoost", xgb, X_test, y_test))

    # 6. Summary table -------------------------------------------------
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    summary = pd.DataFrame(results).set_index("model")
    print(summary.to_string())
    best = summary["AUC-ROC"].idxmax()
    print(f"\n  Best model by AUC-ROC: {best} ({summary.loc[best, 'AUC-ROC']})")
    print("="*60 + "\n")

    # 7. Feature importance (Random Forest) for Rafey ------------------
    feat_names = X.columns.tolist()
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
