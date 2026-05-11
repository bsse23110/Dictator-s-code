# src/cv_utils.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import config as cfg

METRIC_FNS = {
    "AUC-ROC":   lambda y, p: roc_auc_score(y, p),
    "F1":        lambda y, p: f1_score(y, (p > 0.5).astype(int)),
    "Precision": lambda y, p: precision_score(y, (p > 0.5).astype(int)),
    "Recall":    lambda y, p: recall_score(y, (p > 0.5).astype(int))
}

def run_cv(model, X, y, gap_threshold=0.08, return_model=False):
    print(f"\n[CV] Running {cfg.CV_FOLDS}-fold stratified CV on {len(y):,} rows...")
    X, y = pd.DataFrame(X), pd.Series(y, name=cfg.TARGET_COL)
    kfold = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    
    fold_records, train_agg, val_agg = [], {}, {}
    for name in METRIC_FNS:
        train_agg[name], val_agg[name] = [], []

    best_fold_auc, best_model = -1.0, None

    for fold, (tr_idx, va_idx) in enumerate(kfold.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model.fit(X_tr, y_tr)
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_va = model.predict_proba(X_va)[:, 1]

        record = {"fold": fold}
        for name, fn in METRIC_FNS.items():
            val_agg[name].append(fn(y_va, p_va))
            train_agg[name].append(fn(y_tr, p_tr))
            record[f"val_{name}"] = round(val_agg[name][-1], 4)
            record[f"train_{name}"] = round(train_agg[name][-1], 4)

        fold_records.append(record)
        if val_agg["AUC-ROC"][-1] > best_fold_auc:
            best_fold_auc = val_agg["AUC-ROC"][-1]
            best_model = model
        print(f"  Fold {fold}: AUC-ROC {record['val_AUC-ROC']} | Gap: {record['train_AUC-ROC'] - record['val_AUC-ROC']:.4f}")

    # Aggregate & Gap Check
    summary = {}
    for name in METRIC_FNS:
        summary[f"{name}_mean"] = np.mean(val_agg[name])
        summary[f"{name}_std"]  = np.std(val_agg[name])
        gap = np.mean(train_agg[name]) - np.mean(val_agg[name])
        summary[f"{name}_gap"] = gap
        if gap > gap_threshold:
            print(f"  ⚠️  {name} overfitting: train-val gap = {gap:.4f} (> {gap_threshold})")

    print(f"  ✅ CV Complete. Mean AUC-ROC: {summary['AUC-ROC_mean']:.4f} ± {summary['AUC-ROC_std']:.4f}")
    return pd.DataFrame(fold_records), summary, (best_model if return_model else None)
