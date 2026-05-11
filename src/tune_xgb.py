# src/tune_xgb.py
import sys, os, pickle, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import config as cfg
from mlflow_logger import setup_experiment, MLflowRun
from cv_utils import run_cv
from optuna.integration.mlflow import MLflowCallback

def _build_objective(X_tr, y_tr):
    base_scale = (y_tr == 0).sum() / (y_tr == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400, step=50),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 8),
            "scale_pos_weight":  base_scale,
            "random_state":      cfg.RANDOM_STATE,
            "verbosity":         0,
            "tree_method":       "hist",   # ✅ 3-5x faster on tabular data
            "eval_metric":       "auc",    # ✅ Aligns early stopping with Optuna objective
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.RANDOM_STATE)
        fold_aucs = []

        for tr_idx, va_idx in cv.split(X_tr, y_tr):
            X_f_tr, X_f_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            y_f_tr, y_f_va = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

            # ✅ XGB 3.x: callbacks go in constructor, not fit()
            es_cb = xgb.callback.EarlyStopping(rounds=30)
            model = XGBClassifier(**params, callbacks=[es_cb])
            model.fit(
                X_f_tr, y_f_tr,
                eval_set=[(X_f_va, y_f_va)],
                verbose=False
            )
            fold_aucs.append(roc_auc_score(y_f_va, model.predict_proba(X_f_va)[:, 1]))

        return np.mean(fold_aucs)
    return objective


def main():
    print("\n[Phase 3.2] Tuning XGBoost with Optuna + 3-Fold CV + Aligned Early Stopping...")

    # 1️⃣ Load data ONCE
    df = pd.read_csv(cfg.PREPROCESSED_PATH)
    with open(cfg.SPLITS_DIR / "indices.pkl", "rb") as f:
        idx = pickle.load(f)

    X_tr = df.loc[idx["train"]].drop(columns=[cfg.TARGET_COL])
    y_tr = df.loc[idx["train"], cfg.TARGET_COL]
    X_va = df.loc[idx["val"]].drop(columns=[cfg.TARGET_COL])
    y_va = df.loc[idx["val"], cfg.TARGET_COL]

    # 2️⃣ Setup MLflow
    setup_experiment()

    # 3️⃣ Optimize
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_STATE)
    )
    mlflow_cb = MLflowCallback()
    study.optimize(
        _build_objective(X_tr, y_tr),
        n_trials=cfg.OPTUNA_TRIALS_XGB,
        callbacks=[mlflow_cb],
        show_progress_bar=True
    )

    print(f"\n✅ Best XGB CV AUC-ROC : {study.best_value:.4f}")
    print(f"   Params              : {study.best_params}")

    # 4️⃣ Held-out Val Check (unseen by Optuna)
    best_params = study.best_params.copy()
    best_params.update({"random_state": cfg.RANDOM_STATE, "verbosity": 0, "tree_method": "hist", "eval_metric": "auc"})

    probe_model = XGBClassifier(**best_params)
    probe_model.fit(X_tr, y_tr)
    holdout_auc = roc_auc_score(y_va, probe_model.predict_proba(X_va)[:, 1])
    print(f"\n📊 Held-out Val AUC-ROC (unseen): {holdout_auc:.4f}")

    gap = study.best_value - holdout_auc
    if gap > 0.05:
        print(f"  ⚠️  CV→Holdout gap: {gap:.4f}. Tighten reg_alpha/reg_lambda or cap max_depth.")
    else:
        print(f"  ✅ CV→Holdout gap healthy: {gap:.4f}")

    # 5️⃣ Retrain on full train+val
    X_full = pd.concat([X_tr, X_va])
    y_full = pd.concat([y_tr, y_va])
    best_params["scale_pos_weight"] = (y_full == 0).sum() / (y_full == 1).sum()

    best_model = XGBClassifier(**best_params)
    best_model.fit(X_full, y_full)

    # 6️⃣ Final CV + MLflow logging
    with MLflowRun("XGB_Optuna_Final") as run:
        run.log_params(best_params)
        run.log_metrics({
            "best_cv_auc":    study.best_value,
            "holdout_auc":    holdout_auc,
            "cv_holdout_gap": gap,
        })
        cv_model = XGBClassifier(**best_params)
        _, cv_summary, _ = run_cv(cv_model, X_full, y_full)
        run.log_metrics({f"cv_{k}": v for k, v in cv_summary.items()})

        final_gap = cv_summary.get("AUC-ROC_gap", 0)
        if final_gap > 0.05:
            print(f"  ⚠️  Final CV gap: {final_gap:.4f}. Increase reg_alpha/reg_lambda.")
        else:
            print(f"  ✅ Final generalization healthy. Gap: {final_gap:.4f}")

    # 7️⃣ Serialize
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_xgb_latest.pkl")
    print("✅ Saved to models/best_xgb_latest.pkl")

    return best_model, best_params

if __name__ == "__main__":
    main()