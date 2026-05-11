# src/tune_rf.py
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import optuna
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import config as cfg
from mlflow_logger import setup_experiment, MLflowRun
from cv_utils import run_cv
from optuna.integration.mlflow import MLflowCallback

def _build_objective(X_tr, y_tr, X_va, y_va):
    """Closure captures data once. Zero I/O inside trials."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),          # ↓ Cap depth to prevent deep overfitting
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 40), # ↑ Force larger splits
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),   # ↑ Strong leaf regularization
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]), # Never use "None"
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),        # Row subsampling (bagging reg)
            "class_weight": "balanced",
            "random_state": cfg.RANDOM_STATE,
            "n_jobs": -1,
            "oob_score": False  # Explicit val set is more reliable for Optuna
        }
        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        y_va_proba = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, y_va_proba)
    return objective

def main():
    print("\n[Phase 3.1] Tuning Random Forest with Optuna + Strong Regularization...")
    
    # 1️⃣ Load data ONCE (fixes I/O bug)
    df = pd.read_csv(cfg.PREPROCESSED_PATH)
    with open(cfg.SPLITS_DIR / "indices.pkl", "rb") as f:
        idx = pickle.load(f)
    X_tr = df.loc[idx["train"]].drop(columns=[cfg.TARGET_COL])
    y_tr = df.loc[idx["train"], cfg.TARGET_COL]
    X_va = df.loc[idx["val"]].drop(columns=[cfg.TARGET_COL])
    y_va = df.loc[idx["val"], cfg.TARGET_COL]

    # 2️⃣ Setup MLflow properly (fixes experiment_id kwarg bug)
    setup_experiment()  # Calls mlflow.set_experiment() internally

    # 3️⃣ Optimize
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_STATE))
    mlflow_cb = MLflowCallback()  # Auto-links to active experiment
    study.optimize(_build_objective(X_tr, y_tr, X_va, y_va), 
                   n_trials=cfg.OPTUNA_TRIALS_RF, 
                   callbacks=[mlflow_cb], 
                   show_progress_bar=True)
    
    print(f"\n✅ Best RF Val AUC-ROC: {study.best_value:.4f}")
    print(f"   Params: {study.best_params}")
    
    
    X_full = pd.concat([X_tr, X_va])
    y_full = pd.concat([y_tr, y_va])
    best_params = study.best_params.copy()
    best_params.update({"random_state": cfg.RANDOM_STATE, "n_jobs": -1})
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_full, y_full)
    
    # 5️⃣ Final CV & MLflow Logging
    with MLflowRun("RF_Optuna_Final") as run:
        run.log_params(best_params)
        # Pass a fresh clone to CV to avoid state leakage
        cv_model = RandomForestClassifier(**best_params)
        _, cv_summary, _ = run_cv(cv_model, X_full, y_full)
        run.log_metrics({f"cv_{k}": v for k, v in cv_summary.items()})
        
        gap = cv_summary.get("AUC-ROC_gap", 0)
        if gap > 0.05:
            print(f"  ⚠️  Gap: {gap:.4f}. RF still memorizing noise. Reduce max_depth or increase min_samples_leaf in next run.")
        else:
            print(f"  ✅ Generalization healthy. Train-Val Gap: {gap:.4f}")
            
    return best_model, best_params

if __name__ == "__main__":
    main()