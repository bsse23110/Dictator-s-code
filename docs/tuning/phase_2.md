# 📊 Phase 2: CV Scoring & MLflow Tracking Backbone

## 🎯 Objective
Build a reproducible, leak-proof evaluation engine and centralized experiment tracker. This phase enables reliable hyperparameter tuning (Phase 3), automatic overfitting detection, and hands-off MLflow logging.

---

## 📁 Module Overview
| File | Purpose | Key Outputs |
|------|---------|-------------|
| `src/cv_utils.py` | Stratified 5-fold CV with metric aggregation & gap monitoring | Fold-level DataFrame, summary dict, best-fold model |
| `src/mlflow_logger.py` | MLflow experiment setup, run lifecycle, Optuna callback, artifact logging | `mlruns/` directory, comparison CSVs, feature importance plots |

---

## ⚙️ `cv_utils.py` Logic
- **Stratified Splitting**: `StratifiedKFold(n_splits=5, shuffle=True)` preserves the ~80/20 `no_show` ratio in every fold.
- **Metric Computation**: Calculates `AUC-ROC`, `F1`, `Precision`, `Recall` on both train & validation subsets per fold.
- **Overfitting Guard**: Computes `train_score - val_score`. If gap `> 0.08`, prints a warning (triggers Phase 3 regularization).
- **Model-Agnostic**: Accepts any scikit-learn compatible estimator. Caller handles scaling (LR) vs raw inputs (trees).
- **Return Format**: 
  ```python
  fold_df, summary_dict, best_model = run_cv(model, X, y)
  ```

---

## 📡 `mlflow_logger.py` Logic
- **Experiment Setup**: `setup_experiment()` initializes local `./mlruns` tracking URI & sets `NoShow_Predictor` namespace.
- **Optuna Integration**: `get_optuna_callback()` returns `optuna.integration.MLflowCallback` → auto-logs `trial.params`, `trial.values`, `trial.number`.
- **Context Manager**: `MLflowRun` guarantees clean run closure:
  ```python
  with MLflowRun("RandomForest_v1") as run:
      run.log_params({"n_estimators": 200})
      run.log_metrics({"AUC-ROC": 0.76})
  ```
- **Artifact Logging**: 
  - `log_comparison_table()` → saves & tracks model comparison CSV
  - `log_feature_importance()` → generates & tracks top-N predictor bar chart
  - `register_model()` → serializes model via `mlflow.sklearn.log_model()`

---

## 🧪 Quick Validation
```bash
# Test CV backbone
python -c "import sys; sys.path.insert(0,'src'); from cv_utils import run_cv; import pandas as pd; from sklearn.ensemble import RandomForestClassifier; X,y = pd.read_csv('data/preprocessed_data.csv').drop('no_show',axis=1), pd.read_csv('data/preprocessed_data.csv')['no_show']; _,s,_ = run_cv(RandomForestClassifier(n_estimators=50,max_depth=5,random_state=42),X,y); print(s)"

# Test MLflow setup
python -c "import sys; sys.path.insert(0,'src'); from mlflow_logger import setup_experiment; print('Exp ID:', setup_experiment())"
```

---

## 🔗 Handoff & Best Practices
- **Input**: Consumes `X, y` matrices. Caller must pass only `train` indices from `data/splits/indices.pkl` to prevent leakage.
- **Scaling**: Deferred to model-specific wrappers. Trees use raw; LR uses `StandardScaler`.
- **Tracking**: All runs live in `./mlruns`. View via `mlflow ui` on `http://localhost:5000`.
- **Next Phase**: `tune_rf.py` & `tune_xgb.py` will import both modules to wire Optuna search spaces, early stopping, and auto-logging.

> ✅ Phase 2 delivers deterministic CV scoring + zero-config MLflow tracking.
