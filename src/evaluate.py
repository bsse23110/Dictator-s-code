# src/evaluate.py
import sys, os, joblib, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import config as cfg
from mlflow_logger import setup_experiment, MLflowRun

def calc_metrics(y_true, y_pred, y_proba):
    return {
        "AUC-ROC": round(roc_auc_score(y_true, y_proba), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4)
    }

def main():
    print("\n[Phase 4] Final Unseen Test Evaluation & Handoff Generation...")
    setup_experiment()

    # 1️⃣ Load strict hold-out test set (NEVER seen by Optuna/CV)
    df = pd.read_csv(cfg.PREPROCESSED_PATH)
    with open(cfg.SPLITS_DIR / "indices.pkl", "rb") as f:
        idx = pickle.load(f)

    X_test = df.loc[idx["test"]].drop(columns=[cfg.TARGET_COL])
    y_test = df.loc[idx["test"], cfg.TARGET_COL]
    feature_names = list(X_test.columns)

    # 2️⃣ Load serialized models
    models = {
        "Random Forest (Tuned)": joblib.load(cfg.PROJECT_ROOT / "models" / "best_rf.pkl"),
        "XGBoost (Tuned)": joblib.load(cfg.PROJECT_ROOT / "models" / "best_xgb_latest.pkl")
    }

    # 3️⃣ Evaluate on unseen test data
    results = {}
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        results[name] = calc_metrics(y_test, y_pred, y_proba)

    df_res = pd.DataFrame(results).T
    df_res.index.name = "Model"
    print("\n📊 Final Test Set Results:")
    print(df_res.to_string())

    # 4️⃣ Select production candidate by Test AUC-ROC
    best_name = df_res["AUC-ROC"].idxmax()
    print(f"\n🏆 Production Candidate: {best_name} | AUC-ROC: {df_res.loc[best_name, 'AUC-ROC']}")

    # 5️⃣ Feature Importance Chart (Top 10)
    winner_model = models[best_name]
    importances = pd.Series(winner_model.feature_importances_, index=feature_names)
    top10 = importances.sort_values(ascending=False).head(10)

    os.makedirs(cfg.PROJECT_ROOT / "results", exist_ok=True)
    plt.figure(figsize=(7, 5))
    top10.plot(kind='barh', color='#4C72B0')
    plt.xlabel('Importance')
    plt.title(f'Top 10 Predictors - {best_name}')
    plt.tight_layout()
    fig_path = cfg.PROJECT_ROOT / "results" / "feature_importance.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # 6️⃣ Save comparison table & log to MLflow
    csv_path = cfg.PROJECT_ROOT / "results" / "comparison_table.csv"
    df_res.to_csv(csv_path)

    with MLflowRun("Final_Evaluation") as run:
        run.log_metrics({f"test_{k.lower().replace('-','_')}": v for k, v in df_res.loc[best_name].items()})
        run.log_artifact(str(csv_path))
        run.log_artifact(str(fig_path))
        print(f"\n✅ Artifacts saved: {csv_path}, {fig_path}")
        print(f"📦 MLflow logged. Handoff ready for Hashim.")

    return df_res, best_name

if __name__ == "__main__":
    main()