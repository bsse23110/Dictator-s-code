# src/mlflow_logger.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg
from optuna.integration import MLflowCallback as OptunaMLflow

def setup_experiment():
    mlflow.set_tracking_uri(os.path.join(cfg.PROJECT_ROOT, "mlruns"))
    mlflow.set_experiment(cfg.MLFLOW_EXP_NAME)
    exp = mlflow.get_experiment_by_name(cfg.MLFLOW_EXP_NAME)
    print(f"\n[MLflow] Experiment '{cfg.MLFLOW_EXP_NAME}' ready | ID: {exp.experiment_id}")
    return exp.experiment_id

def get_optuna_callback(experiment_id):
    """Returns Optuna callback that auto-logs params & metrics per trial."""
    return OptunaMLflow(tracking_uri=mlflow.get_tracking_uri(), experiment_id=experiment_id)

class MLflowRun:
    def __init__(self, model_name, run_name=None):
        self.model_name = model_name
        self.run_name = run_name or model_name
        self.run_id = None

    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self.run.info.run_id
        print(f"[MLflow] Run started: {self.run_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
        print(f"[MLflow] Run '{self.run_name}' finalized.")

    def log_params(self, params):
        mlflow.log_params(params)

    def log_artifact(self, path):
        mlflow.log_artifact(path)
        print(f"[MLflow] Logged artifact: {path}")

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    def log_comparison_table(self, metrics_df, filename="comparison_table.csv"):
        path = os.path.join(cfg.PROJECT_ROOT, filename)
        metrics_df.to_csv(path, index=True)
        mlflow.log_artifact(path)
        print(f"[MLflow] Logged comparison table -> {filename}")

    def log_feature_importance(self, model, feature_names, top_n=10, save_path="feature_importance.png"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top = importances.sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(6, 4))
        top.plot(kind='barh', color='#4C72B0')
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Predictors - {self.model_name}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        mlflow.log_artifact(save_path)
        print(f"[MLflow] Logged feature importance chart.")
        os.remove(save_path)

    def register_model(self, model_uri="model"):
        mlflow.sklearn.log_model(model_uri, artifact_path="model")
        print(f"[MLflow] Model logged at artifact path: {model_uri}")
