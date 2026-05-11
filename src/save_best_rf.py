# save_best_rf.py  (run AFTER tuning completes)
import sys, os, joblib, pandas as pd, pickle
sys.path.insert(0, 'src'); import config as cfg
from sklearn.ensemble import RandomForestClassifier

with open(cfg.SPLITS_DIR / "indices.pkl", "rb") as f: idx = pickle.load(f)
df = pd.read_csv(cfg.PREPROCESSED_PATH)
X_full = pd.concat([df.loc[idx["train"]].drop(cfg.TARGET_COL, axis=1),
                    df.loc[idx["val"]].drop(cfg.TARGET_COL, axis=1)])
y_full = pd.concat([df.loc[idx["train"], cfg.TARGET_COL],
                    df.loc[idx["val"], cfg.TARGET_COL]])

# 👇 PASTE THE BEST_PARAMS DICT PRINTED BY OPTUNA HERE
best_params = {
    "n_estimators": 550,
    "max_depth": 12,
    "min_samples_split": 26,
    "min_samples_leaf": 14,
    "max_features": "log2",
    "max_samples": 0.6470731109447447
}

best_params.update({
    "random_state": cfg.RANDOM_STATE,
    "n_jobs": -1
})

model = RandomForestClassifier(**best_params).fit(X_full, y_full)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_rf.pkl")
print("✅ Saved models/best_rf.pkl")
