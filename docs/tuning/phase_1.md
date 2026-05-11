# 📊 Phase 1: Data Integrity & Reproducible Splitting

## 🎯 Objective
Establish a leak-proof, deterministic foundation for modeling by:
1. Centralizing all paths, seeds, and tuning constants
2. Creating strictly isolated `train`, `validation`, and `test` sets with stratified sampling
3. Auditing the preprocessed dataset for target leakage, temporal contamination, and structural anomalies
4. Persisting split indices to guarantee 100% reproducibility across Optuna trials, CV folds, and final evaluation

---

## 📁 Module Structure
| File | Role | Dependencies |
|------|------|--------------|
| `src/config.py` | Centralized constants, paths, hyperparameter limits | `pathlib`, `os` |
| `src/data_split.py` | Deterministic stratified splitting + index persistence | `pandas`, `sklearn`, `pickle`, `config` |
| `src/leakage_check.py` | Structural & statistical leakage detection | `pandas`, `config` |

---

## ⚙️ 1. Configuration (`config.py`)
All magic numbers, paths, and experiment metadata are centralized here. No hardcoded values leak into downstream modules.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `RANDOM_STATE` | `42` | Deterministic splits, shuffling, and model initialization |
| `TEST_SIZE` | `0.20` | Strict hold-out set (never seen during tuning) |
| `VAL_RATIO` | `0.20` | Validation carve-out from training pool (early stopping + Optuna) |
| `CV_FOLDS` | `5` | Stratified cross-validation folds |
| `OPTUNA_TRIALS_RF` / `XGB` | `80` / `100` | Search budget per model |
| `TARGET_COL` | `"no_show"` | Binary target (`0`=Show, `1`=No-Show) |
| `MLFLOW_EXP_NAME` | `"NoShow_Predictor"` | Experiment tracking namespace |

---

## 🔪 2. Data Splitting (`data_split.py`)
Creates a **three-way stratified split** and saves exact row indices to disk.

### 🔹 Algorithmic Flow
1. Load `data/preprocessed_data.csv` → `X`, `y`
2. **Split 1**: `train_test_split(..., test_size=0.20, stratify=y)` → isolates `TEST` (22,106 rows)
3. **Split 2**: `train_test_split(..., test_size=0.20, stratify=y_train)` → carves `VAL` (17,685 rows) from temp-train
4. Persist indices to `data/splits/indices.pkl`

### 🔹 Resulting Distribution
| Set | % of Total | Rows | Target Rate |
|-----|------------|------|-------------|
| `train` | ~64% | 70,736 | ~20.1% |
| `val` | ~16% | 17,685 | ~20.1% |
| `test` | ~20% | 22,106 | ~20.1% |

*Stratification preserves the 80/20 class imbalance across all splits.*

---

## 🔍 3. Leakage Detection (`leakage_check.py`)
Runs structural hygiene checks on the **full preprocessed feature matrix** before any modeling occurs.

| Check | Threshold | Action |
|-------|-----------|--------|
| Target Correlation | `|r| > 0.90` | Flag & drop (direct target leakage) |
| Zero Variance | `nunique() ≤ 1` | Drop (breaks scalers, adds no signal) |
| Temporal Keywords | `future_`, `next_`, `post_`, `remaining_` | Flag & drop (forward-looking aggregates) |
| Target Encoding Overfit | `neighbourhood_no_show_rate` `|r| > 0.98` | Warn (likely fitted on full data, not train-only) |

Returns `safe_features` list used by downstream CV/tuning modules.

---

## 🔄 Execution Workflow
Run from project root:
```bash
python src/data_split.py
python src/leakage_check.py
```
**Order matters**: Splits must be created before leakage checks if you later want fold-aware validation, but here we audit the full preprocessed schema first. Both are safe to run independently.

---

## ✅ Expected Output
```
[Phase 1.2] Creating reproducible train/val/test splits...
  ✅ Train: 70,736 | Val: 17,685 | Test: 22,106

[Phase 1.3] Running leakage & feature quality checks...
  ✅ No obvious leakage detected.
  📦 Features retained: 17/17
```
Artifacts created:
- `data/splits/indices.pkl` → `{"train": [...], "val": [...], "test": [...]}`
- Console leakage report → safe feature whitelist

---

## 📝 Technical Notes & Best Practices
| Design Choice | Rationale |
|---------------|-----------|
| **Leakage check on full preprocessed data** | Detects *design-level* leaks (bad encodings, future features). *Procedural* leaks are prevented later by strict index masking. |
| **Index persistence (.pkl)** | Guarantees identical splits across Optuna trials, CI runs, and team members. Eliminates `random_state` drift. |
| **Val = 20% of train (not full data)** | Preserves maximal training signal while leaving enough rows for reliable early stopping & Optuna trial scoring. |
| **No scaling in Phase 1** | Scaling is deferred to `cv_utils.py` to prevent leakage from validation/test distributions into training folds. |

---