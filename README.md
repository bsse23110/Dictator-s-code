
## Files

| File | What it does |
|------|-------------|
| `src/preprocessing.py` | Loads raw CSV, cleans column names, fixes data issues, encodes target |
| `src/feature_engineering.py` | Builds all engineered features from the clean DataFrame |
| `src/train.py` | Runs the full pipeline and trains LR, Random Forest, XGBoost |
| `data/KaggleV2-May-2016.csv` | Raw dataset (not committed — keep locally) |

---

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost
python src/train.py data/KaggleV2-May-2016.csv
```

---

## Features Created

| Feature | Description |
|---------|-------------|
| `lead_time` | Days between booking and appointment — **strongest predictor (55.9% importance)** |
| `is_weekend` | 1 if appointment is on Saturday/Sunday |
| `appointment_weekday` | Day of week (0=Mon … 6=Sun) |
| `scheduled_month` | Month the appointment was booked |
| `appointment_month` | Month the appointment takes place |
| `age_group` | Age binned: 0=child, 1=teen, 2=adult, 3=senior |
| `previous_no_show_rate` | Fraction of past appointments this patient missed (0 for new patients) |
| `previous_appointments` | How many prior records this patient has |
| `neighbourhood_no_show_rate` | Smoothed target encoding of neighbourhood — fitted on train only |

---

## Cleaning Decisions

- **Negative age** (1 record): replaced with median age (37)
- **Handicap levels 2–4**: capped to 1 (binary flag) — too few records for stable estimates
- **Gender**: encoded as 1=Female, 0=Male
- **Target label**: `No-show = Yes` → 1 (missed), `No` → 0 (attended)
- **Dropped**: `appointment_id`, `patient_id`, raw datetime columns, raw neighbourhood string

---

## Baseline Results

| Model | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.6809 | 0.4128 | 0.3148 | 0.5997 |
| Random Forest | 0.7466 | 0.4566 | 0.3173 | 0.8136 |
| **XGBoost** | **0.7492** | **0.4578** | **0.3230** | **0.7858** |

---


- Use `X_train` / `X_test` (unscaled) for Random Forest and XGBoost
- Use `X_train_sc` / `X_test_sc` (scaled) for Logistic Regression
- Keep `class_weight='balanced'` / `scale_pos_weight` — dataset is 80/20 imbalanced
- Use AUC-ROC as the Optuna objective
- `NeighbourhoodEncoder` is fitted on train only — don't refit on test
