# Hospital No-Show Predictor

A production-ready machine learning system that predicts whether a patient will miss a scheduled hospital appointment. Built with a rigorous MLOps pipeline — from data leakage checks and stratified cross-validation to Optuna hyperparameter tuning, MLflow experiment tracking, and a Dockerized Flask web API with a polished frontend.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [API Endpoints](#api-endpoints)
  - [Training from Scratch](#training-from-scratch)
  - [MLflow UI](#mlflow-ui)
- [Key Design Decisions](#key-design-decisions)
- [License](#license)

---

## Problem Statement

Patient no-shows cost the healthcare industry billions annually in lost revenue and underutilized resources. In the US alone, no-show rates range from 5–30%, leading to appointment bottlenecks, increased wait times, and reduced quality of care.

This project builds a **classifier that predicts no-show risk at the time of booking**, enabling proactive interventions such as automated reminders, overbooking optimization, or same-day re-scheduling. The system is trained on **110,527 real appointment records** from Vitoria, Brazil (public Kaggle dataset).

---

## Tech Stack

| Component           | Technology                                                                 |
|---------------------|----------------------------------------------------------------------------|
| **ML Framework**    | scikit-learn, XGBoost (with GPU support)                                   |
| **Hyperparameter Tuning** | Optuna (TPESampler, 80–100 trials per model)                         |
| **Experiment Tracking** | MLflow (local `mlruns/` directory)                                       |
| **Web Framework**   | Flask 3.x                                                                  |
| **Frontend**        | Pure HTML/CSS/JS (no build step required)                                  |
| **Containerization**| Docker + docker-compose                                                    |
| **Language**        | Python 3.10                                                                |

---

## Dataset

**Source**: [Kaggle — Hospital Appointment No-Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)  
**Records**: 110,527  
**Period**: May–June 2016  
**Location**: Vitoria, Brazil  
**Raw Features**: PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age, Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handicap, SMS_received, No-show

---

## Feature Engineering

The system engineers **17 features** from raw data, grouped into five categories:

### Temporal (5)
- `lead_time` — days between scheduling and appointment
- `is_weekend` — whether the appointment falls on a weekend
- `appointment_weekday` — day of week (0–6)
- `scheduled_month`, `appointment_month`

### Demographic (2)
- `age`, `age_group` — binned into child/teen/adult/senior
- `gender`

### Medical / Program (6)
- `scholarship`, `hypertension`, `diabetes`, `alcoholism`, `handicap`, `sms_received`

### Historical Behaviour (2)
- `previous_no_show_rate` — fraction of prior appointments the patient missed
- `previous_appointments` — count of prior appointments for that patient

### Geographic (1)
- `neighbourhood_no_show_rate` — smoothed target encoding of neighbourhood, using empirical Bayes shrinkage to handle rare neighbourhoods

### Preventing Leakage

The pipeline performs a **train/val/test split before feature engineering** to prevent target leakage. A dedicated `leakage_check.py` module scans for:
- High correlation between features and target (>0.90)
- Zero-variance columns
- Temporal keyword leakage
- Target encoding overfit (>0.98 correlation)

---

## Model Pipeline

```
Raw CSV → preprocessing.py → feature_engineering.py → data_split.py
                                                           │
                      ┌────────────────────────────────────┼────────────────────┐
                      ▼                                    ▼                    ▼
              Logistic Regression               Random Forest             XGBoost
              (baseline, sklearn)           (tuned, 80 trials)      (tuned, 100 trials)
                      │                                    │                    │
                      └────────────────────────────────────┼────────────────────┘
                                                           ▼
                                              evaluate.py → comparison_table.csv
                                                           │
                                                           ▼
                                              best_xgb_latest.pkl (production)
```

### Training Pipeline (4 Phases)

1. **Data Integrity & Splitting** — Stratified 64/16/20 train/val/test split; leakage detection; reproducible indices persisted to `data/splits/indices.pkl`
2. **CV & Tracking Backbone** — Stratified 5-fold cross-validation with AUC-ROC, F1, precision, recall per fold; overfitting gap monitoring (>0.08 triggers warning); MLflow logging
3. **Hyperparameter Tuning** — Optuna with TPESampler:
   - Random Forest: 80 trials | ranges: n_estimators (100–600), max_depth (3–12), min_samples_split (5–40), min_samples_leaf (5–30), max_features (sqrt/log2), max_samples (0.6–1.0)
   - XGBoost: 100 trials | ranges: n_estimators (50–400), max_depth (3–7), learning_rate (0.02–0.15), L1/L2 regularization, column/row subsampling + early stopping (30 rounds)
4. **Final Evaluation** — Strict held-out test set (never seen by Optuna or CV); full comparison table; feature importance visualization; production model selection

### Models Trained

| Model | Tuning | Key Characteristic |
|-------|--------|--------------------|
| Logistic Regression | None (baseline) | Linear, requires feature scaling |
| Random Forest | 80 Optuna trials | Ensemble of 550 trees, strong regularization |
| XGBoost | 100 Optuna trials | Gradient-boosted, GPU-accelerated, early stopping |

---

## Results

### Final Test Set Performance

| Model | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Random Forest (Tuned) | 0.748 | 0.082 | 0.662 | 0.044 |
| **XGBoost (Tuned)** | **0.752** | **0.458** | **0.320** | **0.805** |

### Top Predictors (by feature importance)

1. **Lead time** (~58%) — the dominant predictor; longer gaps between scheduling and appointment strongly correlate with no-shows
2. Neighbourhood no-show rate (~8%)
3. Patient previous no-show rate (~7%)
4. Age group (~5%)
5. Appointment / scheduled month (~4% each)

### Key Takeaways

- **XGBoost** is the production winner with **0.752 AUC-ROC** and **80.5% recall**, meaning it catches ~4 out of 5 no-shows
- The **~32% precision** is acceptable for low-cost interventions (e.g., automated SMS reminders cost virtually nothing, so flagging 68% false positives is still operationally useful)
- The **~0.75 AUC ceiling** reflects the inherent limits of the dataset (no socioeconomic, weather, or transportation data)
- Random Forest has near-zero recall (4.4%) — it classifies almost everyone as "will attend," making it unsuitable for this use case
- Both models generalize well (CV gap < 0.05), indicating no overfitting

---

## Project Structure

```
├── api/                          # Flask web API
│   ├── __init__.py
│   ├── inference.py              # Feature building for single predictions
│   ├── main.py                   # Flask routes (/, /health, /predict)
│   ├── model_loader.py           # Lazy model/encoder loading
│   └── schemas.py                # Input validation (13 fields)
├── data/
│   ├── KaggleV2-May-2016.csv     # Raw dataset (110,527 records)
│   ├── preprocessed_data.csv     # Cleaned feature matrix
│   └── splits/indices.pkl        # Reproducible train/val/test indices
├── docs/tuning/                  # Pipeline documentation (phases 1–3)
├── mlruns/                       # MLflow experiment tracking data
├── models/
│   ├── best_rf.pkl               # Tuned Random Forest
│   ├── best_xgb_latest.pkl       # Tuned XGBoost (production)
│   ├── metadata.pkl              # Feature order + model name
│   └── neighbourhood_encoder.pkl # Fitted encoder for API inference
├── results/
│   ├── comparison_table.csv      # RF vs XGB test metrics
│   └── feature_importance.png    # Top 10 predictors chart
├── src/                          # Training & tuning pipeline
│   ├── config.py                 # Centralized configuration
│   ├── cv_utils.py               # Stratified cross-validation
│   ├── data_split.py             # Train/val/test splitting
│   ├── evaluate.py               # Final test evaluation
│   ├── leakage_check.py          # Target leakage detection
│   ├── mlflow_logger.py          # MLflow experiment tracking
│   ├── tune_rf.py                # Random Forest Optuna tuning
│   ├── tune_xgb.py               # XGBoost Optuna tuning
│   └── train_pipeline.py         # Pipeline orchestrator
├── static/index.html             # Single-page frontend
├── feature_engineering.py        # 17-feature engineering logic
├── preprocessing.py              # Data cleaning & encoding
├── train.py                      # Baseline training entry point
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Compose service definition
└── Hospital_No_Show_Predictor_IEEE_report.pdf  # Full project report (IEEE format)
```

---

## Quick Start

### Docker (Recommended)

```bash
docker-compose up --build
```

Open [http://localhost:80](http://localhost:80)

### Local Development

```bash
pip install -r requirements.txt
python -m api.main
```

Open [http://localhost:8000](http://localhost:8000)

---

## Usage

### Web Interface

The frontend (`static/index.html`) provides an interactive form with:

- Toggle switches for binary features (scholarship, comorbidities, SMS received)
- Slider for age
- Date pickers for scheduled and appointment dates
- Dropdown for neighbourhood (81 neighbourhoods from Vitoria, Brazil)
- Animated gauge chart showing the predicted risk probability (0–100%)
- Color-coded risk badges: **Low** (<30%, green), **Medium** (30–60%, yellow), **High** (>60%, red)
- Plain-English explanation of the prediction
- Collapsible panel showing all 17 engineered features

### API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Serves the web frontend |
| `/health` | GET | Health check with model info |
| `/predict` | POST | Predict no-show probability |

**`POST /predict`** example:

```json
{
  "gender": 0,
  "age": 35,
  "scholarship": 0,
  "hypertension": 0,
  "diabetes": 0,
  "alcoholism": 0,
  "handicap": 0,
  "sms_received": 1,
  "scheduled_date": "2026-06-15",
  "appointment_date": "2026-06-22",
  "neighbourhood": "JARDIM DA PENHA",
  "previous_appointments": 2,
  "previous_no_shows": 1
}
```

Response:

```json
{
  "probability": 0.45,
  "prediction": 0,
  "risk_level": "Medium Risk",
  "explanation": "This patient has a medium probability of missing their appointment. The main risk factors are a previous no-show rate of 50.0% and a lead time of 7 days.",
  "features_used": { "lead_time": 7, "previous_no_show_rate": 0.5, ... }
}
```

### Training from Scratch

```bash
# Baseline training (3 models, no tuning)
python train.py

# Full tuning pipeline (4 phases)
python src/train_pipeline.py    # Phase 1: data splitting + leakage check
python src/tune_rf.py           # Phase 3.1: Random Forest tuning
python src/tune_xgb.py          # Phase 3.2: XGBoost tuning
python src/evaluate.py          # Phase 4: final evaluation
```

### MLflow UI

```bash
mlflow ui --mlruns-dir mlruns
# Opens at http://localhost:5000
```

---

## Key Design Decisions

1. **Target encoding for neighbourhoods** — Using smoothed empirical Bayes shrinkage instead of one-hot encoding avoids the curse of dimensionality while capturing geographic no-show patterns. The smoothing parameter prevents overfitting on rare neighbourhoods.

2. **Split before feature engineering** — All temporal and historical features are computed from the training set only, preventing any information leakage from the validation or test sets.

3. **Gap monitoring in CV** — If the gap between CV and validation performance exceeds 0.08 AUC, the pipeline triggers a warning, flagging potential overfitting before the model is deployed.

4. **GPU fallback** — XGBoost automatically detects CUDA availability and falls back to CPU `hist` tree method if no GPU is present, ensuring portability across environments.

5. **Recall over precision** — The production XGBoost model prioritizes recall (80.5%) over precision (32%). This is intentional: false positives (sending an SMS to someone who would have attended) cost virtually nothing, while false negatives (missing a true no-show) waste a valuable appointment slot.

6. **Lazy model loading** — The API loads model artifacts on the first request rather than at import time, keeping the container startup fast.

---
