"""
api/main.py
-----------
FastAPI application with /predict and /health endpoints.
Loads the trained XGBoost model and serves predictions.
"""

import sys
import os

# Ensure api/ and project root are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import PatientInput, PredictionResult, HealthResponse
from api.inference import build_features
from api.model_loader import load_artifacts, get_model, get_metadata

# ── App Initialization ──────────────────────────────────────────────

app = FastAPI(
    title="Hospital No-Show Predictor",
    description="Predicts the probability that a patient will miss their medical appointment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
@app.on_event("startup")
def startup():
    load_artifacts()


# ── Helper ───────────────────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


def _explanation(prob: float, risk: str) -> str:
    if risk == "Low Risk":
        return "This patient is likely to attend the appointment. No immediate intervention needed."
    elif risk == "Medium Risk":
        return "This patient has a moderate chance of missing the appointment. Consider sending an SMS reminder."
    else:
        return "This patient is at high risk of missing the appointment. A phone call reminder or rescheduling is strongly recommended."


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    meta = get_metadata()
    return HealthResponse(model=meta["model_name"], features=len(meta["feature_order"]))


@app.post("/predict", response_model=PredictionResult)
def predict(patient: PatientInput):
    try:
        model = get_model()

        input_dict = patient.model_dump()
        # Convert date to Python date objects
        input_dict["appointment_date"] = patient.appointment_date
        input_dict["scheduled_date"] = patient.scheduled_date

        feats_df, feats_dict = build_features(input_dict)
        proba = float(model.predict_proba(feats_df)[:, 1][0])
        pred = int(proba >= 0.5)
        risk = _risk_level(proba)
        explanation = _explanation(proba, risk)

        return PredictionResult(
            probability=round(proba, 4),
            prediction=pred,
            risk_level=risk,
            explanation=explanation,
            features_used={k: (round(v, 4) if isinstance(v, float) else v) for k, v in feats_dict.items()},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve Frontend (production) ──────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
