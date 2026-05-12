
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from api.schemas import validate_patient_input
from api.inference import build_features
from api.model_loader import load_artifacts, get_model, get_metadata

app = Flask(__name__, static_folder="../static", static_url_path="")
CORS(app)

# ── Load model at startup ───────────────────────────────────────────
load_artifacts()


# ── Helpers ──────────────────────────────────────────────────────────

def risk_level(prob):
    if prob < 0.30: return "Low Risk"
    elif prob < 0.60: return "Medium Risk"
    else: return "High Risk"

def explanation(prob, risk):
    if risk == "Low Risk":
        return "This patient is likely to attend the appointment. No immediate intervention needed."
    elif risk == "Medium Risk":
        return "This patient has a moderate chance of missing the appointment. Consider sending an SMS reminder."
    else:
        return "This patient is at high risk of missing the appointment. A phone call reminder or rescheduling is strongly recommended."


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return send_from_directory("../static", "index.html")


@app.route("/health")
def health():
    meta = get_metadata()
    return jsonify(status="ok", model=meta["model_name"], features=len(meta["feature_order"]))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        errors = validate_patient_input(data)
        if errors:
            return jsonify(error="Validation failed", details=errors), 400

        model = get_model()
        feats_df, feats_dict = build_features(data)
        proba = float(model.predict_proba(feats_df)[:, 1][0])
        pred = int(proba >= 0.5)
        risk = risk_level(proba)

        return jsonify(
            probability=round(proba, 4),
            prediction=pred,
            risk_level=risk,
            explanation=explanation(proba, risk),
            features_used={k: (round(v, 4) if isinstance(v, float) else v) for k, v in feats_dict.items()},
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
