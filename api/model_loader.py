"""
api/model_loader.py
-------------------
Loads the trained XGBoost model, NeighbourhoodEncoder, and metadata
for inference. Call load_artifacts() once at startup.
"""

import joblib
import os

_MODEL = None
_ENCODER = None
_METADATA = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_artifacts():
    """Load model, encoder, and metadata into global cache. Call once at startup."""
    global _MODEL, _ENCODER, _METADATA

    _MODEL    = joblib.load(os.path.join(MODEL_DIR, "best_xgb_latest.pkl"))
    _ENCODER  = joblib.load(os.path.join(MODEL_DIR, "neighbourhood_encoder.pkl"))
    _METADATA = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

    print(f"[model_loader] Model loaded: {_METADATA['model_name']}")
    print(f"[model_loader] Features: {len(_METADATA['feature_order'])}")
    return _MODEL, _ENCODER, _METADATA


def get_model():
    if _MODEL is None:
        load_artifacts()
    return _MODEL


def get_encoder():
    if _ENCODER is None:
        load_artifacts()
    return _ENCODER


def get_metadata():
    if _METADATA is None:
        load_artifacts()
    return _METADATA
