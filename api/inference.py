"""
api/inference.py
----------------
Converts raw patient input into model-ready features using the
saved NeighbourhoodEncoder and the same logic as feature_engineering.py.
"""

import pandas as pd
import numpy as np
from datetime import date

from api.model_loader import get_encoder, get_metadata


def _age_group(age: int) -> int:
    """Bin age into ordinal group: 0=child, 1=teen, 2=adult, 3=senior."""
    if age <= 12:
        return 0
    elif age <= 17:
        return 1
    elif age <= 59:
        return 2
    else:
        return 3


def build_features(input_data: dict) -> pd.DataFrame:
    """
    Convert form input dict to a single-row DataFrame with all 17 features
    in the exact order expected by the model.
    """
    meta = get_metadata()
    encoder = get_encoder()

    # Compute derived temporal features
    appointment_date: date = input_data["appointment_date"]
    scheduled_date: date = input_data["scheduled_date"]

    lead_time_days = max(0, (appointment_date - scheduled_date).days)
    weekday = appointment_date.weekday()           # 0=Mon … 6=Sun
    is_weekend = 1 if weekday >= 5 else 0
    scheduled_month = scheduled_date.month
    appointment_month = appointment_date.month

    # Demographic
    age = input_data["age"]
    age_grp = _age_group(age)

    # Historical behaviour
    prev_appts = input_data.get("previous_appointments", 0)
    prev_noshows = input_data.get("previous_no_shows", 0)
    prev_rate = prev_noshows / prev_appts if prev_appts > 0 else 0.0

    # Neighbourhood encoding via saved encoder
    neighbourhood_name = input_data["neighbourhood"]
    # Create a mini DataFrame for the encoder's transform method
    mini_df = pd.DataFrame({"neighbourhood": [neighbourhood_name]})
    nh_rate = encoder.transform(mini_df).iloc[0]

    # Build the row in exact feature order
    row = {
        "gender":                     input_data["gender"],
        "age":                        age,
        "scholarship":                input_data["scholarship"],
        "hypertension":               input_data["hypertension"],
        "diabetes":                    input_data["diabetes"],
        "alcoholism":                 input_data["alcoholism"],
        "handicap":                   input_data["handicap"],
        "sms_received":               input_data["sms_received"],
        "lead_time":                  lead_time_days,
        "is_weekend":                 is_weekend,
        "appointment_weekday":        weekday,
        "scheduled_month":            scheduled_month,
        "appointment_month":          appointment_month,
        "age_group":                  age_grp,
        "previous_no_show_rate":      prev_rate,
        "previous_appointments":      prev_appts,
        "neighbourhood_no_show_rate": nh_rate,
    }

    # Enforce column order
    ordered = {col: row[col] for col in meta["feature_order"]}
    return pd.DataFrame([ordered]), row
