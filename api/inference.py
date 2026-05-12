
import pandas as pd
from datetime import date

from api.model_loader import get_encoder, get_metadata


def _age_group(age: int) -> int:
    if age <= 12: return 0
    elif age <= 17: return 1
    elif age <= 59: return 2
    else: return 3


def build_features(data: dict):
    """Convert input dict to a single-row DataFrame with all 17 features."""
    meta = get_metadata()
    encoder = get_encoder()

    appt = date.fromisoformat(data["appointment_date"])
    sched = date.fromisoformat(data["scheduled_date"])

    lead_time_days = max(0, (appt - sched).days)
    weekday = appt.weekday()
    is_weekend = 1 if weekday >= 5 else 0
    scheduled_month = sched.month
    appointment_month = appt.month
    age = data["age"]
    age_grp = _age_group(age)
    prev_appts = data.get("previous_appointments", 0)
    prev_noshows = data.get("previous_no_shows", 0)
    prev_rate = prev_noshows / prev_appts if prev_appts > 0 else 0.0

    nh = data["neighbourhood"]
    mini_df = pd.DataFrame({"neighbourhood": [nh]})
    nh_rate = encoder.transform(mini_df).iloc[0]

    row = {
        "gender":                     data["gender"],
        "age":                        age,
        "scholarship":                data["scholarship"],
        "hypertension":               data["hypertension"],
        "diabetes":                    data["diabetes"],
        "alcoholism":                 data["alcoholism"],
        "handicap":                   data["handicap"],
        "sms_received":               data["sms_received"],
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
    ordered = {col: row[col] for col in meta["feature_order"]}
    return pd.DataFrame([ordered]), row
