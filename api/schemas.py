
REQUIRED_FIELDS = {
    "gender":                 (int, 0, 1),
    "age":                    (int, 0, 120),
    "scholarship":            (int, 0, 1),
    "hypertension":           (int, 0, 1),
    "diabetes":               (int, 0, 1),
    "alcoholism":             (int, 0, 1),
    "handicap":               (int, 0, 1),
    "sms_received":           (int, 0, 1),
    "scheduled_date":         (str,),
    "appointment_date":       (str,),
    "neighbourhood":          (str,),
    "previous_appointments":  (int, 0, 999),
    "previous_no_shows":      (int, 0, 999),
}


def validate_patient_input(data: dict) -> list:
    """Return list of error strings. Empty list means valid."""
    errors = []
    for field, spec in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing field: {field}")
            continue
        val = data[field]
        expected_type = spec[0]
        if not isinstance(val, expected_type):
            errors.append(f"Field '{field}' must be {expected_type.__name__}, got {type(val).__name__}")
        elif len(spec) == 3:
            lo, hi = spec[1], spec[2]
            if not (lo <= val <= hi):
                errors.append(f"Field '{field}' must be between {lo} and {hi}, got {val}")
    if data.get("previous_no_shows", 0) > data.get("previous_appointments", 0):
        errors.append("previous_no_shows cannot exceed previous_appointments")
    return errors
