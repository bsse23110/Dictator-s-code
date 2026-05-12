
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_PATH = "data/KaggleV2-May-2016.csv"

RENAME_MAP = {
    "PatientId":      "patient_id",
    "AppointmentID":  "appointment_id",
    "Gender":         "gender",
    "ScheduledDay":   "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age":            "age",
    "Neighbourhood":  "neighbourhood",
    "Scholarship":    "scholarship",
    "Hipertension":   "hypertension",   # fix typo
    "Diabetes":       "diabetes",
    "Alcoholism":     "alcoholism",
    "Handcap":        "handicap",       # fix typo
    "SMS_received":   "sms_received",
    "No-show":        "no_show",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_clean(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load the raw CSV and return a fully cleaned DataFrame.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with correct dtypes and no obvious data errors.
    """
    df = pd.read_csv(path)

    # 1. Rename columns -------------------------------------------------
    df = df.rename(columns=RENAME_MAP)

    # 2. Parse datetime columns ----------------------------------------
    df["scheduled_day"]   = pd.to_datetime(df["scheduled_day"],   utc=True)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True)

    # Normalise appointment_day to date-only (time is always midnight)
    df["appointment_day"] = df["appointment_day"].dt.normalize()

    # 3. Fix Age --------------------------------------------------------
    # Negative ages are clearly erroneous → replace with median
    median_age = df.loc[df["age"] >= 0, "age"].median()
    n_bad_age  = (df["age"] < 0).sum()
    if n_bad_age:
        print(f"[preprocessing] Replacing {n_bad_age} negative age(s) with median ({median_age:.0f})")
        df.loc[df["age"] < 0, "age"] = median_age
    df["age"] = df["age"].astype(int)

    # 4. Fix Handicap ---------------------------------------------------
    # Original column has values 0-4; levels >1 are very rare and may be
    # miscoded.  We cap at 1 (binary: has some handicap vs. none) to avoid
    # spurious ordinal signal.
    df["handicap"] = df["handicap"].clip(upper=1)

    # 5. Encode binary gender ------------------------------------------
    df["gender"] = (df["gender"].str.upper() == "F").astype(int)
    # gender: 1 = Female, 0 = Male

    # 6. Encode target label -------------------------------------------
    df["no_show"] = (df["no_show"].str.strip().str.lower() == "yes").astype(int)
    # no_show: 1 = patient did NOT show up, 0 = showed up

    # 7. Drop ID columns (not useful for ML) ---------------------------
    df = df.drop(columns=["appointment_id"])
    # Keep patient_id — needed by feature_engineering for historical rates

    # 8. Verify no nulls -----------------------------------------------
    assert df.isnull().sum().sum() == 0, "Unexpected nulls after cleaning!"

    print(f"[preprocessing] Clean dataset shape: {df.shape}")
    print(f"[preprocessing] No-show rate: {df['no_show'].mean():.2%}")

    return df


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else RAW_PATH
    df = load_and_clean(path)
    print(df.head())
    print(df.dtypes)
