

import pandas as pd
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _age_group(age_series: pd.Series) -> pd.Series:
    """Bin age into four ordinal groups."""
    bins   = [-1, 12, 17, 59, 200]
    labels = [0, 1, 2, 3]          # child, teen, adult, senior
    return pd.cut(age_series, bins=bins, labels=labels).astype(int)


def _compute_lead_time(df: pd.DataFrame) -> pd.Series:
    """Days between scheduling and appointment (floor to 0 if negative)."""
    lead = (df["appointment_day"] - df["scheduled_day"].dt.normalize()).dt.days
    return lead.clip(lower=0)


def _historical_no_show_rates(
    df: pd.DataFrame,
    target_col: str = "no_show",
) -> Tuple[pd.Series, pd.Series]:
    """
    For each row compute:
      - how many prior appointments the patient had
      - what fraction they missed (excluding the current row)

    The data is sorted by scheduled_day so 'prior' is well-defined.

    Returns two Series aligned to df.index.
    """
    df_sorted = df.sort_values("scheduled_day").copy()

    # cumulative sum and count *before* the current row, grouped by patient
    grp = df_sorted.groupby("patient_id")[target_col]
    cum_sum   = grp.cumsum() - df_sorted[target_col]   # exclude self
    cum_count = grp.cumcount()                          # count before current row

    # Rate: 0 if no prior history
    rate = np.where(cum_count > 0, cum_sum / cum_count, 0.0)

    # Re-align to original index
    prev_no_show_rate  = pd.Series(rate,        index=df_sorted.index, name="previous_no_show_rate")
    prev_appointments  = pd.Series(cum_count,   index=df_sorted.index, name="previous_appointments")

    return (
        prev_no_show_rate.reindex(df.index),
        prev_appointments.reindex(df.index),
    )


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------

class NeighbourhoodEncoder:
    """
    Target-encodes neighbourhood using training-set no-show rates.
    Falls back to the global mean for unseen neighbourhoods.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self._global_mean: float = 0.0
        self._map: dict = {}

    def fit(self, df: pd.DataFrame, target_col: str = "no_show") -> "NeighbourhoodEncoder":
        global_mean = df[target_col].mean()
        self._global_mean = global_mean

        stats = df.groupby("neighbourhood")[target_col].agg(["sum", "count"])
        # Smoothed target encoding: (count * mean + k * global_mean) / (count + k)
        stats["smoothed"] = (
            (stats["sum"] + self.smoothing * global_mean)
            / (stats["count"] + self.smoothing)
        )
        self._map = stats["smoothed"].to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        return df["neighbourhood"].map(self._map).fillna(self._global_mean)


def engineer_features(
    df: pd.DataFrame,
    neighbourhood_encoder: Optional[NeighbourhoodEncoder] = None,
    is_train: bool = True,
) -> Tuple[pd.DataFrame, Optional[NeighbourhoodEncoder]]:
    """
    Build the full feature matrix from the cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of preprocessing.load_and_clean().
    neighbourhood_encoder : NeighbourhoodEncoder or None
        Pass None (or omit) when building training features — the encoder
        will be fitted here and returned.
        Pass the fitted encoder when processing the test set.
    is_train : bool
        If True, fits the neighbourhood encoder on df.
        If False, uses the provided encoder.

    Returns
    -------
    features : pd.DataFrame
        Model-ready feature matrix (no raw dates, no string columns).
    encoder : NeighbourhoodEncoder
        Fitted neighbourhood encoder (same object returned for reuse on test).
    """

    feats = df.copy()

    # --- Temporal features -------------------------------------------
    feats["lead_time"]           = _compute_lead_time(feats)
    feats["is_weekend"]          = feats["appointment_day"].dt.dayofweek.isin([5, 6]).astype(int)
    feats["appointment_weekday"] = feats["appointment_day"].dt.dayofweek.astype(int)
    feats["scheduled_month"]     = feats["scheduled_day"].dt.month.astype(int)
    feats["appointment_month"]   = feats["appointment_day"].dt.month.astype(int)

    # --- Demographic features ----------------------------------------
    feats["age_group"] = _age_group(feats["age"])

    # --- Historical behaviour ----------------------------------------
    prev_rate, prev_count = _historical_no_show_rates(feats)
    feats["previous_no_show_rate"] = prev_rate.values
    feats["previous_appointments"] = prev_count.values

    # --- Neighbourhood target encoding --------------------------------
    if is_train or neighbourhood_encoder is None:
        neighbourhood_encoder = NeighbourhoodEncoder()
        neighbourhood_encoder.fit(feats)

    feats["neighbourhood_no_show_rate"] = neighbourhood_encoder.transform(feats)

    # --- Drop columns not used in model training ----------------------
    drop_cols = ["scheduled_day", "appointment_day", "patient_id", "neighbourhood"]
    feats = feats.drop(columns=[c for c in drop_cols if c in feats.columns])

    print(f"[feature_engineering] Feature matrix shape: {feats.shape}")
    print(f"[feature_engineering] Columns: {feats.columns.tolist()}")

    return feats, neighbourhood_encoder


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import load_and_clean

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/KaggleV2-May-2016.csv"
    df_clean  = load_and_clean(data_path)
    feats, _  = engineer_features(df_clean, is_train=True)
    print(feats.head())
    print(feats.describe())
