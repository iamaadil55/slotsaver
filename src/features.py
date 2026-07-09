"""Leakage-safe feature engineering and temporal train/test split.

THE two ideas that make this project credible live in this file:

1. Patient-history features must only use appointments that happened BEFORE
   the one being scored (add_history_features).
2. The test set must be strictly LATER in time than the training set
   (temporal_split). Random splits let the model peek at the future.
"""

import pandas as pd

CATEGORICAL = ["gender", "neighbourhood", "weekday"]
BINARY = ["scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received"]
NUMERIC = ["age", "lead_time_days", "prior_appointments", "prior_noshows", "prior_noshow_rate"]
FEATURES = CATEGORICAL + BINARY + NUMERIC
LABEL = "no_show"


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient history, computed only from PAST appointments.

    Sort each patient's appointments by date, then for appointment #i:
      prior_appointments = i          (how many came before)
      prior_noshows      = missed among those i
      prior_noshow_rate  = prior_noshows / prior_appointments (NaN if no history)

    The classic mistake is including the current row's own label (e.g. via a
    plain cumsum) — that hands the model the answer. We shift by excluding
    the current appointment explicitly.

    Known simplification (documented in implementation-notes.md): we treat an
    appointment's outcome as known immediately after its appointment_day.
    """
    df = df.sort_values(["patient_id", "appointment_day", "scheduled_day"]).copy()
    g = df.groupby("patient_id")
    df["prior_appointments"] = g.cumcount()
    # cumsum includes the current row -> subtract it back out.
    df["prior_noshows"] = g["no_show"].cumsum() - df["no_show"]
    df["prior_noshow_rate"] = df["prior_noshows"] / df["prior_appointments"]
    # first-ever appointment: rate is NaN ("no history"), which LightGBM handles
    # natively and the sklearn pipeline imputes + flags with a missing indicator.
    df["weekday"] = df["appointment_day"].dt.day_name()
    return df


def temporal_split(df: pd.DataFrame, test_frac: float = 0.25):
    """Train on the past, test on the future.

    We cut at a DATE (the test_frac quantile of appointment days), not at a row
    count, so no single day is split across train and test.
    """
    cutoff = df["appointment_day"].quantile(1 - test_frac)
    train = df[df["appointment_day"] < cutoff]
    test = df[df["appointment_day"] >= cutoff]
    print(f"temporal_split(): cutoff={cutoff.date()}  "
          f"train={len(train)} rows (to {train['appointment_day'].max().date()})  "
          f"test={len(test)} rows (from {test['appointment_day'].min().date()})")
    return train, test


def rule_baseline(df: pd.DataFrame) -> pd.Series:
    """What a clinic could do with no ML at all:
    flag if the patient already missed >=2 appointments OR booked >14 days out.
    Any model that can't beat this doesn't earn its complexity.
    """
    return ((df["prior_noshows"] >= 2) | (df["lead_time_days"] > 14)).astype(int)
