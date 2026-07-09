"""Load, clean, and synthesize appointment data.

The real dataset: https://www.kaggle.com/datasets/joniarroba/noshowappointments
(~110k appointments, Brazil, 6 weeks of 2016). Download the CSV into data/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# The Kaggle CSV has misspelled column names — we keep an explicit rename map
# so the rest of the codebase uses clean names.
RENAME = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",   # sic in the raw file
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",            # sic in the raw file
    "SMS_received": "sms_received",
    "No-show": "no_show_raw",
}


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the Kaggle CSV and standardize column names and types."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RENAME)
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"]).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"]).dt.tz_localize(None)
    # Label: 1 = patient did NOT show up (the event we want to predict).
    df["no_show"] = (df["no_show_raw"] == "Yes").astype(int)
    df = df.drop(columns=["no_show_raw"])
    df["patient_id"] = df["patient_id"].astype("int64")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove impossible records. Every drop is counted and printed —
    silent data cleaning is how errors hide."""
    n0 = len(df)

    # Age -1 exists in the real data; ages > 110 are almost certainly errors.
    bad_age = (df["age"] < 0) | (df["age"] > 110)

    # lead time = days between booking and appointment.
    # ScheduledDay has a timestamp, AppointmentDay is midnight — compare dates only,
    # otherwise every same-day booking looks negative.
    lead = (df["appointment_day"].dt.normalize() - df["scheduled_day"].dt.normalize()).dt.days
    bad_lead = lead < 0  # "scheduled after the appointment happened" — data errors

    df = df.loc[~bad_age & ~bad_lead].copy()
    df["lead_time_days"] = lead[~bad_age & ~bad_lead]

    print(f"clean(): dropped {bad_age.sum()} bad-age rows, "
          f"{bad_lead.sum()} negative-lead-time rows ({n0} -> {len(df)})")
    return df


def make_synthetic(n: int = 20_000, n_patients: int = 8_000, seed: int = 42) -> pd.DataFrame:
    """Generate fake data with the SAME schema as the cleaned Kaggle data.

    Used by smoke_test.py and as a notebook fallback so the pipeline can be
    exercised before you download the real CSV. Patterns are planted on purpose
    (longer lead time and prior no-shows raise risk) so models have signal to find.
    """
    rng = np.random.default_rng(seed)
    patient_id = rng.integers(0, n_patients, size=n)
    age = rng.integers(0, 95, size=n)
    lead = rng.exponential(scale=10, size=n).astype(int).clip(0, 60)
    sms = (lead > 2) & (rng.random(n) < 0.6)  # SMS mostly sent for non-same-day bookings

    start = pd.Timestamp("2016-04-29")
    appointment_day = start + pd.to_timedelta(rng.integers(0, 42, size=n), unit="D")
    scheduled_day = appointment_day - pd.to_timedelta(lead, unit="D")

    # Planted ground truth: base rate ~20%, driven by lead time, youth, and no SMS.
    logit = -1.9 + 0.045 * lead - 0.010 * age - 0.35 * sms.astype(int)
    # Give some patients a persistent no-show tendency (this is what prior_* features catch).
    patient_effect = rng.normal(0, 0.9, size=n_patients)
    logit = logit + patient_effect[patient_id]
    p = 1 / (1 + np.exp(-logit))
    no_show = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "appointment_id": np.arange(n),
        "gender": rng.choice(["F", "M"], size=n, p=[0.65, 0.35]),
        "scheduled_day": scheduled_day,
        "appointment_day": appointment_day,
        "age": age,
        "neighbourhood": rng.choice([f"NB_{i}" for i in range(40)], size=n),
        "scholarship": rng.integers(0, 2, size=n),
        "hypertension": (age > 50).astype(int) & rng.integers(0, 2, size=n),
        "diabetes": rng.binomial(1, 0.07, size=n),
        "alcoholism": rng.binomial(1, 0.03, size=n),
        "handicap": rng.binomial(1, 0.02, size=n),
        "sms_received": sms.astype(int),
        "no_show": no_show,
        "lead_time_days": lead,
    })
    return df


def load_or_synthesize(csv_path: str | Path = "data/KaggleV2-May-2016.csv") -> tuple[pd.DataFrame, bool]:
    """Return (cleaned dataframe, is_real_data)."""
    csv_path = Path(csv_path)
    if csv_path.exists():
        return clean(load_raw(csv_path)), True
    print(f"WARNING: {csv_path} not found -> using SYNTHETIC data. "
          "Results are meaningless until you use the real CSV (see README).")
    return make_synthetic(), False
