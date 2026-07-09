"""SlotSaver scoring API. Run locally:  uvicorn api:app --reload

Endpoints:
  GET  /health              — liveness + which data mode we're in
  POST /score               — one appointment in, calibrated risk + tier + reasons out
  GET  /call-list           — a clinic day's ranked call list (?day=YYYY-MM-DD&capacity=20)

Design note: the model trains at startup from the same build_scoring_artifacts()
the notebooks and dashboard use — one source of truth. In a real deployment
you'd persist the trained artifacts and load them, but retraining takes seconds
here and keeps the demo dependency-free.
"""

from datetime import date

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.calibrate import ev_call_threshold
from src.explain import top_reasons_frame
from src.features import CATEGORICAL, FEATURES
from src.pipeline import build_scoring_artifacts, daily_call_list

app = FastAPI(title="SlotSaver", version="0.3.0")
ART = build_scoring_artifacts()  # trains + calibrates at startup (~seconds)


class Appointment(BaseModel):
    """One appointment to score. History fields default to 'new patient'."""
    age: int = Field(ge=0, le=110)
    lead_time_days: int = Field(ge=0, le=365)
    gender: str = "F"
    weekday: str = "Monday"
    neighbourhood: str = "UNKNOWN"
    scholarship: int = 0
    hypertension: int = 0
    diabetes: int = 0
    alcoholism: int = 0
    handicap: int = 0
    sms_received: int = 0
    prior_appointments: int = 0
    prior_noshows: int = 0


@app.get("/health")
def health():
    return {"status": "ok", "real_data": ART["is_real"],
            "test_window_days": int(ART["scored_test"]["appointment_day"].nunique())}


@app.post("/score")
def score(appt: Appointment):
    row = appt.model_dump()
    row["prior_noshow_rate"] = (row["prior_noshows"] / row["prior_appointments"]
                                if row["prior_appointments"] > 0 else np.nan)
    X = pd.DataFrame([row])[FEATURES]
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")

    raw = float(ART["model"].predict_proba(X)[:, 1][0])
    p = float(np.clip(ART["iso"].predict([raw])[0], 0.01, 0.99))
    tier = "staff_call" if p >= 0.35 else ("extra_reminder" if p >= 0.15 else "sms_only")
    return {
        "p_noshow": round(p, 3),
        "tier": tier,
        "why": top_reasons_frame(ART["model"], X)[0],
        "worth_calling_at_default_economics": p > ev_call_threshold(),
    }


@app.get("/call-list")
def call_list(day: str, capacity: int = 20):
    try:
        d = date.fromisoformat(day)
    except ValueError:
        raise HTTPException(422, "day must be YYYY-MM-DD")
    day_df, summary = daily_call_list(ART, d, capacity=capacity)
    if summary["appointments"] == 0:
        available = sorted(ART["scored_test"]["appointment_day"].dt.date.unique())
        raise HTTPException(404, f"no appointments on {day}; available: "
                                 f"{available[0]}..{available[-1]}")
    cols = ["patient_id", "p_noshow", "tier", "why", "appts_today", "call_today"]
    return {"summary": summary,
            "calls": day_df.loc[day_df["call_today"], cols].to_dict(orient="records")}
