"""End-to-end scoring pipeline — one function the notebook, the dashboard, and
(later, Level 3) the API all share. Single source of truth: if the modeling
changes, every surface updates together.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.calibrate import ev_call_threshold, fit_isotonic, risk_tiers, three_way_temporal_split
from src.data import load_or_synthesize
from src.explain import top_reasons_frame
from src.features import CATEGORICAL, FEATURES, LABEL, add_history_features


def to_xy(d: pd.DataFrame):
    X = d[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X, d[LABEL]


def build_scoring_artifacts(csv_path: str = "data/KaggleV2-May-2016.csv") -> dict:
    """Load -> features -> 3-way temporal split -> plain LGBM -> isotonic ->
    calibrated scores + tiers on the test window (our stand-in for 'tomorrow's
    appointments' until Level 3 wires a live feed)."""
    df, is_real = load_or_synthesize(csv_path)
    df = add_history_features(df)
    train, cal, test = three_way_temporal_split(df)

    X_train, y_train = to_xy(train)
    X_cal, y_cal = to_xy(cal)
    X_test, _ = to_xy(test)

    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    iso = fit_isotonic(model.predict_proba(X_cal)[:, 1], y_cal)

    scored = test.copy()
    p = iso.predict(model.predict_proba(X_test)[:, 1])
    # Real-data finding (2026-07-06): isotonic's sparse extreme bins produced
    # p = 1.0 for a handful of patients. Never claim certainty about future
    # human behavior — a "100%" that shows up once destroys the clinic's trust.
    scored["p_noshow"] = np.clip(p, 0.01, 0.99)
    scored["tier"] = risk_tiers(scored["p_noshow"]).values
    return {"model": model, "iso": iso, "scored_test": scored,
            "X_test": X_test, "is_real": is_real}


def daily_call_list(art: dict, day, capacity: int = 20, cost_per_call: float = 5.0,
                    success: float = 0.3, revenue: float = 200.0):
    """The product's core object: one clinic day, ranked by calibrated risk.

    Decision logic (Level 2a's two constraints combined):
    - ECONOMICS: only patients above p* = cost/(success*revenue) are worth a call.
    - CAPACITY: of those, staff call only the top `capacity` by probability.
    Everyone else falls back to their tier's cheaper intervention.

    Returns (day_df, summary). Expected value uses probabilities, not labels —
    exactly what a clinic could compute BEFORE the day happens.
    """
    s = art["scored_test"]
    all_rows = s[s["appointment_day"].dt.date == day].sort_values(
        "p_noshow", ascending=False).copy()

    # Real-data finding (2026-07-06): one child had 5 appointments on the same
    # day and occupied 5 of the top slots — 25% of call capacity burned on one
    # family. A call list is per PATIENT, not per appointment: keep each
    # patient's highest-risk row and record how many appointments they have.
    all_rows["appts_today"] = all_rows.groupby("patient_id")["patient_id"].transform("size")
    day_df = all_rows[~all_rows["patient_id"].duplicated(keep="first")].copy()

    thr = ev_call_threshold(revenue, success, cost_per_call)
    day_df["worth_calling"] = day_df["p_noshow"] > thr
    day_df["call_today"] = False
    call_idx = day_df.index[day_df["worth_calling"].to_numpy()][:capacity]
    day_df.loc[call_idx, "call_today"] = True

    day_df["why"] = top_reasons_frame(art["model"], art["X_test"].loc[day_df.index])

    called = day_df[day_df["call_today"]]
    summary = {
        "appointments": len(all_rows),
        "unique_patients": len(day_df),
        "expected_noshows": float(all_rows["p_noshow"].sum()),
        "calls": int(len(called)),
        "expected_noshows_reached": float(called["p_noshow"].sum()),
        "expected_value_usd": float((called["p_noshow"] * success * revenue).sum()
                                    - len(called) * cost_per_call),
        "ev_threshold": float(thr),
    }
    return day_df, summary
