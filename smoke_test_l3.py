"""Level 3 checks on synthetic data. Run: python smoke_test_l3.py"""

import numpy as np
import pandas as pd

from src.calibrate import three_way_temporal_split
from src.data import make_synthetic
from src.fairness import audit, flag_concerns
from src.features import add_history_features
from src.monitor import any_alert, drift_report, psi
from src.pipeline import build_scoring_artifacts, daily_call_list


def main() -> None:
    art = build_scoring_artifacts("data/does_not_exist.csv")  # synthetic
    scored = art["scored_test"]

    # --- Tie-aware ranking ---
    assert "raw_score" in scored.columns
    days = sorted(scored["appointment_day"].dt.date.unique())
    day_df, summary = daily_call_list(art, days[-1], capacity=20)
    pairs = list(zip(day_df["p_noshow"].values, day_df["raw_score"].values))
    assert all(a >= b for a, b in zip(pairs, pairs[1:])), \
        "list must be sorted by (calibrated p, raw score)"

    # --- Fairness audit ---
    tables = audit(scored)
    assert set(tables) == {"gender", "scholarship", "age_band"}
    for t in tables.values():
        assert (t["n"] > 0).all()
        br = t["benefit_rate"].dropna()
        assert br.between(0, 1).all()
        assert t["population_share"].sum() > 0.99
    concerns = flag_concerns(tables)   # may be empty on synthetic; must not crash
    assert isinstance(concerns, list)

    # --- Drift ---
    df = add_history_features(make_synthetic())
    train, calw, test = three_way_temporal_split(df)
    assert psi(train["age"], train["age"].sample(2000, random_state=0)) < 0.05, \
        "same distribution must read as stable"
    shifted = test.copy()
    shifted["lead_time_days"] = shifted["lead_time_days"] + 14
    assert psi(train["lead_time_days"], shifted["lead_time_days"]) > 0.25, \
        "a 14-day shift must trigger ALERT-level PSI"
    report = drift_report(train, shifted)
    assert any_alert(report), "retrain trigger must fire on shifted window"

    # --- API (in-process, no server needed) ---
    from fastapi.testclient import TestClient
    import api
    client = TestClient(api.app)

    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    r = client.post("/score", json={"age": 25, "lead_time_days": 30,
                                    "prior_appointments": 4, "prior_noshows": 3})
    body = r.json()
    assert r.status_code == 200 and 0 < body["p_noshow"] < 1
    assert body["tier"] in {"sms_only", "extra_reminder", "staff_call"}
    assert "risk" in body["why"]

    api_days = sorted(api.ART["scored_test"]["appointment_day"].dt.date.unique())
    r = client.get(f"/call-list?day={api_days[-1]}&capacity=10")
    assert r.status_code == 200 and len(r.json()["calls"]) <= 10

    r = client.get("/call-list?day=1999-01-01")
    assert r.status_code == 404

    print("high-risk /score example:", body)
    print("\nSMOKE TEST L3 PASSED — ranking, fairness, drift, and API all work.")


if __name__ == "__main__":
    main()
