"""Nightly batch job (simulated): score tomorrow, log it, check for drift.

Run: python nightly_job.py
In production this runs on a scheduler (cron / Task Scheduler / Airflow):
  1. score tomorrow's appointments -> write the call list where staff see it
  2. append predictions to a log (future evaluation needs them!)
  3. compare the recent window against the training reference -> drift report
  4. if any ALERT: retrain/recalibrate (here: we just say so loudly)
"""

from pathlib import Path

from src.monitor import any_alert, drift_report
from src.pipeline import build_scoring_artifacts, daily_call_list, to_xy  # noqa: F401
from src.data import load_or_synthesize
from src.features import add_history_features
from src.calibrate import three_way_temporal_split

OUT = Path("logs")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    art = build_scoring_artifacts()
    scored = art["scored_test"]

    # 1+2. "Tomorrow" = last test day. Score, save the call list + full log.
    day = sorted(scored["appointment_day"].dt.date.unique())[-1]
    day_df, summary = daily_call_list(art, day, capacity=20)
    day_df[["patient_id", "appointment_id", "p_noshow", "tier", "call_today", "why"]] \
        .to_csv(OUT / f"call_list_{day}.csv", index=False)
    print(f"[nightly] {day}: {summary['unique_patients']} patients scored, "
          f"{summary['calls']} calls -> logs/call_list_{day}.csv")

    # 3. Drift: training reference vs the most recent scored window.
    df, _ = load_or_synthesize()
    df = add_history_features(df)
    train, _, test = three_way_temporal_split(df)
    report = drift_report(train, test)
    report.to_csv(OUT / f"drift_report_{day}.csv", index=False)
    print(report.to_string(index=False))

    # 4. The trigger.
    if any_alert(report):
        print("[nightly] DRIFT ALERT -> schedule retrain + recalibration, "
              "and re-check which constraint (capacity/threshold) binds.")
    else:
        print("[nightly] no drift alert — model and calibration still valid.")


if __name__ == "__main__":
    main()
