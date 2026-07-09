"""Drift monitoring: is tomorrow still the world the model was trained on?

Published no-show models lose AUC when deployed (0.93 -> 0.73 in one
prospective study). The main mechanism is distribution shift: booking
behavior, patient mix, or clinic policy changes, and quietly invalidates
the model AND the calibration AND even which constraint (capacity vs
threshold) binds. Monitoring is how you notice before the clinic does.

Tool: PSI (Population Stability Index), the industry-standard drift score.
Rule-of-thumb thresholds: < 0.10 stable · 0.10–0.25 watch · > 0.25 alert.
"""

import numpy as np
import pandas as pd

PSI_WATCH = 0.10
PSI_ALERT = 0.25


def psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%).
    Bins come from the REFERENCE quantiles — the training-time view of normal."""
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference.dropna(), bins=edges)[0] / max(len(reference.dropna()), 1)
    cur_pct = np.histogram(current.dropna(), bins=edges)[0] / max(len(current.dropna()), 1)
    ref_pct = np.clip(ref_pct, 1e-6, None)   # avoid log(0)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status(value: float) -> str:
    if value > PSI_ALERT:
        return "ALERT"
    if value > PSI_WATCH:
        return "watch"
    return "ok"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 features: tuple = ("lead_time_days", "age")) -> pd.DataFrame:
    """Compare a current window against the training reference.

    Also tracks the label base rate when available — base-rate drift breaks
    calibration even when every feature looks stable (isotonic learned the
    OLD relationship between scores and outcomes).
    """
    rows = []
    for f in features:
        v = psi(reference[f], current[f])
        rows.append({"check": f"PSI {f}", "value": round(v, 4), "status": status(v)})
    if "no_show" in reference.columns and "no_show" in current.columns:
        delta = float(current["no_show"].mean() - reference["no_show"].mean())
        # crude but effective: >5 percentage points of base-rate movement
        # means recalibrate regardless of what the features say
        s = "ALERT" if abs(delta) > 0.05 else ("watch" if abs(delta) > 0.02 else "ok")
        rows.append({"check": "base-rate delta", "value": round(delta, 4), "status": s})
    return pd.DataFrame(rows)


def any_alert(report: pd.DataFrame) -> bool:
    """The retrain/recalibrate trigger for the nightly job."""
    return bool((report["status"] == "ALERT").any())
