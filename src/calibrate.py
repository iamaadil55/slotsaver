"""Calibration: making scores mean what they say.

A model is CALIBRATED when, among appointments it gives probability p, roughly
a fraction p actually no-show. Like a weather forecaster: if it rains on 30% of
the days they say "30% rain", they're calibrated.

Why Level 2 needs this: the intervention math is
    expected value of a call = p * success_rate * revenue_per_slot - cost_per_call
That formula is only meaningful if p is a real probability. Our Level 1 model
used class_weight="balanced", which deliberately distorts probabilities to fix
class imbalance — great for ranking, useless for cost math.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def three_way_temporal_split(df: pd.DataFrame, cal_frac: float = 0.15,
                             test_frac: float = 0.25):
    """past -> TRAIN, recent past -> CALIBRATION, future -> TEST.

    Rules enforced here:
    - The calibrator must be fit on data the model never trained on
      (calibrating on training scores just memorizes their bias).
    - The test window is identical to Level 1's (last test_frac of days),
      so every Level 2 number is directly comparable to Level 1.
    """
    test_cut = df["appointment_day"].quantile(1 - test_frac)
    rest = df[df["appointment_day"] < test_cut]
    cal_cut = rest["appointment_day"].quantile(1 - cal_frac)

    train = rest[rest["appointment_day"] < cal_cut]
    cal = rest[rest["appointment_day"] >= cal_cut]
    test = df[df["appointment_day"] >= test_cut]

    print(f"three_way_temporal_split(): train={len(train)} (to {train['appointment_day'].max().date()})  "
          f"cal={len(cal)} ({cal['appointment_day'].min().date()}..{cal['appointment_day'].max().date()})  "
          f"test={len(test)} (from {test['appointment_day'].min().date()})")
    return train, cal, test


def reliability_table(y_true, p_pred, n_bins: int = 10) -> pd.DataFrame:
    """For each probability bin: what the model predicted vs what happened.
    This is the data behind a reliability curve — perfect calibration means
    mean_predicted ≈ actual_rate in every bin."""
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p_pred, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}–{bins[b + 1]:.1f}",
            "mean_predicted": p_pred[mask].mean(),
            "actual_rate": y_true[mask].mean(),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, p_pred, n_bins: int = 10) -> float:
    """ECE: the average gap between 'predicted' and 'actual' across bins,
    weighted by how many predictions land in each bin. 0 = perfectly calibrated.
    Rule of thumb: < 0.02 is good, > 0.05 means the probabilities lie."""
    t = reliability_table(y_true, p_pred, n_bins)
    weights = t["count"] / t["count"].sum()
    return float((weights * (t["mean_predicted"] - t["actual_rate"]).abs()).sum())


def fit_isotonic(p_cal, y_cal) -> IsotonicRegression:
    """Learn a monotonic mapping raw_score -> honest probability on the
    CALIBRATION window. Monotonic = the ranking is preserved exactly; only the
    values are re-labeled. (Platt scaling is the parametric alternative —
    better when the calibration set is tiny; isotonic wins with 10k+ rows.)"""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p_cal), np.asarray(y_cal))
    return iso


def ev_call_threshold(revenue_per_slot: float = 200.0,
                      intervention_success_rate: float = 0.3,
                      cost_per_call: float = 5.0) -> float:
    """Call a patient only if expected value is positive:
        p * success * revenue - cost > 0   =>   p > cost / (success * revenue)
    The threshold is pure economics — no ML in it. The model's job is only to
    supply an honest p. Same assumption warning as business_value(): these
    three numbers must be re-estimated per clinic.
    """
    return cost_per_call / (intervention_success_rate * revenue_per_slot)


def risk_tiers(p, low: float = 0.15, high: float = 0.35) -> pd.Series:
    """Map calibrated probabilities to the product's intervention tiers:
    SMS-only (cheap, everyone low-risk), extra reminder, staff call.
    Tier boundaries are product decisions, not statistics — tune with the clinic.
    """
    p = np.asarray(p)
    return pd.Series(np.select([p < low, p < high], ["sms_only", "extra_reminder"],
                               default="staff_call"))
