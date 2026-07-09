"""Level 2a pipeline check on synthetic data. Run: python smoke_test_l2.py"""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.calibrate import (ev_call_threshold, expected_calibration_error,
                           fit_isotonic, risk_tiers, three_way_temporal_split)
from src.data import make_synthetic
from src.features import CATEGORICAL, FEATURES, LABEL, add_history_features


def main() -> None:
    df = add_history_features(make_synthetic(n=20_000, n_patients=8_000))
    train, cal, test = three_way_temporal_split(df)

    # --- Temporal ordering guards ---
    assert train["appointment_day"].max() < cal["appointment_day"].min(), "train overlaps cal"
    assert cal["appointment_day"].max() < test["appointment_day"].min(), "cal overlaps test"

    def to_xy(d):
        X = d[FEATURES].copy()
        for c in CATEGORICAL:
            X[c] = X[c].astype("category")
        return X, d[LABEL]

    X_train, y_train = to_xy(train)
    X_cal, y_cal = to_xy(cal)
    X_test, y_test = to_xy(test)

    balanced = LGBMClassifier(n_estimators=200, class_weight="balanced",
                              random_state=42, verbose=-1).fit(X_train, y_train)
    plain = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1).fit(X_train, y_train)

    p_bal = balanced.predict_proba(X_test)[:, 1]
    p_plain = plain.predict_proba(X_test)[:, 1]
    iso = fit_isotonic(plain.predict_proba(X_cal)[:, 1], y_cal)
    p_iso = iso.predict(p_plain)

    ece_bal = expected_calibration_error(y_test, p_bal)
    ece_iso = expected_calibration_error(y_test, p_iso)
    print(f"ECE balanced={ece_bal:.4f}  plain={expected_calibration_error(y_test, p_plain):.4f}  "
          f"isotonic={ece_iso:.4f}")
    print(f"Brier balanced={brier_score_loss(y_test, p_bal):.4f}  isotonic={brier_score_loss(y_test, p_iso):.4f}")

    # --- Calibration must actually improve honesty, without hurting ranking ---
    assert ece_iso < ece_bal, "isotonic failed to beat balanced-model ECE"
    auc_plain = roc_auc_score(y_test, p_plain)
    auc_iso = roc_auc_score(y_test, p_iso)
    assert abs(auc_plain - auc_iso) < 0.02, "isotonic should preserve ranking (monotonic)"

    # --- Threshold + tiers sanity ---
    assert abs(ev_call_threshold() - 5 / 60) < 1e-9
    tiers = risk_tiers(p_iso)
    assert set(tiers.unique()) <= {"sms_only", "extra_reminder", "staff_call"}

    print("\nSMOKE TEST L2 PASSED — calibration pipeline works end to end.")


if __name__ == "__main__":
    main()
