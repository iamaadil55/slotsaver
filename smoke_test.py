"""End-to-end pipeline check on synthetic data. Run: python smoke_test.py

Passing means: imports work, features are leakage-safe, the temporal split is
ordered, both models train, and metrics compute. It does NOT validate real-data
performance — download the Kaggle CSV for that.
"""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import make_synthetic
from src.evaluate import business_value, comparison_table, evaluate_rule, evaluate_scores
from src.features import (BINARY, CATEGORICAL, FEATURES, LABEL, NUMERIC,
                          add_history_features, rule_baseline, temporal_split)


def main() -> None:
    df = make_synthetic(n=20_000, n_patients=8_000)
    df = add_history_features(df)

    # --- Leakage guards ---
    first = df.groupby("patient_id").head(1)
    assert (first["prior_appointments"] == 0).all(), "first appointment must have no history"
    assert (first["prior_noshows"] == 0).all(), "first appointment must have no prior no-shows"
    assert df["prior_noshows"].le(df["prior_appointments"]).all(), "priors can't exceed count"

    train, test = temporal_split(df, test_frac=0.25)
    assert train["appointment_day"].max() < test["appointment_day"].min(), \
        "temporal split violated: train overlaps test"

    X_train, y_train = train[FEATURES], train[LABEL]
    X_test, y_test = test[FEATURES], test[LABEL]
    K = max(50, int(0.05 * len(test)))

    results = [evaluate_rule("rule baseline", y_test, rule_baseline(test), K)]

    # --- Logistic regression ---
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
    ])
    logreg = Pipeline([("prep", preprocess),
                       ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    logreg.fit(X_train, y_train)
    lr_scores = logreg.predict_proba(X_test)[:, 1]
    results.append(evaluate_scores("logistic regression", y_test, lr_scores, K))

    # --- LightGBM ---
    X_train_lgb, X_test_lgb = X_train.copy(), X_test.copy()
    for c in CATEGORICAL:
        X_train_lgb[c] = X_train_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight="balanced",
                          random_state=42, verbose=-1)
    lgbm.fit(X_train_lgb, y_train)
    lgb_scores = lgbm.predict_proba(X_test_lgb)[:, 1]
    results.append(evaluate_scores("lightgbm", y_test, lgb_scores, K))

    table = comparison_table(results)
    print("\n", table, "\n")

    # Synthetic data has planted signal -> models must comfortably beat coin flips.
    lr_auc = table.loc["logistic regression", "roc_auc"]
    lgb_auc = table.loc["lightgbm", "roc_auc"]
    assert lr_auc > 0.6 and lgb_auc > 0.6, f"models failed to find planted signal ({lr_auc}, {lgb_auc})"

    print(business_value(y_test, lgb_scores, K))
    print("\nSMOKE TEST PASSED — pipeline works end to end.")


if __name__ == "__main__":
    main()
