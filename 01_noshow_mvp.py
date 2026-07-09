# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="21351492"
# # SlotSaver — Level 1: Can we predict which patients won't show up?
#
# **The business question:** clinic staff can call ~20 high-risk patients a day.
# Which 20 should they call?
#
# **Pipeline:** load → clean → EDA → leakage-safe features → temporal split →
# rule baseline → logistic regression → LightGBM → business translation.
#
# Data: [Kaggle Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
# — place `KaggleV2-May-2016.csv` in `data/`. Without it, this notebook runs on
# synthetic data (fine for testing the pipeline, meaningless for conclusions).

# %% colab={"base_uri": "https://localhost:8080/"} id="XxoTFQo0_gRW" outputId="2855e97a-acb1-4671-877f-93cefe8e6394"
import sys
if "google.colab" in sys.modules:          # this cell does nothing on your PC
    # !unzip -q -o slotsaver.zip
    # %cd slotsaver
    # !mkdir -p data
    # !cp /content/noshowappointments*.csv data/KaggleV2-May-2016.csv
    # %pip install -q lightgbm

# %% id="24507653"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_or_synthesize
from src.features import (BINARY, CATEGORICAL, FEATURES, LABEL, NUMERIC,
                          add_history_features, rule_baseline, temporal_split)
from src.evaluate import (business_value, comparison_table, evaluate_rule,
                          evaluate_scores)

pd.set_option("display.max_columns", 40)

# %% [markdown] id="8c172a9d"
# ## 1. Load and clean
#
# Cleaning drops impossible records (negative ages, appointments "scheduled"
# after they happened) and prints exactly what it dropped. Never clean silently.

# %% colab={"base_uri": "https://localhost:8080/", "height": 285} id="80871749" outputId="823d3298-6e3b-404b-9550-6cb234fb61c3"
df, is_real = load_or_synthesize("data/KaggleV2-May-2016.csv")
print(f"\nRows: {len(df)}   Real data: {is_real}")
print(f"No-show base rate: {df[LABEL].mean():.1%}")
df.head(3)

# %% [markdown] id="a28d6d93"
# ## 2. EDA — the two relationships that matter most
#
# Everything else in the EDA notebook graveyard is optional; these two drive the model.

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="caa83f1f" outputId="4e3163f8-49f9-4a23-d99c-24512611e07a"
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (a) No-show rate vs how far ahead the appointment was booked
buckets = pd.cut(df["lead_time_days"], [-1, 0, 3, 7, 14, 30, 200],
                 labels=["same day", "1–3d", "4–7d", "8–14d", "15–30d", ">30d"])
df.groupby(buckets, observed=True)[LABEL].mean().plot.bar(ax=axes[0], color="#4C72B0")
axes[0].set_title("No-show rate by booking lead time")
axes[0].set_ylabel("no-show rate")

# (b) No-show rate vs the patient's own track record (computed leakage-safe below,
#     but we can preview it with a quick history count)
tmp = add_history_features(df)
hist = tmp["prior_noshows"].clip(0, 3)
tmp.groupby(hist)[LABEL].mean().plot.bar(ax=axes[1], color="#C44E52")
axes[1].set_title("No-show rate by # prior no-shows (3 = 3+)")
plt.tight_layout()
plt.show()

# %% [markdown] id="5f313766"
# Typical finding on the real data: same-day bookings almost always show up;
# risk climbs steeply with lead time and with the patient's own no-show history.
# This is why the rule baseline uses exactly these two signals.

# %% [markdown] id="26bc465a"
# ## 3. Features (leakage-safe) and temporal split
#
# Two rules, enforced in `src/features.py`:
#
# 1. `prior_*` features count only appointments strictly before the current one.
# 2. Test set = the last ~25% of appointment days. The model trains on the past
#    and is judged on the future, exactly like deployment.

# %% colab={"base_uri": "https://localhost:8080/"} id="a5888a6d" outputId="e8912d77-8b12-45bd-8836-456f60859942"
df = add_history_features(df)
train, test = temporal_split(df, test_frac=0.25)

X_train, y_train = train[FEATURES], train[LABEL]
X_test, y_test = test[FEATURES], test[LABEL]

# Sanity check the leakage guard: a patient's first-ever appointment must have zero history.
first_appts = df.groupby("patient_id").head(1)
assert (first_appts["prior_appointments"] == 0).all()
assert (first_appts["prior_noshows"] == 0).all()
print("Leakage guard passed: first appointments have zero history.")

# %% [markdown] id="2978ba95"
# ## 4. How we keep score
#
# - **k** = number of patients staff can call per test period (here: 20/day equivalent).
# - **precision@k**: of the k we flag, how many truly no-show. This is the metric
#   the office manager feels.
# - **PR-AUC** for overall ranking quality on an imbalanced label. ROC-AUC for
#   comparability with published studies.
# - **Accuracy does not appear.** With an ~80/20 split it rewards predicting
#   "everyone shows up".

# %% colab={"base_uri": "https://localhost:8080/"} id="64b2b689" outputId="e71092f6-386a-479c-f21c-37d003512746"
n_test_days = test["appointment_day"].nunique()
K = 20 * n_test_days  # "20 calls per clinic day" scaled to the whole test window
print(f"Test window: {n_test_days} days -> k = {K} calls")

# %% [markdown] id="355565d2"
# ## 5. Baseline 1 — rules (no ML)
#
# "Flag if the patient missed ≥2 appointments before, or booked >14 days ahead."

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="4bd4e1f2" outputId="c3a55b7d-fe7d-4815-a2e6-7b7f41ba375b"
results = [evaluate_rule("rule baseline", y_test, rule_baseline(test), K)]
comparison_table(results)

# %% [markdown] id="05fe5399"
# ## 6. Baseline 2 — logistic regression
#
# The field's workhorse (used in ~68% of published no-show studies). Linear,
# fast, interpretable coefficients.

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} id="d16accb5" outputId="3ef22725-7c2f-4fb5-8425-8cabd41194ae"
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), NUMERIC),
    ("bin", "passthrough", BINARY),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
])

logreg = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
logreg.fit(X_train, y_train)
lr_scores = logreg.predict_proba(X_test)[:, 1]
results.append(evaluate_scores("logistic regression", y_test, lr_scores, K))
comparison_table(results)

# %% [markdown] id="faa4edc1"
# ## 7. Main model — LightGBM
#
# Gradient-boosted trees: the strongest practical choice for tabular data.
# Handles non-linear interactions (e.g. "long lead time matters more for young
# patients") and missing values natively — no imputation needed.

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="d9f49f0b" outputId="71024039-f819-48cb-fe55-3bc10fb29492"
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for c in CATEGORICAL:
    X_train_lgb[c] = X_train_lgb[c].astype("category")
    X_test_lgb[c] = X_test_lgb[c].astype("category")

lgbm = LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=31,
    class_weight="balanced", random_state=42, verbose=-1,
)
lgbm.fit(X_train_lgb, y_train)
lgb_scores = lgbm.predict_proba(X_test_lgb)[:, 1]
results.append(evaluate_scores("lightgbm", y_test, lgb_scores, K))
comparison_table(results)

# %% [markdown] id="7faf079e"
# ## 8. What drives the predictions?

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="6589a8bc" outputId="eebcc1cc-57fa-4d49-f05b-30c35f1ce1c3"
imp = pd.Series(lgbm.feature_importances_, index=X_train_lgb.columns).sort_values()
imp.plot.barh(figsize=(7, 4), color="#55A868", title="LightGBM feature importance (split count)")
plt.tight_layout()
plt.show()

# %% [markdown] id="382cde85"
# ## 9. Business translation — the slide that matters
#
# Metrics don't convince office managers; recovered slots do. Every parameter
# below is an explicit assumption — challenge them, don't hide them.

# %% colab={"base_uri": "https://localhost:8080/"} id="10750108" outputId="f7ff4aad-8d4d-4e68-81c2-ac12c16212cc"
bv = business_value(y_test, lgb_scores, K)
for k_, v in bv.items():
    print(f"{k_:>28}: {v}")

# %% [markdown] id="c803783f"
# ## 10. Conclusions and honest limitations
#
# **Conclusions (real data, temporal split):**
#
# - **LightGBM beat logistic regression by enough to matter:** ROC-AUC 0.730 vs 0.681,
#   and precision@140 of 0.621 vs 0.536. At the clinic's call budget that difference is
#   ~12 extra true no-shows caught per week, so the added complexity pays for itself.
#   LightGBM is the model we carry forward.
# - **Both models beat the rule baseline where it counts.** The rule flags 10,140
#   appointments — a third of the test set, a call list no clinic can work — at 28.6%
#   precision. The model compresses that into 140 targeted calls at 62.1% precision,
#   3× the 20.2% base rate.
# - **What it means at k=140:** if staff call the model's top 20 patients each day,
#   ~12–13 of those calls reach true would-be no-shows. Over the 7-day test window that
#   is ~87 no-shows reached → ~26 recovered slots → ~$5,220 recovered revenue against
#   ~$700 outreach cost (assuming 30% of reached patients convert and $200/slot — both
#   assumptions must be re-estimated per clinic).
#
# **Limitations:** 6 weeks of history → most patients have no prior visits, so
# `prior_*` features are sparse; retrospective ≠ prospective (published live
# deployments lose AUC); intervention effect (does a call actually change
# behavior?) is NOT measured here — that's uplift modeling, Level 3.
#
# **Next (Level 2):** probability calibration, cost-sensitive threshold choice,
# SHAP explanations per patient, Streamlit dashboard with the daily call list.
#
