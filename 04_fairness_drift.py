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

# %% [markdown] id="9a9a98cd"
# # SlotSaver — Level 3: Fairness audit & drift monitoring
#
# Two questions a responsible deployment must answer continuously:
#
# 1. **Fairness** — are the probabilities equally honest for every group, and
#    does every group's no-shows get a fair share of the help?
# 2. **Drift** — is the world still the one the model was trained on?
#
# (The serving side of Level 3 lives in `api.py` + `Dockerfile` +
# `nightly_job.py`; this notebook is the analysis side.)

# %% colab={"base_uri": "https://localhost:8080/"} id="42166d42" outputId="e0d7a356-104b-40f6-f0ad-484d5e03c66f"
import sys
if "google.colab" in sys.modules:          # this cell does nothing on your PC
    # !unzip -q -o slotsaver.zip
    # %cd slotsaver
    # !mkdir -p data
    # !cp /content/noshowappointments*.csv data/KaggleV2-May-2016.csv
    # %pip install -q lightgbm

# %% id="9b9a018f"
import pandas as pd

from src.calibrate import three_way_temporal_split
from src.data import load_or_synthesize
from src.fairness import add_age_band, audit, flag_concerns
from src.features import add_history_features
from src.monitor import any_alert, drift_report, psi
from src.pipeline import build_scoring_artifacts

pd.set_option("display.max_columns", 20)

# %% colab={"base_uri": "https://localhost:8080/"} id="64366300" outputId="15c8a973-e235-48fd-ae9f-2334945d8b49"
art = build_scoring_artifacts("data/KaggleV2-May-2016.csv")
scored = art["scored_test"]

# %% [markdown] id="d6b9b281"
# ## 1. Fairness audit
#
# Reading guide for each table:
#
# - `calibration_gap` = mean predicted − actual. Near 0 is honest. A group at
#   +0.05 is being systematically over-flagged; at −0.05, under-served.
# - `benefit_rate` = of this group's *actual* no-shows, the share placed in the
#   staff_call tier (strongest help). Large gaps between groups = unequal help.
# - `call_tier_share` vs `population_share` = over/under-representation on the
#   call list. Differing base rates can justify differences — but you must be
#   able to say *why*, out loud, before shipping.

# %% colab={"base_uri": "https://localhost:8080/", "height": 163} id="749fc667" outputId="fd4211c3-3b5c-4546-e888-77ebd3247ec2"
tables = audit(scored)
tables["gender"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 163} id="b816b3bd" outputId="2352cc93-5f6a-4e87-b791-329722a1e59b"
tables["scholarship"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 257} id="22b8d0c1" outputId="112200b7-4522-4087-fbab-ce5a5f30a374"
tables["age_band"]

# %% [markdown] id="a93964dd"
# ### Automatic flags
#
# Thresholds (|calibration_gap| > 0.05, min/max benefit ratio < 0.5) are review
# triggers, not verdicts — every flag demands a human explanation.

# %% colab={"base_uri": "https://localhost:8080/"} id="3bc7c6c9" outputId="c7c07ced-22d6-4381-d444-c0ba12ff9c7a"
concerns = flag_concerns(tables)
if concerns:
    for c in concerns:
        print("⚠", c)
else:
    print("No automatic flags at current thresholds — still read the tables; "
          "thresholds are conventions, not guarantees.")

# %% [markdown] id="300b9593"
# ## 2. Drift monitoring
#
# PSI compares distributions against the training reference. Thresholds:
# < 0.10 stable · 0.10–0.25 watch · > 0.25 alert. First: the honest check of
# our own test window against training — some drift is already visible in any
# real system.

# %% colab={"base_uri": "https://localhost:8080/"} id="6ec3273e" outputId="b07a5680-df12-4342-df32-eb57307d9ce2"
df, _ = load_or_synthesize("data/KaggleV2-May-2016.csv")
df = add_history_features(df)
train, cal, test = three_way_temporal_split(df)

report = drift_report(train, test)
print(report.to_string(index=False))
print("\nRetrain trigger fires:", any_alert(report))

# %% [markdown] id="bd7e9518"
# ### Simulated future drift — what an alert looks like
#
# Suppose the clinic changes its booking policy and lead times stretch by two
# weeks. Watch PSI catch it:

# %% colab={"base_uri": "https://localhost:8080/"} id="63fb9e67" outputId="e0737d32-a683-4009-9e99-224ff6e01490"
shifted = test.copy()
shifted["lead_time_days"] = shifted["lead_time_days"] + 14

print(f"PSI lead_time (normal window) : {psi(train['lead_time_days'], test['lead_time_days']):.4f}")
print(f"PSI lead_time (+14d shift)    : {psi(train['lead_time_days'], shifted['lead_time_days']):.4f}  <- ALERT")

# %% [markdown] id="9a067167"
# When this alert fires in `nightly_job.py`, the playbook is: retrain the
# model, refit the calibrator (calibration breaks FIRST under drift), and
# re-check which constraint — capacity or threshold — now binds.
#

# %% [markdown] id="0c2d34fb"
# ## 3. Conclusions
#
# Based on the analysis performed in this notebook, here are the key takeaways for the responsible deployment of SlotSaver:
#
# ### Fairness Audit Findings
# - **Calibration**: The `46-65` and `0-12` age bands show the largest positive calibration gaps (~0.023–0.026), meaning the model slightly over-predicts their no-show probability. While below the 0.05 alert threshold, these groups are being 'over-flagged' for intervention relative to their actual behavior.
# - **Benefit Distribution**: There is a significant disparity in `benefit_rate` across age bands. The `13-25` group has a benefit rate of 0.36, while the `65+` group is at only 0.10. This indicates that younger no-shows are much more likely to be placed in the high-touch 'staff_call' tier than elderly no-shows. This warrants a review of whether the model features (like 'lead time') are unfairly penalizing specific demographics.
#
# ### Drift & Reliability
# - **Baseline Stability**: The current test window shows stable PSI for features like `age` and `lead_time_days`, though the `base-rate delta` sits at 'watch' status, suggesting a slight shift in the overall no-show frequency.
# - **Drift Sensitivity**: The simulation demonstrates that a 14-day shift in booking lead times causes the PSI to spike to >9.0, triggering an immediate alert.
# - **Operational Strategy**: Calibration is the most fragile part of the pipeline under drift. When drift alerts fire, the immediate playbook must include refitting the Isotonic Regressor to ensure the probabilities shown to clinic staff remain 'honest' even if the underlying feature distributions have changed.
