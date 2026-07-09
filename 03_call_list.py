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

# %% [markdown] id="8a54449f"
# # SlotSaver — Level 2b: The call list, with reasons
#
# Everything so far produced *numbers*. This notebook produces the *product*:
# for one clinic day, a ranked call list where every row carries a calibrated
# probability, an intervention tier, and a plain-language "why".
#
# (The interactive version is `app.py` — run `streamlit run app.py` locally.
# This notebook is the GitHub-viewable equivalent.)

# %% colab={"base_uri": "https://localhost:8080/"} id="ef8ef574" outputId="fbe92344-16dc-4fba-c6f7-594fc8ec203d"
import sys
if "google.colab" in sys.modules:          # this cell does nothing on your PC
    # !unzip -q -o slotsaver.zip
    # %cd slotsaver
    # !mkdir -p data
    # !cp /content/noshowappointments*.csv data/KaggleV2-May-2016.csv
    # %pip install -q lightgbm

# %% id="4034b015"
import matplotlib.pyplot as plt
import pandas as pd

from src.explain import global_importance, shap_contributions, describe
from src.pipeline import build_scoring_artifacts, daily_call_list

pd.set_option("display.max_colwidth", 120)

# %% [markdown] id="ca9b17ac"
# ## 1. Build the scoring artifacts
#
# One call — same function the dashboard uses. Under the hood it repeats the
# whole Level 1+2a story: leakage-safe features → temporal split → plain
# LightGBM → isotonic calibration → tiers.

# %% colab={"base_uri": "https://localhost:8080/"} id="ec92062a" outputId="db987d71-d74d-4fcf-be98-e1fd05b48db0"
art = build_scoring_artifacts("data/KaggleV2-May-2016.csv")
scored = art["scored_test"]
days = sorted(scored["appointment_day"].dt.date.unique())
print(f"Test window days available: {days[0]} .. {days[-1]}")

# %% [markdown] id="f0547864"
# ## 2. One clinic day, as the office manager would see it
#
# Decision logic: economics decides who's *worth* calling (p > p*), capacity
# decides who actually *gets* called (top 20 by risk), tiers cover everyone else.

# %% colab={"base_uri": "https://localhost:8080/"} id="43abc2ef" outputId="f6f47cbd-4213-45eb-bb6f-cefbeca34873"
day = days[-1]
day_df, summary = daily_call_list(art, day, capacity=20)
print(f"Clinic day {day}:")
for k, v in summary.items():
    print(f"  {k:>26}: {v:.2f}" if isinstance(v, float) else f"  {k:>26}: {v}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 833} id="2ce68dd5" outputId="46c91d11-95ea-4337-d35b-d43d9dd8628c"
cols = ["call_today", "p_noshow", "tier", "why", "age", "lead_time_days"]
day_df[cols].head(25)

# %% [markdown] id="1ca505ca"
# ## 3. What drives predictions overall (global view)
#
# Mean |TreeSHAP contribution| per feature — more faithful than split counts
# because it measures impact on the actual prediction.

# %% colab={"base_uri": "https://localhost:8080/", "height": 457} id="0ad53cb8" outputId="1e3ce07f-d1c6-4348-ac50-2a376010413b"
global_importance(art["model"], art["X_test"]).plot.barh(
    figsize=(7, 4.5), color="#55A868", title="What the model actually uses (mean |SHAP|)")
plt.tight_layout()
plt.show()

# %% [markdown] id="d3aea7f0"
# ## 4. One patient, fully explained (local view)
#
# The top row of today's call list, decomposed feature by feature. This is the
# transparency a clinic needs before trusting the tool — and what regulators
# increasingly expect from decision-support systems.

# %% colab={"base_uri": "https://localhost:8080/", "height": 505} id="aa8ab3d6" outputId="56a091b6-433d-40c1-bbcc-90491d0a76ad"
top_idx = day_df.index[0]
x_row = art["X_test"].loc[[top_idx]]
contribs, bias = shap_contributions(art["model"], x_row)

breakdown = (pd.DataFrame({
    "feature": x_row.columns,
    "value": [describe(f, x_row.iloc[0][f]) for f in x_row.columns],
    "contribution_logodds": contribs[0].round(3),
}).sort_values("contribution_logodds", key=abs, ascending=False))
print(f"Patient at top of the {day} call list — "
      f"calibrated p(no-show) = {day_df.loc[top_idx, 'p_noshow']:.1%}")
breakdown

# %% [markdown] id="e99280cb"
# ## 5. Conclusions
#
# **Two patients from today's call list**
#
# - **Patient 90538 (p = 99%):** flagged because of 15 prior no-shows and a 100% historical miss rate, even though booking same-day usually lowers risk — history outweighs lead time here. Age 9 means the call goes to their parent or guardian.
# - **Patient 102802 (p = 80%):** flagged mainly because they booked 176 days ahead and have missed 100% of past visits. Given that history, a better intervention than a reminder might be rescheduling to a sooner date or arranging transportation assistance, since the long lead time is a primary driver of their risk.
#
# The feature **"Booked 32 days ahead (↑ risk)"** represents a correlation that the model learned from historical data. However, correlation does not prove causation. We cannot conclude that simply reducing the booking lead time will reduce no-show rates because other hidden factors may influence both appointment timing and patient attendance. To establish causation, a randomized controlled experiment (A/B test) would be required, where similar patients are randomly assigned to different booking lead times and their no-show rates are compared.
#
# The dashboard reports **expected value** using only the model's predicted probabilities rather than the actual appointment outcomes. This is the correct approach for deployment because, in real-world use, future labels are unknown at the time decisions are made. Using predicted probabilities provides an honest estimate of the expected benefit of calling patients before the appointment occurs.
#
# This approach has several limitations. The generated reasons explain the model's predictions, not the patient's actual thoughts or intentions. The intervention tier boundaries are still product design choices and may need refinement based on operational feedback. Finally, the evaluation uses a held-out test window as a substitute for future appointments. A live production system with continuous data collection and monitoring would be required to fully validate performance in real-world clinical settings.
#
