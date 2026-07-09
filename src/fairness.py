"""Fairness audit: does the system treat demographic groups even-handedly?

Framing matters. In this product a high risk score triggers HELP (reminders,
transport info, easier rescheduling) — so the fairness question is inverted
from the usual credit-scoring one. We are not asking "who gets denied?" but
"does every group get its fair share of help, and are the probabilities
equally honest for everyone?"

Three checks per group:
- calibration_gap: mean predicted p minus actual no-show rate. If the model
  is honest for one group and inflated for another, decisions are skewed.
- benefit_rate: among the group's ACTUAL no-shows, what fraction landed in
  the staff_call tier (i.e. would have received the strongest help)? This is
  an equal-opportunity-style metric, pointed at benefit instead of harm.
- call_share vs population_share: is the group over/under-represented on the
  call list relative to its size? Not automatically bad (base rates differ),
  but a flag to investigate.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AGE_BINS = [-1, 12, 25, 45, 65, 200]
AGE_LABELS = ["0-12", "13-25", "26-45", "46-65", "65+"]


def add_age_band(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["age_band"] = pd.cut(scored["age"], AGE_BINS, labels=AGE_LABELS)
    return scored


def fairness_table(scored: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """One row per group value; scored must have no_show, p_noshow, tier."""
    rows = []
    total = len(scored)
    total_calls = (scored["tier"] == "staff_call").sum()
    for value, g in scored.groupby(group_col, observed=True):
        if len(g) == 0:
            continue
        in_call_tier = g["tier"] == "staff_call"
        noshows = g[g["no_show"] == 1]
        # AUC within the group is only defined if both outcomes occur
        auc = (round(roc_auc_score(g["no_show"], g["p_noshow"]), 3)
               if g["no_show"].nunique() == 2 else np.nan)
        rows.append({
            group_col: value,
            "n": len(g),
            "population_share": round(len(g) / total, 3),
            "actual_noshow_rate": round(g["no_show"].mean(), 3),
            "mean_predicted_p": round(g["p_noshow"].mean(), 3),
            "calibration_gap": round(g["p_noshow"].mean() - g["no_show"].mean(), 3),
            "auc_within_group": auc,
            "call_tier_share": round(in_call_tier.sum() / total_calls, 3) if total_calls else np.nan,
            "benefit_rate": round((noshows["tier"] == "staff_call").mean(), 3) if len(noshows) else np.nan,
        })
    return pd.DataFrame(rows).set_index(group_col)


def audit(scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Full audit across the three sensitive dimensions available in this data.
    (Race/income are not in the dataset — but neighbourhood and scholarship
    can proxy for them, which is exactly why they belong in the audit.)"""
    scored = add_age_band(scored)
    return {
        "gender": fairness_table(scored, "gender"),
        "scholarship": fairness_table(scored, "scholarship"),
        "age_band": fairness_table(scored, "age_band"),
    }


def flag_concerns(tables: dict[str, pd.DataFrame], calib_gap_limit: float = 0.05,
                  benefit_ratio_limit: float = 0.5) -> list[str]:
    """Turn tables into plain-language flags a reviewer can act on."""
    concerns = []
    for dim, t in tables.items():
        bad_calib = t[t["calibration_gap"].abs() > calib_gap_limit]
        for idx in bad_calib.index:
            concerns.append(f"{dim}={idx}: calibration gap "
                            f"{t.loc[idx, 'calibration_gap']:+.3f} (probabilities dishonest for this group)")
        br = t["benefit_rate"].dropna()
        if len(br) >= 2 and br.max() > 0 and (br.min() / br.max()) < benefit_ratio_limit:
            concerns.append(f"{dim}: benefit_rate ranges {br.min():.2f}–{br.max():.2f} — "
                            "some groups' no-shows get far less help than others")
    return concerns
