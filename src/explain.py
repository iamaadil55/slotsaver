"""Per-patient explanations in plain language.

Uses LightGBM's built-in TreeSHAP (`pred_contrib=True`) — the exact algorithm
the `shap` library uses for tree models, without the extra dependency. Each
prediction decomposes into one contribution per feature (in log-odds space):
positive pushes toward "no-show", negative toward "shows up".

IMPORTANT FRAMING (put this in every interview answer): SHAP explains what the
MODEL used, which is correlation, not causation. "Booked 32 days ahead ↑ risk"
means long-lead patients missed more often in the data — not yet proof that
shortening lead times would fix it. Measuring that is an intervention
experiment (Level 3+).
"""

import numpy as np
import pandas as pd


def shap_contributions(model, X: pd.DataFrame):
    """Return (per-feature contributions [n_rows, n_features], bias [n_rows]).
    Contributions are in log-odds space; we only use sign and magnitude."""
    raw = model.booster_.predict(X, pred_contrib=True)
    return raw[:, :-1], raw[:, -1]


def describe(feature: str, value) -> str:
    """One appointment field -> a phrase an office manager understands."""
    if feature == "lead_time_days":
        return f"booked {int(value)}d ahead"
    if feature == "prior_noshows":
        return f"{int(value)} prior no-shows"
    if feature == "prior_noshow_rate":
        return "no visit history" if pd.isna(value) else f"missed {value:.0%} of past visits"
    if feature == "prior_appointments":
        return f"{int(value)} past visits"
    if feature == "sms_received":
        return "SMS sent" if value == 1 else "no SMS sent"
    if feature == "age":
        return f"age {int(value)}"
    if feature == "weekday":
        return str(value)
    if feature == "neighbourhood":
        return f"area {value}"
    if feature == "scholarship":
        return "welfare scholarship" if value == 1 else "no scholarship"
    if feature in ("hypertension", "diabetes", "alcoholism", "handicap"):
        return feature if value == 1 else f"no {feature}"
    return f"{feature}={value}"


def top_reasons_frame(model, X: pd.DataFrame, k: int = 3) -> list[str]:
    """For each row: the k features that pushed this prediction hardest,
    as 'booked 32d ahead (↑ risk) · 2 prior no-shows (↑ risk) · SMS sent (↓ risk)'."""
    contribs, _ = shap_contributions(model, X)
    out = []
    for i in range(len(X)):
        row = X.iloc[i]
        order = np.argsort(-np.abs(contribs[i]))[:k]
        parts = []
        for j in order:
            feat = X.columns[j]
            arrow = "↑" if contribs[i, j] > 0 else "↓"
            parts.append(f"{describe(feat, row[feat])} ({arrow} risk)")
        out.append(" · ".join(parts))
    return out


def global_importance(model, X: pd.DataFrame) -> pd.Series:
    """Mean |contribution| per feature — 'what drives predictions overall'.
    More faithful than split counts because it's in the model's output units."""
    contribs, _ = shap_contributions(model, X)
    return pd.Series(np.abs(contribs).mean(axis=0), index=X.columns).sort_values()
