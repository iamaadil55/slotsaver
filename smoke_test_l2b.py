"""Level 2b pipeline check on synthetic data. Run: python smoke_test_l2b.py"""

import ast

import numpy as np

from src.explain import global_importance, top_reasons_frame
from src.pipeline import build_scoring_artifacts, daily_call_list


def main() -> None:
    art = build_scoring_artifacts("data/does_not_exist.csv")  # forces synthetic
    scored = art["scored_test"]
    assert scored["p_noshow"].between(0.01, 0.99).all(), \
        "probs must be clipped — never claim certainty about future behavior"

    days = sorted(scored["appointment_day"].dt.date.unique())
    day_df, summary = daily_call_list(art, days[-1], capacity=20)

    # --- Decision logic guards ---
    assert day_df["patient_id"].is_unique, "call list is per patient, never per appointment"
    assert (day_df["appts_today"] >= 1).all()
    assert day_df["p_noshow"].is_monotonic_decreasing, "call list must be ranked by risk"
    assert summary["calls"] <= 20, "capacity must cap the call list"
    assert day_df.loc[day_df["call_today"], "worth_calling"].all(), \
        "never call below the EV threshold"
    called_p = day_df.loc[day_df["call_today"], "p_noshow"]
    uncalled_worth = day_df.loc[day_df["worth_calling"] & ~day_df["call_today"], "p_noshow"]
    if len(called_p) and len(uncalled_worth):
        assert called_p.min() >= uncalled_worth.max() - 1e-9, \
            "called patients must be the highest-risk of the worth-calling set"

    # --- Explanations ---
    assert day_df["why"].str.len().gt(0).all(), "every row needs a why-string"
    assert day_df["why"].str.contains("risk").all(), "why-strings must show direction"
    reasons = top_reasons_frame(art["model"], art["X_test"].head(5), k=3)
    assert len(reasons) == 5 and all(r.count("·") == 2 for r in reasons), "top-3 format"

    gi = global_importance(art["model"], art["X_test"].head(2000))
    assert len(gi) == art["X_test"].shape[1] and (gi.values >= 0).all()

    # --- EV summary sanity ---
    assert np.isfinite(summary["expected_value_usd"])
    assert abs(summary["ev_threshold"] - 5 / 60) < 1e-9

    # --- Dashboard file must at least be valid Python ---
    ast.parse(open("app.py").read())

    print(f"call list day={days[-1]}: {summary['appointments']} appts, "
          f"{summary['calls']} calls, EV=${summary['expected_value_usd']:.0f}")
    print("example why:", day_df["why"].iloc[0])
    print("\nSMOKE TEST L2B PASSED — call-list pipeline works end to end.")


if __name__ == "__main__":
    main()
