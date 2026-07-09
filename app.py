"""SlotSaver — daily call-list dashboard. Run locally:  streamlit run app.py

Works without the Kaggle CSV (synthetic demo mode, clearly flagged).
"""

import streamlit as st

from src.pipeline import build_scoring_artifacts, daily_call_list

st.set_page_config(page_title="SlotSaver — Daily Call List", layout="wide")


@st.cache_resource(show_spinner="Training + calibrating model (first load only)...")
def get_artifacts():
    return build_scoring_artifacts()


art = get_artifacts()
scored = art["scored_test"]
days = sorted(scored["appointment_day"].dt.date.unique())

st.title("SlotSaver — daily no-show call list")
if not art["is_real"]:
    st.warning("Running on SYNTHETIC data (no Kaggle CSV in data/). Demo mode only.")

with st.sidebar:
    day = st.selectbox("Clinic day", days, index=len(days) - 1)
    capacity = st.slider("Staff call capacity (calls/day)", 5, 100, 20, step=5)
    st.subheader("Economics — explicit assumptions")
    revenue = st.number_input("Revenue per slot ($)", 50, 1000, 200, step=25)
    success = st.slider("Call success rate", 0.05, 0.90, 0.30, step=0.05)
    cost = st.number_input("Cost per call ($)", 1, 100, 5, step=1)
    show_actual = st.checkbox("Show actual outcomes (evaluation mode)", value=False,
                              help="Only possible on historical data — a live clinic wouldn't have this column.")

day_df, summary = daily_call_list(art, day, capacity, cost, success, revenue)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Appointments", summary["appointments"])
c2.metric("Expected no-shows", f"{summary['expected_noshows']:.0f}")
c3.metric("Staff calls today", summary["calls"])
c4.metric("Expected value of calls", f"${summary['expected_value_usd']:,.0f}")

st.caption(f"EV threshold at these economics: p* = {summary['ev_threshold']:.3f} "
           f"(call only when p × {success:.0%} × ${revenue} beats the ${cost} call cost). "
           f"Capacity caps the list at {capacity}.")

TIER_BADGE = {"staff_call": "🔴 staff call", "extra_reminder": "🟡 extra reminder",
              "sms_only": "🟢 SMS only"}

view = day_df.assign(
    call=day_df["call_today"].map({True: "📞 CALL", False: ""}),
    risk=(day_df["p_noshow"] * 100).round(1).astype(str) + "%",
    tier_label=day_df["tier"].map(TIER_BADGE),
)[["call", "risk", "tier_label", "why", "appts_today", "age", "gender",
   "lead_time_days", "patient_id", "appointment_id"]]

if show_actual:
    view["actual (eval only)"] = day_df["no_show"].map({1: "❌ no-show", 0: "✅ showed"})

st.dataframe(view, width="stretch", hide_index=True)

st.divider()
st.caption(
    "Ethics: risk scores exist to help patients attend — reminders, easier rescheduling, "
    "transport information — never to deny or deprioritize care. Probabilities are "
    "isotonic-calibrated on a held-out window. Reasons are model correlations (TreeSHAP), "
    "not proven causes. Economics inputs are assumptions to re-estimate per clinic."
)
