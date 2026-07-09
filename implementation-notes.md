# Implementation Notes — SlotSaver

# Level 2b — Explanations + call-list dashboard (2026-07-06)

## Goal
Per-patient plain-language reasons (TreeSHAP) and the Streamlit daily call list
combining calibrated risk, EV threshold, capacity, and tiers.

## Key decisions
- **LightGBM's built-in `pred_contrib=True` instead of the `shap` package** — identical TreeSHAP values, zero extra dependency/install pain; `shap` can be added later purely for fancy plots.
- **`src/pipeline.py` as single source of truth** — notebook 03, dashboard, and future Level 3 API all call the same build; modeling changes propagate everywhere at once.
- **Call decision = EV threshold ∩ capacity ∩ risk ranking** — encodes the Level 2a finding that capacity binds at cheap-call economics while the threshold binds at expensive-call economics.
- **Expected value on the dashboard uses probabilities, not labels** — what a clinic could compute before the day happens; actual outcomes shown only behind an "evaluation mode" toggle.
- **Correlation-not-causation warning baked into explain.py docstring, notebook 03, and the app footer** — SHAP explains the model, not the patient.

## Deviations
- None from the approved spec (tier badges, calibrated probability, top-3 whys).

## Post-release fixes from Aadil's real-data run (2026-07-06)
- **p = 1.000 appeared on the call list** — isotonic's top bin on the calibration
  window was pure no-shows, so it mapped extreme scores to certainty. Fix: clip
  calibrated probabilities to [0.01, 0.99] in the pipeline. A dashboard must never
  claim 100% about future human behavior.
- **One child (age 9, 15+ prior no-shows) held 5 of the top 20 call slots** — five
  same-day appointments, each scored as a separate row. Fix: daily_call_list now
  deduplicates per patient (keeps highest-risk row, adds appts_today count);
  capacity now buys 20 *patients*, not 20 rows. Lesson: always read the actual
  rows your model outputs — neither smoke tests nor metrics caught this.

## Open questions
- Streamlit runs locally only for now; hosting (HF Spaces) is a Level 3 deployment item.
- Isotonic produces coarse steps at the high end (15 of 20 called patients tie at
  p=0.533 in the 2016-06-08 run) — sparse calibration data up there. Improvement for
  Level 3: rank ties by raw model score, display the calibrated value.
- Reason strings for `neighbourhood` leak raw area names — fine on public data, review before any real deployment.

## Verification
- `python smoke_test_l2b.py` (synthetic): probs in [0,1], list ranked, capacity cap,
  never-call-below-threshold, called = highest-risk of worth-calling set, why-strings
  formatted with direction, global importance valid, app.py parses — PASSED.
- `app.py` executed end-to-end in bare mode (widgets at defaults) — ran clean.

---

# Level 2a — Calibration + cost-sensitive threshold (2026-07-06)

## Goal
Honest probabilities on top of the L1 model: 3-way temporal split, reliability
analysis, isotonic calibration, EV-based call threshold, product risk tiers.

## Key decisions
- **Same test window as Level 1** (last 25% of appointment days) so all numbers stay comparable.
- **Calibration window carved from the END of the training period** (temporal correctness: fit calibrator on data the model never saw, never on the test set).
- **Plain LightGBM + isotonic** rather than calibrating the class_weight="balanced" model — the balanced trick exists only to help ranking, and plain LGBM ranks ~identically here; keeps the story simple.
- **Isotonic over Platt**: calibration window has ~12k rows on real data, enough for non-parametric.
- **Threshold is economics, not statistics**: p* = cost/(success × revenue); shown at $5/$15/$30 call costs to demonstrate capacity vs threshold binding.

## Deviations
- None from the approved Level 2 scope; SHAP + Streamlit dashboard deferred to Level 2b as instructed ("Level 2: calibration").

## Open questions
- Recalibration cadence under drift — revisit with monitoring at Level 3.
- Tier boundaries (0.15/0.35) are product guesses; need clinic input or sensitivity analysis.

## Verification
- `python smoke_test_l2.py` (synthetic): temporal ordering guards, ECE balanced=0.188 →
  isotonic=0.009, AUC preserved within 0.02, threshold math exact, tiers valid — PASSED in sandbox.

---

# Level 1 — MVP

## Goal
Scaffold the Level 1 MVP per the approved deep-dive plan: leakage-safe features,
temporal split, rule baseline → LR → LightGBM, business-framed evaluation.

## Key decisions
- **Thin notebook, logic in `src/`** — testable, reusable at Level 2+, and reads better in a portfolio.
- **Notebook authored in jupytext percent format** and converted to `.ipynb` (both kept in sync).
- **No SMOTE.** Class weights + threshold/top-k framing instead; avoids the SMOTE-before-split trap common in published notebooks.
- **Rule baseline reported at its own operating point** (it flags however many it flags), models at top-k — noted in the comparison table.
- **k = 20 calls/day × test days** — grounded in a staffing story, not an arbitrary percentile.
- **Synthetic-data fallback** so the pipeline runs before the Kaggle download (clearly labeled meaningless for conclusions).

## Deviations
- None from the approved plan.

## Open questions
- `prior_*` features treat outcomes as known right after appointment_day. Point-in-time
  correctness at *scheduling* time would exclude appointments still in the future when
  the current one was booked. Acceptable simplification for L1; revisit at L3.
- Neighbourhood one-hot uses min_frequency=50; consider target encoding at L2 (with
  temporal CV to avoid leakage).

## Verification
- `python smoke_test.py` on synthetic data: leakage guards (first appointment has zero
  history; priors ≤ count), temporal ordering assert, both models train, AUC > 0.6 on
  planted signal, business_value computes. Run in sandbox — PASSED (see chat log).
