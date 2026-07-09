# SlotSaver — Patient No-Show Prediction

![smoke-tests](https://github.com/iamaadil55/slotsaver/actions/workflows/ci.yml/badge.svg)

Predict which medical appointments are likely to be missed, so clinic staff can target
reminders, calls, and waitlist backfill at the right patients — instead of reminding everyone.

**Why it matters:** outpatient no-show rates run 10–30%. Every missed slot is lost revenue
(~$200/appointment, industry estimate) and delayed care for someone else on the waitlist.

## Project structure

```
slotsaver/
├── 01_noshow_mvp.ipynb      # Level 1: EDA → features → models → business metrics
├── 02_calibration.ipynb     # Level 2a: honest probabilities + cost-sensitive threshold
├── 03_call_list.ipynb       # Level 2b: the daily call list with per-patient reasons
├── 04_fairness_drift.ipynb  # Level 3: fairness audit + drift monitoring
├── app.py                   # Level 2b: Streamlit dashboard (run locally)
├── api.py                   # Level 3: FastAPI scoring service
├── nightly_job.py           # Level 3: batch scoring + drift report (cron-able)
├── Dockerfile               # Level 3: container for the API
├── src/
│   ├── data.py              # Load, clean, and (for testing) synthesize appointment data
│   ├── features.py          # Leakage-safe feature engineering + temporal split
│   ├── evaluate.py          # PR-AUC, precision@k, metrics comparison table
│   ├── calibrate.py         # 3-way temporal split, ECE, isotonic, EV threshold, tiers
│   ├── explain.py           # TreeSHAP per-patient reasons in plain language
│   ├── pipeline.py          # One end-to-end build shared by every surface
│   ├── fairness.py          # Per-group calibration gaps, benefit rates, audit flags
│   └── monitor.py           # PSI drift detection + retrain trigger
├── smoke_test.py            # Level 1 pipeline check (synthetic data)
├── smoke_test_l2.py         # Level 2a calibration check
├── smoke_test_l2b.py        # Level 2b call-list + explanation check
├── smoke_test_l3.py         # Level 3 ranking/fairness/drift/API check
├── data/                    # Put the Kaggle CSV here (not committed)
└── requirements.txt         # (requirements-api.txt = slim set for the container)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Get the data:** download `KaggleV2-May-2016.csv` from
[Kaggle: Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
(free account required) and place it in `data/`.

No data yet? Everything still runs — the notebook falls back to synthetic data so you can
test the pipeline first. Verify your setup with:

```bash
python smoke_test.py
```

Then launch the notebook **from the project root** (so `src/` imports work):

```bash
jupyter lab 01_noshow_mvp.ipynb
```

## Run the dashboard (Level 2b)

```bash
pip install streamlit
streamlit run app.py
```

Runs on your own computer (not Colab). Without the Kaggle CSV it starts in
clearly-flagged synthetic demo mode, so you can see the product before wiring
real data. Sidebar controls: clinic day, staff call capacity, and the three
economic assumptions — watch the call list and expected value react.

## Run the API (Level 3)

```bash
pip install fastapi uvicorn
uvicorn api:app --reload
```

Then open http://127.0.0.1:8000/docs for interactive Swagger docs. Endpoints:
`GET /health`, `POST /score` (one appointment → calibrated risk + tier + reasons),
`GET /call-list?day=YYYY-MM-DD&capacity=20`. With Docker instead:
`docker build -t slotsaver . && docker run -p 8000:8000 slotsaver`.

The nightly batch job (scores tomorrow, logs predictions, checks drift,
fires a retrain trigger on PSI > 0.25): `python nightly_job.py`.

## What makes this project non-trivial (read before interviews)

1. **Temporal leakage is the quality bar.** Train/test is split by appointment *date*
   (past → future), never randomly. A random split lets the model peek at the future and
   inflates every metric.
2. **Patient history features exclude the current appointment.** `prior_noshows` counts
   only earlier appointments — computing it naively includes the label you're predicting.
3. **A rule baseline comes first.** "≥2 prior no-shows OR booked >14 days ahead" is the
   yardstick. ML must beat it to justify itself.
4. **Accuracy is banned.** ~80% of patients show up, so "predict everyone shows" scores 80%
   accuracy and helps nobody. We use PR-AUC and precision@k (k = how many patients staff
   can actually call per day).

## Results

Real data (110,516 appointments after cleaning, 20.2% no-show base rate).
Temporal split: train through 2016-05-30, test = last 7 appointment days (30,728 appts).
k = 140 (20 staff calls/day × 7 test days).

| Model | ROC-AUC | PR-AUC | Precision@140 | Recall@140 |
|---|---|---|---|---|
| Rule baseline* | — | — | 0.286 | 0.514 |
| Logistic regression | 0.681 | 0.314 | 0.536 | 0.013 |
| LightGBM | **0.730** | **0.346** | **0.621** | 0.015 |

\* The rule is binary, so it's evaluated at its own operating point: it flags 10,140
appointments (a third of the test set — an unworkable call list), catching half of all
no-shows but with only 28.6% precision.

**Reading:** calling the model's top 20/day means ~62% of those calls hit true
would-be no-shows — 3× the base rate. Business translation at k=140:
~87 true no-shows reached → ~26 slots recovered → ~$5,220 recovered vs $700 outreach
cost (assumptions: $200/slot, 30% intervention success, $5/call — all explicit and
challengeable). ROC-AUC 0.73 under a temporal split matches what the strongest
published *prospective* study reported — honest numbers, not leaderboard numbers.

## Roadmap

All four levels are complete: model → calibration → explanations → dashboard →
API + Docker + nightly scoring + drift monitoring + fairness audit → production
design. **[`LEVEL4_PRODUCTION_DESIGN.md`](LEVEL4_PRODUCTION_DESIGN.md)** covers
FHIR/EHR integration, HIPAA-shaped data handling, the randomized trial that
would prove causal impact, feedback-loop handling, and the MLOps lifecycle.
CI runs all four smoke-test suites on every push (badge above).

## Known limitations (honest by design)

- Only 6 weeks of data → thin per-patient history; most patients have 0–1 prior visits.
- Retrospective evaluation only; published prospective studies show live AUC drops.
- `prior_*` features assume outcomes are known immediately after each appointment day.
- Brazilian public-health data from 2016; patterns may not transfer to other systems.
