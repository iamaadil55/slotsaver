# SlotSaver — Level 4: Production Design

*Design document — written thinking, not code. Levels 1–3 built and validated the
system on public data; this document specifies what turning it into a real
clinical product would require. Every section states its assumptions.*

**Status of claims:** architecture and MLOps sections are standard engineering
practice. The HIPAA section is an engineer's map of the territory, **not legal
advice** — a real deployment needs a compliance officer and counsel.

---

## 1. What exists vs what this designs

| Layer | Built (L1–L3) | Designed here (L4) |
|---|---|---|
| Model | LightGBM + isotonic, temporal validation | retraining/recalibration lifecycle |
| Serving | FastAPI + Docker, trains at startup | artifact persistence, registry, staged rollout |
| Data | Kaggle CSV / synthetic | live EHR feed via FHIR |
| Monitoring | PSI + base-rate drift, nightly job | alert routing, evaluation loop under intervention |
| Fairness | per-group audit + flags | per-group threshold policy, governance |
| Impact | expected value from probabilities | randomized trial measuring *causal* effect |

## 2. Target architecture

```
┌────────────┐  FHIR R4 (read)   ┌──────────────┐        ┌─────────────────┐
│ Clinic EHR │ ────────────────> │ Ingest + PIT │ ─────> │ Feature store    │
│ (Epic/     │                   │ feature build │        │ (point-in-time)  │
│  athena…)  │ <──────────────── └──────────────┘        └────────┬────────┘
└────────────┘  FHIR Task (write:                                  │ nightly
                 call-list items)                                  v
        ^                                    ┌──────────────────────────────┐
        │                                    │ Scoring service (Docker)     │
┌───────┴──────┐    ranked list + whys       │ model + isotonic from        │
│ Staff UI     │ <────────────────────────── │ MODEL REGISTRY (versioned)   │
│ (dashboard)  │                             └──────────┬───────────────────┘
└──────────────┘                                        │ predictions log
                                                        v
                                   ┌────────────────────────────────┐
                                   │ Monitoring: PSI, calibration,  │
                                   │ fairness, outcome joins        │──> alerts
                                   └────────────────────────────────┘──> retrain trigger
```

Key change from L3: the API **loads versioned artifacts from a registry**
instead of training at startup. Training becomes a separate, audited job.

## 3. EHR integration (FHIR R4)

Standard resources map cleanly onto our schema:

| FHIR resource | Gives us | Our field |
|---|---|---|
| `Appointment` | start time, created time, status | appointment_day, lead_time (created→start), label (status: noshow/fulfilled) |
| `Patient` | birthDate, gender, address | age, gender, neighbourhood proxy |
| `Condition` | diagnoses | hypertension/diabetes-style flags |
| `Communication` | reminder sent? | sms_received |
| `Task` (write-back) | — | one Task per call-list entry, assigned to front desk, with p + whys in `note` |

Phase 1 is **read-only** (poll `Appointment?date=ge{today}` nightly; the label
arrives when status becomes `noshow`/`fulfilled`). Phase 2 writes `Task`
resources so the call list lives inside the EHR workflow instead of a separate
dashboard — adoption dies when staff need a second screen. Auth via SMART on
FHIR (OAuth2 client credentials for backend services).

**Point-in-time correctness fix (L1 open question):** with a live feed, the
`prior_*` features are computed from appointments whose outcome was *known at
scoring time* — the approximation we documented in Level 1 disappears.

## 4. Data protection (HIPAA-shaped, not legal advice)

- **PHI inventory:** everything we touch is PHI once tied to a real patient.
  Minimum-necessary: the model needs no names, no free text, no exact address —
  ingest only the fields in FEATURES plus a pseudonymous patient key.
- **BAA** with the clinic before any PHI flows; hosting on a cloud with a BAA
  (AWS/GCP/Azure healthcare configurations).
- **Security Rule safeguards → concrete choices:** TLS in transit; encryption at
  rest (managed KMS); role-based access (staff see the call list, nobody
  queries raw tables ad hoc); audit log on every read of the scored list;
  short retention on prediction logs with outcomes joined then de-identified.
- **Analytics/retraining on de-identified data** (Safe Harbor: strip the 18
  identifiers; ages >89 bucketed — note our 65+ band already complies).
- **The scores themselves are PHI.** The call list is medical-adjacent data
  about identified patients; it inherits every safeguard above.

## 5. Proving causal impact — the randomized trial

Everything so far shows the model *predicts*; nothing yet shows calls *work*.
Recovered-revenue numbers from observational data would be confounded (we call
the riskiest people; their outcomes differ for many reasons).

- **Design:** randomize at the **patient** level (not appointment — the same
  patient appearing in both arms contaminates), stratified by risk tier.
  Control: clinic's existing blanket reminders. Treatment: SlotSaver-targeted
  calls + tiered interventions.
- **Primary endpoint:** attendance rate difference. **Secondary:** recovered
  slots, staff time spent, patient satisfaction. **Guardrails:** per-group
  benefit rates in both arms (the fairness audit runs *inside* the trial).
- **Rough size:** detecting a 3-percentage-point drop from an 18% base rate at
  α=0.05, power 0.8 needs roughly n ≈ 16·p(1−p)/δ² ≈ 16·0.18·0.82/0.03² ≈
  **2,600 patients per arm** — about 2–3 weeks of a large clinic's volume.
  (Approximation formula; a real protocol uses exact two-proportion power.)
- **Stepped-wedge alternative** for small clinics: roll the tool out
  clinic-by-clinic in randomized order; each clinic is its own control.

## 6. The feedback-loop problem (why naive retraining goes wrong)

Once the tool intervenes, future labels are contaminated: a high-risk patient
who *shows up because we called* looks like a model error. Retraining naively
teaches the model its own success is failure.

Mitigations, cheapest first: (1) **log every intervention** alongside every
prediction — non-negotiable from day one; (2) keep a small **never-intervene
holdout slice** (e.g. 5%) as a permanently clean evaluation set; (3) add
`intervention_received` as a feature in retraining; (4) graduate to **uplift
modeling** — predicting who is *persuadable*, not who is risky — once one
trial's worth of randomized data exists.

## 7. MLOps lifecycle

- **Artifact persistence:** training job saves `model.txt` (LightGBM native),
  `isotonic.joblib`, feature list, and metrics to a registry (MLflow Model
  Registry or a versioned bucket). API loads by version tag; `/health` reports
  which version is live.
- **Experiment tracking:** MLflow logs every training run — params, temporal-split
  dates, AUC/PR-AUC/ECE, fairness table snapshot. The fairness table is a
  **release gate**: a model that worsens benefit-rate disparity does not ship.
- **CI (implemented in this repo):** `.github/workflows/ci.yml` runs all four
  smoke tests on every push — leakage guards, calibration, call-list rules,
  drift alarms, API contract. Red X = don't merge.
- **Staged rollout:** *shadow mode* (score silently, compare to reality for 2–4
  weeks) → *assist mode* (staff see the list, keep their own judgment) →
  *default mode* (list drives the workflow). Rollback = repoint the registry tag.
- **Retraining cadence:** monthly, or immediately on PSI/base-rate ALERT —
  and recalibration is cheaper than retraining, so the isotonic refit can run
  weekly on a rolling window.

## 8. Fairness as policy, not just measurement

The Level 3 audit found the model understands elderly patients least
(within-group AUC 0.67 vs 0.73) and their no-shows receive the least help
(benefit rate 0.10 vs 0.36). Production options, all legitimate if stated
openly: collect features that explain elderly no-shows (transport, escort
needs — likely requires one new intake question); set **per-group tier
boundaries** to equalize benefit rates (explicit, documented positive action);
or route low-confidence elderly cases to human review. The chosen policy goes
in the model card, and the audit runs on every retrain as a release gate.

## 9. Cost & scaling sketch (assumptions, not quotes)

Single clinic: one small container + managed Postgres + nightly job — tens of
dollars/month of compute; the real costs are EHR integration effort and
compliance. Multi-tenant: per-clinic calibration (base rates differ by
specialty and region — a lesson straight from the drift work), shared model
optional, per-tenant fairness reports. Break-even vs the ~$150/provider/month
price point from the deep-dive requires roughly one recovered slot per
provider per month — a low bar if the trial confirms even half the modeled
effect.

## 10. Open research directions

Prospective-vs-retrospective gap replication; uplift modeling on trial data;
fair operational policies under capacity constraints; federated learning
across clinics without sharing PHI; causal effect of lead-time reduction
(the notebook-03 question, now testable).

---

*End of roadmap. Levels 1–4: from a Kaggle CSV to a deployable, monitored,
audited product design — with every simplification documented on the way.*
