🇬🇧 [English](README.md) · 🇪🇸 [Español](README_es.md)

# Fraud detection — credit card transactions

Fraud detection on the synthetic Sparkov credit card transaction dataset: ~1.85 million
transactions from 983 cards between January 2019 and December 2020, with a fraud
prevalence of 0.58% in the training period and 0.39% in the holdout.

Two deliverables:

- **[`fraud-portfolio.ipynb`](fraud-portfolio.ipynb)** — self-contained notebook
  (does not import the `fraud` package), executed end to end. The full walkthrough: problem, EDA,
  causal feature engineering, the model ladder, the imbalance-handling decision,
  calibration, and the threshold-and-money section.
- **`src/fraud/`** — installable production package with `uv`: loading and temporal
  split, feature engineering, GPU models, evaluation, threshold, orchestrator, and a
  training/batch-scoring CLI.

## Design decisions 🧭

- **PR-AUC against prevalence, never accuracy.** A constant classifier that never flags
  fraud scores 99.4% accuracy on this dataset. `src/fraud/evaluation.py::binary_metrics`
  reports PR-AUC and its lift over the real prevalence (0.58% train / 0.39% holdout), not
  against the 0.5 of a random ranking.
- **Temporal split, not random.** The model scores future transactions with data from the
  past; a random split would mix a card's future transactions into that same card's
  training set. `src/fraud/data.py::temporal_split` cuts by date, and `fraudTest.csv` is
  treated as the dataset's natural holdout, untouched until final evaluation.
- **Causal velocity features.** The 22–23h hour shows a 5x lift over the base prevalence,
  and the median seconds since a card's previous transaction is 4,908 in fraud versus
  16,623 in legitimate transactions — the burst is real signal. That whole family of
  features (`src/fraud/features.py`) is computed with
  `groupby("cc_num").rolling(window, on=..., closed="left")`, which excludes the current
  row: an aggregate that leaks into itself inflates validation PR-AUC without the model
  having learned anything generalizable.
- **`distance_km` is built and left out of the model.** `merch_lat`/`merch_long` in this dataset are
  generated uniformly around the cardholder's home, so cardholder-to-merchant distance
  doesn't discriminate (≈76 km in both classes). An ablation on the final model confirms it:
  validation PR-AUC 0.9770 without it, 0.9763 with it. It stays computed in
  `build_features` so the negative result is reproducible, but isn't a model input.
- **Loss weighting, not SMOTE.** `scale_pos_weight` (boosted models) / `class_weight="balanced"`
  (LogReg) instead of resampling. Reasons: (1) 9,651 absolute fraud cases isn't a
  label-scarcity problem; (2) SMOTE interpolates in a space with high-cardinality
  categoricals (`merchant`, 693 levels) where Euclidean distance is meaningless; (3)
  weighting doesn't touch the data, so it can't leak into validation; (4) resampling
  distorts probabilities, and this project needs calibrated probabilities for the dollar
  savings calculation. The notebook measures, not just asserts: it compares validation
  PR-AUC across no-treatment / weighting / SMOTENC / undersampling before locking in the
  decision.
- **Threshold by net dollar savings, frozen on validation.** `src/fraud/threshold.py`
  maximizes `Σ amt[TP] − FP_COST·|FP| − Σ amt[FN]` over validation, never over the
  holdout — choosing it there would be the same leakage as tuning any hyperparameter
  against the final evaluation set.

## Tools 🛠️

- Python, pandas, NumPy
- scikit-learn (`Pipeline`, `ColumnTransformer`, `IsotonicRegression`, `DummyClassifier`)
- XGBoost and LightGBM on GPU (`device="cuda"` / `"gpu"`)
- imbalanced-learn (`SMOTENC`, `RandomUnderSampler`, inside `imblearn.Pipeline`)
- SHAP
- matplotlib, seaborn
- uv, ruff, pytest, GitHub Actions

## Skills 🧠

- Causal feature engineering (rolling windows and aggregates with no temporal leakage)
- Classification under extreme imbalance: PR-AUC, probability calibration, cost-benefit
- Temporal validation for event time series (date-based split, production holdout)
- Decision-threshold optimization by expected dollar value
- Model explainability (SHAP) and communicating results to a non-technical audience
- Packaging an ML project as an installable package with CLI, tests, and CI

## Structure

- `src/fraud/data.py` — loads, validates schema, casts `cc_num` to string, sorts
  causally, and cuts `fraudTrain.csv` into train/validation by date (`TemporalSplit`).
- `src/fraud/features.py` — `build_features`, pure and stateless: temporal features,
  amount, causal per-card velocity, merchant/category novelty, and `distance_km` (computed,
  not a model input).
- `src/fraud/modeling.py` — `MODEL_SPECS` registry (`dummy`, `logreg`, `lightgbm`,
  `xgboost`), the last two using GPU (`device="cuda"`/`"gpu"`) with graceful degradation
  if the `[gpu]` extras aren't installed.
- `src/fraud/evaluation.py` — PR-AUC against prevalence, precision@k, confusion matrix.
- `src/fraud/threshold.py` — vectorized threshold sweep, net savings, marginal analysis,
  and `FP_COST` sensitivity.
- `src/fraud/pipeline.py` — single orchestrator: split → features → fit → isotonic
  calibration on validation → frozen threshold. `FraudModel` packages the three
  together, because a threshold without its model isn't a reproducible decision.
- `src/fraud/cli.py` — `train` / `score` / `evaluate` / `simulate` subcommands.
- `src/fraud/monitoring.py`, `simulation.py`, `state.py`, `serving.py` — see MLOps below. `jobs/`, `resources/`, `databricks.yml`: the Databricks bundle.
- `src/fraud/explainability.py` — SHAP on the final model, behind the `[shap]` extra.

## Quick start 🚀

`data/` is in `.gitignore`: download the Sparkov dataset from
[Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place
`fraudTrain.csv`/`fraudTest.csv` in `data/raw/` before running the following.

```bash
uv sync --all-extras
uv run ruff check src/ tests/ jobs/ && uv run ruff format --check src/ tests/ jobs/
uv run pytest -q

uv run fraud train --data data/raw/fraudTrain.csv --model xgboost --out models/
uv run fraud evaluate --model models/xgboost.joblib --data data/raw/fraudTest.csv
uv run fraud score transactions.csv --model models/xgboost.joblib --history data/raw/fraudTrain.csv -o scores.csv
```

`uv run jupyter nbconvert --execute --inplace fraud-portfolio.ipynb` reproduces the
notebook from scratch.

## MLOps: deploy, monitor, simulate 🛰️

```
Volume (CSVs) ─► train_register ─► MLflow run ─► UC model fraud.ml.fraud_model
                                      │              @challenger ─gate─► @champion
                                      ▼
 daily_replay job (1 run = 1 simulated day, optional injected drift)
   simulate_day ─► raw.transactions / ml.predictions / raw.labels (7-day delay)
   refresh_features ─► features.card_state ─► online store ─► endpoint `fraud-scoring`
   monitor ─► monitoring.daily_metrics / alerts ─► AI/BI dashboard + SQL alert
   retention ─► drops predictions/labels older than 12 months
```

- **Registry and lineage.** `FraudModel` carries `metadata` (version = timestamp + git sha, SHA-256 of the
  CSVs, library versions) and `reference` (validation distributions) inside the artifact. `jobs/train_register.py`
  logs it to MLflow, registers `fraud.ml.fraud_model`, sets `@challenger`, and promotes to `@champion` only if
  **validation** PR-AUC is within 0.005 of the current champion (never the holdout: `fraudTest` is report-only).
- **Retraining.** Run `train_register` with `through_date=YYYY-MM-DD` to refit from `raw.transactions` + `raw.labels`
  (every row up to that date must be labeled). Right after a drift set `test_days`, `valid_days` (validation = last N
  days) and `half_life_days` (recency weights); the defaults are wrong for that case, see "Retraining after a drift".
  Nothing triggers it automatically: the runbook below says when a human should.
- **Online path.** The features need a card's whole history, so `features.card_state` (one row per card, history
  as arrays, `src/fraud/state.py`) is published to a Databricks online store and looked up by `cc_num` at
  serving time. The endpoint reuses `build_features(history=...)`: one implementation for batch and online,
  with a parity test (`tests/test_state.py`). Limit: state grows with activity; production would keep windows
  or sketches and accept approximate cumulative features.
- **Monitoring** (`src/fraud/monitoring.py`, numpy/pandas only). Same day, no labels: PSI per feature and for
  the score (7-day window, so weekday volume swings don't read as drift), alert rate vs validation, unseen
  categories, nulls. With labels 7 days late: recall and net savings as a share of fraud dollars (not per
  transaction: that moves with prevalence, 0.59% in validation vs 0.39% in the holdout) on a 14-day matured window.
  Bands: PSI > 0.10 warning, > 0.25 critical; alert rate outside 0.5-2x; recall -10 pp; savings -20%. Only
  `critical` rows feed the SQL alert; warnings are dashboard-only.
- **Simulation** (`src/fraud/simulation.py`). Replays `fraudTest.csv` day by day; from `onset_day` one drift is
  injected. Measured with 45 simulated days, onset on day 21 (`uv run fraud simulate --scenario ...`):

| Scenario | What changes | First critical alert |
|---|---|---|
| `none` | nothing | **one false alarm**: `alert_rate_ratio` day 27 (0.495 vs the 0.5 floor); plus a persistent PSI warning (0.11-0.12 on `tx_count_card_7D`) |
| `category_shift` | 40% of traffic jumps to `shopping_net` | day 23, `psi_category` |
| `amount_inflation` | every amount x1.8 | day 23, `alert_rate_ratio`; PSI on amount features from day 25 |
| `new_fraud_pattern` | fraud becomes small, daytime | day 23, alert rate (but see below); **recall alert day 28** (0.86 vs 0.88 floor, after the 7-day label delay); savings day 37 |

  Read the last row with the first: the alert-rate rule is not specific (it also fires in `none`, at day 27), so
  concept drift is only *confirmed* when labels arrive. The bands were not tuned on the holdout; a replay of the
  validation period with no drift (54 days) raised 0 alerts (max PSI 0.036, alert-rate ratio between 0.51 and 1.55,
  so the lower band is tight even in-sample). The holdout's lower prevalence moves the ratio to ~0.7, which is why
  `none` brushes the floor there.

```bash
uv run fraud simulate --model models/xgboost.joblib --scenario new_fraud_pattern   # local, no Databricks
V="--profile datakrdo --var warehouse_id=<id> --var catalog=workspace"              # Free Edition: use the existing `workspace` catalog
databricks bundle deploy $V                                                         # schemas, jobs, dashboard, alert
databricks fs cp data/raw/fraudTrain.csv dbfs:/Volumes/workspace/raw/landing/ --profile datakrdo   # and fraudTest.csv
databricks bundle run train_register $V
scripts/deploy_endpoint.sh datakrdo fraud <team-group>                              # endpoint on @champion + CAN_QUERY for the group
databricks bundle run daily_replay $V --params scenario=new_fraud_pattern,onset_day=5   # one simulated day per run
```

> Deployed and run on a Databricks Free Edition workspace: `bundle validate --strict` passes, `train_register`
> registers `@champion`, `daily_replay` (replay, monitor, retention, online store publish) runs end to end, and
> the `fraud-scoring` endpoint answers (`serving-endpoints query`). Only a real workspace shows the fixes it took:
> MLflow experiment on serverless, primary key on the feature table, model version lookup, the Feature Store
> wrapper hiding the model, `get_online_store` returning `None` instead of raising, endpoint JSON shape, and
> **the online-store sync failing with `DELTA_SCHEMA_CHANGED`** when the feature table is overwritten or altered
> on every refresh (it streams the table's change feed; the job now MERGEs). **Not available here:** the
> endpoint's inference table (AI Gateway rejects it on this workspace). **Not exercised:** score parity between
> the endpoint and the batch path, and the dashboard rendering with data.

**Runbook.** `psi_*` critical on inputs: check upstream data first (schema, units, a new merchant mix); retrain only
if recall also drops. `alert_rate_ratio` outside the band with stable inputs: suspect concept drift, pull labeled
cases and wait for the recall check. `recall`/`savings_ratio`: retrain with `through_date`; the validation gate
decides promotion. Retention: `ml.predictions` and `raw.labels` are cut at 12 months (`jobs/retention.py`);
`raw.transactions` is not, because the card state (cumulative features) is built from it.

## Retraining after a drift 🔁

Run on the Databricks workspace after `new_fraud_pattern` (fraud turns small and daytime, onset 2020-06-26), scored on the
same 4 held-out days (2020-07-12..15, 37 frauds):

| Model | PR-AUC | Recall | What it is |
|---|---|---|---|
| Champion v4 (old pattern) | 0.003 | 0.00 (0 of 37) | the model that raised the drift alerts |
| v6 / v7 (first retrains) | 0.003 | 0.00 | did **not train on the drift days** |
| **v8** | **0.103** | **0.62** (23 of 37) | validation = last 4 days, recency half-life 14 days |

Two defaults of a naive retrain hid the new pattern, both found by this exercise:

1. **The validation split swallowed the drift.** The last 20% of an 18-month history starts in March, so 15 days of
   drift landed entirely in validation and none in training (`split_by_recency(valid_days=...)`).
2. **The old pattern drowned the new one.** ~7,500 old-pattern frauds against ~180 new ones. Locally, on the same data:
   full history 0.07 PR-AUC, last 60 days 0.28, recency weights (half-life 14 d) 0.38, recent data only 0.50.
   Shorter half-life adapts faster but forgets more; the simulation cannot measure the forgetting, because the old
   fraud disappears entirely in it.

How much labeled drift is needed (local learning curve, same test): 3 days 0.017, 6 days 0.07, 15 days 0.18, 20+ days
~0.20 PR-AUC. With a 7-day label delay the model is nearly blind to the new pattern for about three weeks, so the
retrain trigger should be a minimum count of new labeled frauds (~150-200), and something else (a rule) must cover
the gap. v8's numbers come from 37 frauds: treat them as an existence proof, not an estimate.

## Is this production-ready? ⚖️

No: it is a reference implementation of the MLOps loop. The data is synthetic (rule-based fraud, 983 cards, no
external validation), so the headline metrics (test PR-AUC ~0.955) measure how well the model learns those rules.
The serving design is not viable as built (a daily-refreshed state loses most of the accuracy; see below), endpoint/batch
score parity is untested, the endpoint has no request logging on this workspace type, there are no latency or load
tests, and the review cost (USD 4) is an assumption. What it does show: temporal validation, causal features, lineage,
drift monitoring that fired, a gated retrain loop, and an honest list of what is still open.

## Known blind spots 🔦

**The online path can lose most of the model's accuracy.** The state is refreshed once a day, so a transaction
does not see the card's earlier transactions of the same day, and the burst features (`tx_count_card_1h`,
`seconds_since_prev_tx`) are what the model leans on. Measured by scoring 14 days of holdout both ways (all 983
cards, 122 frauds; `simulation.build_features_frozen`): PR-AUC **0.939 with live history vs 0.739 with a
daily-refreshed state**, recall 0.951 -> 0.844, alerts 200 -> 331 at the same threshold. A daily refresh is therefore
**not** an acceptable serving design for this model; production needs the state updated per transaction
(streaming) or the model retrained on state-as-served features. The demo's `daily_replay` job scores in batch, so it
does not show this.

| Blind spot | Status |
|---|---|
| Validation/holdout/scoring started every card "cold" (no history): train/serve skew | Fixed with `build_features(history=)`. Measured before tokenization: validation 0.9749 -> 0.9758, test 0.9528 -> 0.9545. Final retrain: 0.9758 / 0.9563 (the main notebook, with its own loader, gives 0.9545) |
| `fraud score` required `is_fraud`; output carried PII | Fixed (`require_label=False`); output is ids, amount, score, model version |
| Full card number (PAN) used as key everywhere | Fixed: `cc_num` is HMAC-tokenized on load (`FRAUD_CARD_KEY`; the default key only exists for the demo). Tokenize upstream of the endpoint too |
| Name, street, zip, job stored in the tables / sent to the endpoint | Fixed: only the model's input columns (`state.TX_COLUMNS`). `dob` and home `lat/long` remain quasi-identifiers |
| Artifact had no version, data hash or reference distribution | Fixed (`FraudModel.metadata`, `.reference`) |
| Promotion gate used the holdout | Fixed: gate on validation. Still open: after a retrain it compares validation PR-AUC measured on *different* validation sets (v8: 0.48 vs champion 0.98), so it will not promote a better model by itself; v8 was promoted by hand after comparing both on the same test days |
| Boosters hard-wired to GPU | `--device cpu` / `device=` override (serverless, CI) |
| Re-running a job duplicated rows; monitor crashed before labels existed | Fixed: per-date delete+append, tables created up front (untested on Databricks) |
| CI: unpinned actions, no dependency audit | Actions pinned by SHA, Dependabot, `pip-audit` (found and fixed 6 vulnerable packages) |
| Monitor watches the batch path, not the endpoint; the endpoint has no inference table (not supported on this workspace type) | Open: add a daily sample through the endpoint and compare scores with batch |
| Naive retrain does not learn a recent drift (validation split, dilution) | Fixed with `valid_days` and `half_life_days`; see "Retraining after a drift" |
| Threshold and calibration both fit on the same validation slice; reference recall 0.98 is in-sample | Open: a nested split would be cleaner; effect not measured |
| `age_years` is a model input | Open: no fairness review; remove or justify before a real deployment |
| Label delay (~7 days) unmodeled in training; synthetic data, 983 cards | Open: documented, simulated in monitoring only |
| 7-day PSI window is blind to a one-day anomaly; `toPandas()` of the whole history every day | Open: fine for ~2M rows, not for production scale |

## Limitations ⚠️

Synthetic dataset (Sparkov), no external validation. No device, session, or payment
channel data. `FP_COST` (manual review cost, USD 4) is an estimate external to the
dataset — the notebook's sensitivity section shows how much the recommendation depends
on it. 983 cards is a small population. The model has no way to handle a card with no
history (cold start): velocity features are zero by construction there, not by model
error.

## License

MIT — see the `LICENSE` file.
