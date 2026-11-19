"""Replay the next day of fraudTest as new traffic, score it with @champion, release old labels."""

import pandas as pd
from _common import load_champion, parse_args, read_pandas, replace_date, spark, table

from fraud.data import load_transactions
from fraud.features import build_features
from fraud.modeling import FEATURE_COLUMNS
from fraud.simulation import apply_scenario
from fraud.state import TX_COLUMNS

LABEL_DELAY_DAYS = 7  # fraud is confirmed ~a week later (chargebacks, customer calls)

args = parse_args(
    ("--scenario", {"default": "none"}),
    ("--onset-day", {"type": int, "default": 21}),
)
model = load_champion(args.catalog)

test = load_transactions(f"/Volumes/{args.catalog}/raw/landing/fraudTest.csv")
test["day"] = test["trans_date_trans_time"].dt.normalize()
day0 = test["day"].min()

tx_table = table(args.catalog, "raw", "transactions")
pred_table = table(args.catalog, "ml", "predictions")
history = read_pandas(tx_table)
history["trans_date_trans_time"] = pd.to_datetime(history["trans_date_trans_time"])
# One replayed day per run. The counter comes from the transactions table, which is written
# last: a run that fails midway is retried for the same day, and every write below is per-date.
n_done = spark.table(tx_table).where("source = 'replay'").select("trans_date").distinct().count()
today = day0 + pd.Timedelta(days=n_done)

drifted = apply_scenario(test, args.scenario, day0 + pd.Timedelta(days=args.onset_day))
batch = drifted[drifted["trans_date_trans_time"].dt.normalize() == today]
if batch.empty:
    raise SystemExit(f"No holdout traffic left for {today.date()}: the replay is over.")

feat = build_features(batch.drop(columns="is_fraud"), history=history)  # unlabeled, like production
proba = model.score_features(feat)
cols = list(
    dict.fromkeys(["trans_num", "cc_num", "trans_date_trans_time", "amt", *FEATURE_COLUMNS])
)
preds = feat[cols].assign(
    date=today.date(),
    scenario=args.scenario,
    fraud_probability=proba,
    flagged=proba >= model.threshold,
    model_version=model.metadata["version"],
)
replace_date(preds, pred_table, "date", today.date())


# Labels for traffic from LABEL_DELAY_DAYS ago become known today.
matured = drifted[
    drifted["trans_date_trans_time"].dt.normalize() == today - pd.Timedelta(days=LABEL_DELAY_DAYS)
]
replace_date(
    matured[["trans_num", "is_fraud"]].assign(label_date=today.date()),
    table(args.catalog, "raw", "labels"),
    "label_date",
    today.date(),
)
replace_date(
    batch[TX_COLUMNS].assign(trans_date=today.date(), source="replay"),
    tx_table,
    "trans_date",
    today.date(),
)  # last on purpose: it advances the day counter
print(
    f"Day {n_done} ({today.date()}): scored {len(batch)} tx, {int((proba >= model.threshold).sum())} flagged"
)
