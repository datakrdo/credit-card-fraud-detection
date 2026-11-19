"""Compute drift + delayed-label performance for today's window and write alerts."""

import pandas as pd
from _common import exists, load_champion, parse_args, read_pandas, replace_date, table

from fraud.monitoring import delayed_performance, drift_metrics, evaluate_rules

DRIFT_WINDOW_DAYS, PERF_WINDOW_DAYS = 7, 14

args = parse_args()
model = load_champion(args.catalog)
ref, thr = model.reference, model.threshold

preds = read_pandas(table(args.catalog, "ml", "predictions"))
preds["date"] = pd.to_datetime(preds["date"])
today = preds["date"].max()
scenario = preds.loc[preds["date"] == today, "scenario"].iloc[0]

win = preds[preds["date"] > today - pd.Timedelta(days=DRIFT_WINDOW_DAYS)]
if win["date"].nunique() >= DRIFT_WINDOW_DAYS:  # warm-up: need a full week
    drift = drift_metrics(ref, win, win["fraud_probability"].to_numpy(), threshold=thr)
else:
    drift = None

perf = None
labels_table = table(args.catalog, "raw", "labels")
labels = read_pandas(labels_table) if exists(labels_table) else None
if labels is not None and len(labels):
    hi = today - pd.Timedelta(days=7)  # newest traffic day whose labels have arrived
    matured = preds[
        (preds["date"] > hi - pd.Timedelta(days=PERF_WINDOW_DAYS)) & (preds["date"] <= hi)
    ]
    perf = delayed_performance(
        matured.merge(labels, on="trans_num"), threshold=thr, fp_cost=model.fp_cost
    )

alerts = evaluate_rules(ref, drift, perf) if drift else []
metrics = {**(drift or {}), **(perf or {})}
date = today.date()
replace_date(
    pd.DataFrame(
        [
            {"date": date, "scenario": scenario, "metric": k, "value": float(v)}
            for k, v in metrics.items()
        ]
    ),
    table(args.catalog, "monitoring", "daily_metrics"),
    "date",
    date,
)
replace_date(
    pd.DataFrame(alerts, columns=["rule", "severity", "value", "limit"]).assign(
        date=date, scenario=scenario
    ),
    table(args.catalog, "monitoring", "alerts"),
    "date",
    date,
)
print(f"{date}: {len(alerts)} alerts")
