"""Train on the landing CSVs, log to MLflow, register in UC as @challenger, promote if not worse."""

import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from _common import overwrite, parse_args, read_pandas, spark, table, upsert_feature_table
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, Schema

import fraud
from fraud.data import load_transactions, split_by_recency
from fraud.pipeline import fit_from_split, run_train
from fraud.serving import FraudPyfunc
from fraud.state import TX_COLUMNS, build_card_state

OUTPUT_SCHEMA = Schema([ColSpec("double", "fraud_probability"), ColSpec("boolean", "flagged")])
# Gate on validation PR-AUC, never on the holdout: `fraudTest` stays a report-only set.
# shortcut: in production the gate would be a recent labeled window; here validation is the only
# held-out slice that is not the final holdout.
PROMOTION_TOLERANCE = 0.005  # challenger may be this much worse and still ship

args = parse_args(
    ("--model", {"default": "xgboost"}),
    # Retrain from the production tables instead of the landing CSVs: every row up to this
    # date must already have its label (use a date at least 7 days behind the latest data).
    ("--through-date", {"default": None}),
    # Days at the end held out as test when retraining. Keep it short right after a drift: labels
    # for the new pattern are scarce, and anything held out is data the model cannot learn from.
    ("--test-days", {"type": int, "default": 14}),
    # Validation = last N days before the test (0: last 20% of rows). Needed after a drift.
    ("--valid-days", {"type": int, "default": 0}),
    # Recency weighting for retrains (see fraud.pipeline.recency_weights); 0 = off.
    ("--half-life-days", {"type": float, "default": 0.0}),
    # Serverless jobs have no default experiment, so mlflow.start_run() needs one.
    ("--experiment", {"required": True}),
)
landing = f"/Volumes/{args.catalog}/raw/landing"
name = table(args.catalog, "ml", "fraud_model")

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(args.experiment)
if args.through_date:
    tx = read_pandas(table(args.catalog, "raw", "transactions"))
    late = read_pandas(table(args.catalog, "raw", "labels"))[["trans_num", "is_fraud"]]
    tx = tx.merge(late.rename(columns={"is_fraud": "late"}), on="trans_num", how="left")
    tx["is_fraud"] = tx["is_fraud"].fillna(tx["late"])
    tx["trans_date_trans_time"] = pd.to_datetime(tx["trans_date_trans_time"])
    tx = tx[tx["trans_date_trans_time"] < pd.Timestamp(args.through_date) + pd.Timedelta(days=1)]
    if tx["is_fraud"].isna().any():
        raise SystemExit(f"Unlabeled rows before {args.through_date}: pick an earlier date.")
    tx = (
        tx.drop(columns="late")
        .astype({"is_fraud": int})
        .sort_values(["cc_num", "trans_date_trans_time"], ignore_index=True)
    )
    model, metrics = fit_from_split(
        split_by_recency(tx, test_days=args.test_days, valid_days=args.valid_days or None),
        model_name=args.model,
        device="cpu",
        data_sha256={},
        half_life_days=args.half_life_days or None,
    )
    model.metadata["trained_through"] = args.through_date
else:
    model, metrics = run_train(
        f"{landing}/fraudTrain.csv", f"{landing}/fraudTest.csv", model_name=args.model, device="cpu"
    )

# Seed the history tables the daily job extends (first training only: a retrain must not reset
# the replay): transactions, with their labels for the seed period, + current card state.
history = load_transactions(f"{landing}/fraudTrain.csv")
if not args.through_date:
    overwrite(
        history[[*TX_COLUMNS, "is_fraud"]].assign(
            trans_date=history["trans_date_trans_time"].dt.date, source="train"
        ),
        table(args.catalog, "raw", "transactions"),
    )
    upsert_feature_table(
        build_card_state(history), table(args.catalog, "features", "card_state"), "cc_num"
    )

# Create the tables the dashboard/alert query so they exist from day 0 (before any alert or label).
for ddl in (
    "raw.labels (trans_num STRING, is_fraud BIGINT, label_date DATE)",
    "monitoring.daily_metrics (date DATE, scenario STRING, metric STRING, value DOUBLE)",
    "monitoring.alerts (date DATE, scenario STRING, rule STRING, severity STRING, "
    "value DOUBLE, `limit` DOUBLE)",
):
    spark.sql(f"CREATE TABLE IF NOT EXISTS {args.catalog}.{ddl}")

with mlflow.start_run(run_name=model.metadata["version"]) as run:
    mlflow.log_params(
        {k: v for k, v in model.metadata.items() if k in ("model", "git_sha", "version")}
    )
    mlflow.log_metrics(
        {
            "test_pr_auc": metrics["test"]["pr_auc"],
            "valid_pr_auc": metrics["valid"]["pr_auc"],
            "test_recall": metrics["test"]["recall"],
            "threshold": model.threshold,
        }
    )
    mlflow.log_dict(model.metadata["data_sha256"], "data_sha256.json")
    mlflow.log_dict(model.reference, "reference.json")
    with tempfile.TemporaryDirectory() as tmp:  # plain artifact for the batch jobs (see _common)
        model.save(f"{tmp}/fraud_model.joblib")
        mlflow.log_artifact(f"{tmp}/fraud_model.joblib")

    # The training set only tells the Feature Store *which* table/key to look up at serving
    # time (card_state by cc_num); the pyfunc consumes those columns itself.
    fe = FeatureEngineeringClient()
    keys = spark.createDataFrame(history[TX_COLUMNS].head(1000))
    training_set = fe.create_training_set(
        df=keys,
        feature_lookups=[
            FeatureLookup(
                table_name=table(args.catalog, "features", "card_state"), lookup_key="cc_num"
            )
        ],
        label=None,
    )
    # The endpoint builds its own environment: ship the package source and pin the libs the
    # artifact was trained with, or unpickling FraudModel fails there.
    libs = [f"{k}=={v}" for k, v in model.metadata["libs"].items()]
    info = fe.log_model(
        model=FraudPyfunc(model),
        artifact_path="model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=name,
        output_schema=OUTPUT_SCHEMA,
        code_paths=[str(Path(fraud.__file__).parent)],
        pip_requirements=["mlflow", "joblib", *libs],
    )

client = MlflowClient()
# fe.log_model does not reliably return the registered version: find it by the run that made it.
version = max(
    int(v.version)
    for v in client.search_model_versions(f"name = '{name}'")
    if v.run_id == run.info.run_id
)
client.set_registered_model_alias(name, "challenger", version)
try:
    champ = client.get_model_version_by_alias(name, "champion")
    champ_pr = client.get_run(champ.run_id).data.metrics["valid_pr_auc"]
except Exception:  # no champion yet (first deploy)
    champ_pr = None
valid_pr = metrics["valid"]["pr_auc"]
if champ_pr is None or valid_pr >= champ_pr - PROMOTION_TOLERANCE:
    client.set_registered_model_alias(name, "champion", version)
    print(f"Promoted v{version} to @champion (validation PR-AUC {valid_pr:.4f})")
else:
    print(f"v{version} stays @challenger: {valid_pr:.4f} < champion {champ_pr:.4f}")
