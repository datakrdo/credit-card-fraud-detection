"""Shared bits for the Databricks job scripts (run on serverless; not part of the wheel)."""

import argparse
from contextlib import suppress

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def parse_args(*extra: tuple[str, dict]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    for flag, kw in extra:
        p.add_argument(flag, **kw)
    return p.parse_args()


def table(catalog: str, schema: str, name: str) -> str:
    return f"{catalog}.{schema}.{name}"


def read_pandas(full_name: str):
    return spark.table(full_name).toPandas()


def append(df, full_name: str) -> None:
    spark.createDataFrame(df).write.mode("append").option("mergeSchema", "true").saveAsTable(
        full_name
    )


def load_champion(catalog: str):
    """The @champion FraudModel, from the raw artifact logged next to the registered model.

    The registered model is wrapped by the Feature Store (for the endpoint's lookups) and cannot
    be unwrapped, so batch jobs read the plain joblib artifact from the same run instead.
    """

    import mlflow
    from mlflow.tracking import MlflowClient

    from fraud.pipeline import FraudModel

    mlflow.set_registry_uri("databricks-uc")
    version = MlflowClient().get_model_version_by_alias(f"{catalog}.ml.fraud_model", "champion")
    return FraudModel.load(
        mlflow.artifacts.download_artifacts(f"runs:/{version.run_id}/fraud_model.joblib")
    )


def exists(full_name: str) -> bool:
    return spark.catalog.tableExists(full_name)


def replace_date(df, full_name: str, column: str, date) -> None:
    """Idempotent daily write: drop that date's rows, then append, so a retry never duplicates."""

    if exists(full_name):
        spark.sql(f"DELETE FROM {full_name} WHERE {column} = DATE'{date}'")
    if len(df):
        append(df, full_name)


def ensure_primary_key(full_name: str, column: str) -> None:
    """Feature tables need a primary key (and Change Data Feed, for the online store sync)."""

    spark.sql(f"ALTER TABLE {full_name} ALTER COLUMN {column} SET NOT NULL")
    spark.sql(f"ALTER TABLE {full_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    with suppress(Exception):  # constraint already exists from a previous run
        spark.sql(f"ALTER TABLE {full_name} ADD CONSTRAINT pk_{column} PRIMARY KEY ({column})")


def upsert_feature_table(df, full_name: str, key: str) -> None:
    """Create the feature table once, then MERGE on later runs.

    The online-store sync streams this table's Change Data Feed. Rewriting it with
    overwriteSchema, or re-running ALTER ... SET NOT NULL, registers as a schema change and
    the sync pipeline fails (DELTA_SCHEMA_CHANGED). MERGE changes rows only.
    """

    if not exists(full_name):
        overwrite(df, full_name)
        ensure_primary_key(full_name, key)
        return
    spark.createDataFrame(df).createOrReplaceTempView("_feature_updates")
    spark.sql(
        f"MERGE INTO {full_name} t USING _feature_updates s ON t.{key} = s.{key} "
        "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"
    )


def overwrite(df, full_name: str) -> None:
    spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        full_name
    )
