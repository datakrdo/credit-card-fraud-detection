"""Drop predictions and labels older than 12 months, measured on the data's own clock."""

from _common import exists, parse_args, spark, table

RETENTION_MONTHS = 12

args = parse_args()
# raw.transactions is deliberately not pruned: it is the source of the card state, and
# cumulative features (median, first-time merchant) change if old history disappears.
for schema, name, column in (("ml", "predictions", "date"), ("raw", "labels", "label_date")):
    full = table(args.catalog, schema, name)
    if exists(full):
        spark.sql(
            f"DELETE FROM {full} WHERE {column} < "
            f"add_months((SELECT max({column}) FROM {full}), -{RETENTION_MONTHS})"
        )
        print(f"retention applied to {full}")
