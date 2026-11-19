"""Recompute the per-card state from all transactions and publish it to the online store."""

import time

import pandas as pd
from _common import parse_args, read_pandas, table, upsert_feature_table
from databricks.feature_engineering import FeatureEngineeringClient

from fraud.state import build_card_state

args = parse_args(("--online-store", {"default": "fraud-card-state"}))
tx = read_pandas(table(args.catalog, "raw", "transactions"))
tx["trans_date_trans_time"] = pd.to_datetime(tx["trans_date_trans_time"])
state_table = table(args.catalog, "features", "card_state")
upsert_feature_table(build_card_state(tx), state_table, "cc_num")

fe = FeatureEngineeringClient()
store = fe.get_online_store(name=args.online_store)  # returns None, not an error, if missing
if store is None:  # first run: create it (CU_1 is the smallest) and wait for AVAILABLE
    fe.create_online_store(name=args.online_store, capacity="CU_1")
while store is None or str(store.state).split(".")[-1] != "AVAILABLE":
    time.sleep(15)
    store = fe.get_online_store(name=args.online_store)
fe.publish_table(
    source_table_name=state_table,
    online_store=store,
    online_table_name=f"{state_table}_online",
)
print(f"Published {state_table} -> online store {args.online_store}")
