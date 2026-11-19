"""Per-card history as one row per card: what the online feature store serves at inference.

The model's features need a card's whole past (cumulative median, "first time at this
merchant", 7-day windows), so the state keeps the card's raw history as arrays and the
online path reuses `build_features(history=...)` -- one feature implementation, so batch
and online cannot skew. shortcut: state grows with card activity (~2k rows/card here);
in production store sketches/windows and accept approximate cumulative features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATE_COLUMNS = ["hist_times", "hist_amts", "hist_merchants", "hist_categories"]
# Only what the model needs: name, street, zip, job, etc. never reach the state tables, the
# transactions table or the endpoint request. dob and lat/long stay (age and distance features)
# and remain quasi-identifiers -- see README, known blind spots.
TX_COLUMNS = [
    "trans_num",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "state",
    "city_pop",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]


def build_card_state(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per `cc_num` with its history as arrays (epoch seconds for times)."""

    ordered = frame.sort_values(["cc_num", "trans_date_trans_time"])
    epoch = ordered["trans_date_trans_time"].astype("datetime64[s]").astype("int64")
    cols = {
        "hist_times": epoch,
        "hist_amts": ordered["amt"],
        "hist_merchants": ordered["merchant"],
        "hist_categories": ordered["category"],
    }
    return pd.DataFrame(cols).groupby(ordered["cc_num"]).agg(list).reset_index()


def history_from_state(state: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the minimal history frame `build_features` needs from state rows."""

    state = state.dropna(subset=STATE_COLUMNS)  # cards without state are cold starts
    if state.empty:
        return pd.DataFrame(
            columns=["cc_num", "trans_date_trans_time", "amt", "merchant", "category"]
        )
    flat = state[["cc_num", *STATE_COLUMNS]].explode(STATE_COLUMNS, ignore_index=True)
    return pd.DataFrame(
        {
            "cc_num": flat["cc_num"].astype(str),
            "trans_date_trans_time": pd.to_datetime(flat["hist_times"].astype("int64"), unit="s"),
            "amt": flat["hist_amts"].astype(float),
            "merchant": flat["hist_merchants"],
            "category": flat["hist_categories"],
        }
    )


def score_with_state(model, frame: pd.DataFrame) -> np.ndarray:
    """Online scoring: `frame` = transaction columns + the card's `STATE_COLUMNS` (joined by lookup)."""

    tx = frame[TX_COLUMNS].copy()
    tx["trans_date_trans_time"] = pd.to_datetime(tx["trans_date_trans_time"])
    tx["dob"] = pd.to_datetime(tx["dob"])
    tx["cc_num"] = tx["cc_num"].astype(str)
    state = frame[["cc_num", *STATE_COLUMNS]].drop_duplicates("cc_num")
    history = history_from_state(state.assign(cc_num=state["cc_num"].astype(str)))
    # build_features returns rows in (card, time) order; map back to the request order
    tx = tx.assign(_row=np.arange(len(tx))).sort_values(
        ["cc_num", "trans_date_trans_time"], kind="stable"
    )
    proba = model.predict_proba(tx.drop(columns="_row"), history=history)
    out = np.empty(len(tx))
    out[tx["_row"].to_numpy()] = proba
    return out
