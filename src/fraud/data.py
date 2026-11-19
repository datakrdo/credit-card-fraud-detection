"""Loading, schema validation, and temporal splitting for the Sparkov transactions."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

# Column order as shipped by the Sparkov generator, minus the anonymous index
# column pandas would otherwise read as "Unnamed: 0".
EXPECTED_COLUMNS: Final[list[str]] = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]

# Identifiers and free text that either leak the label indirectly (a name or
# street uniquely tied to a handful of fraud cases in this small, synthetic
# population) or duplicate information already captured elsewhere
# (`unix_time` is `trans_date_trans_time`; `cc_num`/`dob` are consumed by
# `features.py` to derive behavioural features and are not features
# themselves). Dropped from the model input in `pipeline.py`.
LEAKAGE_COLUMNS: Final[list[str]] = [
    "trans_num",
    "first",
    "last",
    "street",
    "unix_time",
    "cc_num",
    "dob",
    "trans_date_trans_time",
]


# `cc_num` is a full card number (PAN, PCI DSS scope). It is replaced by a keyed hash on load, so
# nothing downstream (features, state tables, predictions, score files, endpoint requests) ever
# holds the PAN. Keyed (HMAC) rather than plain SHA: a plain hash of a 16-digit number is
# brute-forceable. Set FRAUD_CARD_KEY from a secret store; the default only exists so the demo runs.
_DEMO_KEY = "demo-key-not-for-production"


def tokenize_cards(cc_num: pd.Series) -> pd.Series:
    """Stable 16-hex token per card number. Apply once, at load: re-applying re-hashes."""

    key = os.environ.get("FRAUD_CARD_KEY", _DEMO_KEY).encode()
    tokens = {
        v: hmac.new(key, str(v).encode(), hashlib.sha256).hexdigest()[:16] for v in cc_num.unique()
    }
    return cc_num.map(tokens)


@dataclass(frozen=True)
class TemporalSplit:
    """Train/validation cut from `fraudTrain.csv`, plus the untouched holdout."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def validate_schema(frame: pd.DataFrame, *, require_label: bool = True) -> None:
    """Check that `frame` has the expected Sparkov columns (and `is_fraud` if required).

    Production traffic arrives unlabeled -- labels show up days later -- so
    scoring passes `require_label=False`.
    """

    expected = (
        EXPECTED_COLUMNS if require_label else [c for c in EXPECTED_COLUMNS if c != "is_fraud"]
    )
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns {sorted(missing)}. "
            "Re-download fraudTrain.csv/fraudTest.csv from the Sparkov dataset."
        )
    if require_label and not frame["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must be binary (0/1).")


def load_transactions(path: str | Path, *, require_label: bool = True) -> pd.DataFrame:
    """Load one Sparkov CSV, parse dates, and sort into causal (per-card) order.

    `cc_num` is cast to string (it identifies a card, not a quantity) and then
    tokenized (see `tokenize_cards`).
    Sorting by `(cc_num, trans_date_trans_time)` is required before any
    rolling/aggregate feature in `features.py` -- those rely on each card's
    rows already being in chronological order.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Expected data/raw/*.csv.")
    frame = pd.read_csv(
        csv_path,
        index_col=0,
        parse_dates=["trans_date_trans_time", "dob"],
        dtype={"cc_num": str},
    )
    validate_schema(frame, require_label=require_label)
    frame["cc_num"] = tokenize_cards(frame["cc_num"])
    return frame.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)


def temporal_split(frame: pd.DataFrame, *, valid_fraction: float = 0.2) -> pd.Series:
    """Boolean mask selecting the most recent `valid_fraction` of rows by date.

    A model that scores tomorrow's transactions has to be validated on a
    future slice of time, not a random sample of the past -- a random split
    lets a card's future transactions leak into the rows used to tune the
    model on that same card's past. Cutting on the date quantile instead of
    shuffling keeps every row in its true chronological position.
    """

    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1.")
    cutoff = frame["trans_date_trans_time"].quantile(1 - valid_fraction)
    return frame["trans_date_trans_time"] > cutoff


def split_by_recency(
    frame: pd.DataFrame,
    *,
    test_days: int = 14,
    valid_fraction: float = 0.2,
    valid_days: int | None = None,
) -> TemporalSplit:
    """Split one labeled history into train / validation / test by time, newest last.

    Retraining in production has no separate holdout file: the most recent `test_days` are
    held out as test, and the rest is cut into train/validation as in `make_split`.

    `valid_days` makes validation the last N days instead of the last `valid_fraction` of rows.
    Use it after a drift: 20% of an 18-month history is ~4 months, so a drift that began two
    weeks ago lands entirely in validation and the model never trains on it (measured here:
    PR-AUC 0.003 on the new pattern, vs 0.2-0.5 once those days are in training).
    """

    cutoff = frame["trans_date_trans_time"].max() - pd.Timedelta(days=test_days)
    recent = frame["trans_date_trans_time"] > cutoff
    older = frame.loc[~recent]
    if valid_days:
        is_valid = older["trans_date_trans_time"] > (
            older["trans_date_trans_time"].max() - pd.Timedelta(days=valid_days)
        )
    else:
        is_valid = temporal_split(older, valid_fraction=valid_fraction)
    return TemporalSplit(
        train=older.loc[~is_valid], valid=older.loc[is_valid], test=frame.loc[recent]
    )


def make_split(
    train_path: str | Path, test_path: str | Path, *, valid_fraction: float = 0.2
) -> TemporalSplit:
    """Load both CSVs and cut `fraudTrain.csv` into train/validation by date.

    `fraudTest.csv` is returned untouched as `test`: it is a later, disjoint
    time period the dataset ships as a holdout, and no row in it participates
    in fitting or model selection until the final evaluation.
    """

    train_full = load_transactions(train_path)
    test = load_transactions(test_path)
    is_valid = temporal_split(train_full, valid_fraction=valid_fraction)
    return TemporalSplit(train=train_full.loc[~is_valid], valid=train_full.loc[is_valid], test=test)
