"""Synthetic Sparkov-schema traffic so end-to-end tests run in CI without the Kaggle CSVs."""

import numpy as np
import pandas as pd
import pytest

from fraud.data import EXPECTED_COLUMNS


def synthetic_frame(n_cards: int = 40, days: int = 60, start: str = "2020-01-01", seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_cards * days * 3
    times = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days * 86400, n), unit="s")
    fraud = rng.random(n) < 0.03
    night = times.hour.isin([22, 23, 0, 1, 2, 3])
    amt = np.where(fraud, rng.uniform(200, 900, n), rng.lognormal(3.5, 0.8, n)).round(2)
    # fraud clusters at night so a model can actually learn something
    times = times.where(~(fraud & ~night), times.normalize() + pd.Timedelta(hours=23))
    return pd.DataFrame(
        {
            "trans_date_trans_time": times,
            "cc_num": rng.integers(0, n_cards, n).astype(str),
            "merchant": rng.choice([f"m{i}" for i in range(15)], n),
            "category": rng.choice(["grocery_pos", "shopping_net", "gas_transport", "travel"], n),
            "amt": amt,
            "first": "a",
            "last": "b",
            "gender": "F",
            "street": "s",
            "city": "c",
            "state": rng.choice(["NY", "CA", "TX"], n),
            "zip": 1,
            "lat": 40.0,
            "long": -70.0,
            "city_pop": 1000,
            "job": "j",
            "dob": pd.Timestamp("1980-01-01"),
            "trans_num": [f"t{start}{i}" for i in range(n)],
            "unix_time": 0,
            "merch_lat": 40.1,
            "merch_long": -70.1,
            "is_fraud": fraud.astype(int),
        }
    )[EXPECTED_COLUMNS].sort_values(["cc_num", "trans_date_trans_time"], ignore_index=True)


@pytest.fixture(scope="session")
def csv_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    train, test = d / "train.csv", d / "test.csv"
    synthetic_frame(days=60, seed=1).to_csv(train)
    synthetic_frame(days=45, start="2020-03-15", seed=2).to_csv(test)
    return train, test
