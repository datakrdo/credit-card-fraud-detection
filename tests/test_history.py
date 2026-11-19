"""Features computed with `history=` must equal the slice of features over the full frame."""

import pandas as pd

from fraud.features import build_features
from tests.conftest import synthetic_frame


def test_history_matches_full_frame_features():
    full = synthetic_frame(n_cards=10, days=20)
    cut = full["trans_date_trans_time"].quantile(0.7)
    past, new = (
        full[full["trans_date_trans_time"] <= cut],
        full[full["trans_date_trans_time"] > cut],
    )

    expected = build_features(full).loc[new.index].reset_index(drop=True)
    got = build_features(new, history=past)

    cols = [c for c in expected.columns if c.startswith(("tx_count", "amt_", "seconds", "is_new"))]
    pd.testing.assert_frame_equal(got[cols], expected[cols], check_dtype=False)


def test_without_history_first_rows_are_cold():
    new = synthetic_frame(n_cards=5, days=5)
    assert (build_features(new).groupby("cc_num")["is_new_merchant_for_card"].first()).all()
