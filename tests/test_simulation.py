import pandas as pd

from fraud.simulation import apply_scenario
from tests.conftest import synthetic_frame


def test_scenarios_are_deterministic_and_only_touch_post_onset_rows():
    f = synthetic_frame(n_cards=10, days=20)
    onset = pd.Timestamp("2020-01-10")
    a, b = (apply_scenario(f, "new_fraud_pattern", onset, seed=3) for _ in range(2))
    pd.testing.assert_frame_equal(a, b)
    early = a[a["trans_date_trans_time"] < onset]
    assert early["amt"].sum() <= f[f["trans_date_trans_time"] < onset]["amt"].sum() + 1e-6


def test_amount_inflation_scales_late_rows():
    f = synthetic_frame(n_cards=5, days=10)
    onset = pd.Timestamp("2020-01-05")
    out = apply_scenario(f, "amount_inflation", onset)
    assert (
        abs(
            out["amt"].sum()
            - (
                f.loc[f.trans_date_trans_time < onset, "amt"].sum()
                + 1.8 * f.loc[f.trans_date_trans_time >= onset, "amt"].sum()
            )
        )
        < 1e-3
    )


def test_frozen_state_rows_do_not_see_same_day_siblings():
    from fraud.features import build_features
    from fraud.simulation import build_features_frozen

    f = synthetic_frame(n_cards=3, days=6)
    day = pd.Timestamp("2020-01-05")
    ts = f["trans_date_trans_time"]
    past, today = f[ts < day], f[(ts >= day) & (ts < day + pd.Timedelta(days=1))]
    frozen = build_features_frozen(today, past)
    live = build_features(today, history=past)
    assert len(frozen) == len(live) == len(today)
    # Same-day siblings are invisible when frozen: windows can only see less than live ones.
    f_ = frozen.sort_values(["cc_num", "trans_date_trans_time"], ignore_index=True)
    assert (f_["tx_count_card_24h"] <= live["tx_count_card_24h"]).all()
    assert (f_["tx_count_card_24h"] < live["tx_count_card_24h"]).any()
    # ...and a card's first transaction of the day is identical either way.
    first = (
        today.groupby("cc_num")["trans_date_trans_time"].transform("min")
        == today["trans_date_trans_time"]
    )
    keys = today.loc[first, ["cc_num", "trans_date_trans_time"]]
    pd.testing.assert_frame_equal(
        f_.merge(keys)["tx_count_card_1h"].to_frame(),
        live.merge(keys)["tx_count_card_1h"].to_frame(),
    )
