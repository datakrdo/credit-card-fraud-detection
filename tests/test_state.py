"""Online path (card state) must give the same features and scores as batch history."""

import numpy as np
import pandas as pd

from fraud.features import build_features
from fraud.modeling import FEATURE_COLUMNS
from fraud.pipeline import run_train
from fraud.state import TX_COLUMNS, build_card_state, history_from_state, score_with_state
from tests.conftest import synthetic_frame


def _split():
    full = synthetic_frame(n_cards=12, days=30)
    cut = full["trans_date_trans_time"].quantile(0.8)
    return full[full["trans_date_trans_time"] <= cut], full[full["trans_date_trans_time"] > cut]


def test_state_history_features_equal_batch_history_features():
    past, new = _split()
    batch = build_features(new, history=past)
    online = build_features(new, history=history_from_state(build_card_state(past)))
    cols = [c for c in FEATURE_COLUMNS if c not in ("category", "state")]
    pd.testing.assert_frame_equal(online[cols], batch[cols], check_dtype=False)


def test_score_with_state_matches_run_score_and_keeps_request_order(csv_paths):
    train_p, test_p = csv_paths
    model, _ = run_train(train_p, test_p, model_name="logreg")
    past, new = _split()
    new = new.sample(50, random_state=0)  # shuffled: request order != (card, time) order
    state = build_card_state(past).set_index("cc_num")
    request = new[TX_COLUMNS].join(state, on="cc_num")

    got = score_with_state(model, request)
    want = model.predict_proba(new.sort_values(["cc_num", "trans_date_trans_time"]), history=past)
    expect = pd.Series(want, index=new.sort_values(["cc_num", "trans_date_trans_time"]).index)
    np.testing.assert_allclose(got, expect.loc[new.index].to_numpy(), rtol=1e-9)
