import numpy as np
import pandas as pd

from fraud.features import build_features
from fraud.monitoring import (
    build_reference,
    delayed_performance,
    drift_metrics,
    evaluate_rules,
    psi,
)
from tests.conftest import synthetic_frame


def _ref_and_feat():
    feat = build_features(synthetic_frame(n_cards=30, days=40))
    rng = np.random.default_rng(0)
    proba = np.clip(feat["is_fraud"] * 0.6 + rng.random(len(feat)) * 0.3, 0, 1).to_numpy()
    return feat, proba, build_reference(feat, proba, threshold=0.5, fp_cost=4.0)


def test_psi_zero_for_same_and_large_for_shifted():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(p, p) == 0
    assert psi(p, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def test_no_alerts_on_reference_data_and_alert_on_shift():
    feat, proba, ref = _ref_and_feat()
    ok = drift_metrics(ref, feat, proba, threshold=0.5)
    assert evaluate_rules(ref, ok, None) == []

    shifted = feat.assign(amt=feat["amt"] * 3, log_amt=np.log1p(feat["amt"] * 3))
    bad = drift_metrics(ref, shifted, proba, threshold=0.5)
    assert any(
        a["rule"] == "psi_amt" and a["severity"] == "critical"
        for a in evaluate_rules(ref, bad, None)
    )


def test_unseen_category_is_reported():
    feat, proba, ref = _ref_and_feat()
    m = drift_metrics(ref, feat.assign(category="brand_new"), proba, threshold=0.5)
    assert m["unseen_category"] == 1.0


def test_recall_drop_triggers_and_too_few_frauds_skip():
    feat, proba, ref = _ref_and_feat()
    labeled = pd.DataFrame(
        {"is_fraud": feat["is_fraud"], "fraud_probability": 0.0, "amt": feat["amt"]}
    )
    perf = delayed_performance(labeled, threshold=0.5, fp_cost=4.0)
    drift = drift_metrics(ref, feat, proba, threshold=0.5)
    assert any(a["rule"] == "recall" for a in evaluate_rules(ref, drift, perf))
    assert delayed_performance(labeled.head(5), threshold=0.5, fp_cost=4.0) is None
