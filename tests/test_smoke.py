"""End to end on synthetic data: train -> score unlabeled -> simulate with monitor."""

from fraud.data import load_transactions
from fraud.pipeline import SCORE_COLUMNS, run_score, run_train
from fraud.simulation import simulate


def test_train_score_unlabeled_and_simulate(csv_paths):
    train_p, test_p = csv_paths
    model, metrics = run_train(train_p, test_p, model_name="logreg")
    assert metrics["test"]["pr_auc"] > metrics["test"]["prevalence"]  # beats random ranking
    assert model.metadata["version"] and model.reference["numeric"]

    test = load_transactions(test_p)
    history = load_transactions(train_p)
    scored = run_score(model, test.drop(columns="is_fraud").head(200), history)
    assert list(scored.columns) == [*SCORE_COLUMNS, "fraud_probability", "flagged", "model_version"]
    assert "first" not in scored.columns  # no PII in outputs

    daily, alerts, _ = simulate(model, history, test, n_days=20, onset_day=10)
    assert len(daily) == 14 and set(alerts.columns) >= {"date", "rule", "severity"}


def test_retrain_from_a_single_labeled_history_uses_a_recency_holdout():
    from fraud.data import split_by_recency
    from fraud.pipeline import fit_from_split
    from tests.conftest import synthetic_frame

    frame = synthetic_frame(n_cards=30, days=60)
    split = split_by_recency(frame, test_days=14)
    assert split.train["trans_date_trans_time"].max() <= split.valid["trans_date_trans_time"].min()
    assert split.valid["trans_date_trans_time"].max() < split.test["trans_date_trans_time"].min()
    model, metrics = fit_from_split(split, model_name="logreg")
    assert metrics["test"]["pr_auc"] > metrics["test"]["prevalence"]


def test_recency_weights_halve_per_half_life_and_fit_accepts_them():
    import numpy as np
    import pandas as pd

    from fraud.data import split_by_recency
    from fraud.pipeline import fit_from_split, recency_weights
    from tests.conftest import synthetic_frame

    t = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-15", "2020-01-29"]))
    np.testing.assert_allclose(recency_weights(t, 14), [0.25, 0.5, 1.0])

    split = split_by_recency(synthetic_frame(n_cards=30, days=60), test_days=14)
    model, metrics = fit_from_split(split, model_name="logreg", half_life_days=14)
    assert metrics["half_life_days"] == 14 and model.metadata["version"]


def test_valid_days_keeps_recent_days_out_of_validation_only_when_asked():
    import pandas as pd

    from fraud.data import split_by_recency
    from tests.conftest import synthetic_frame

    frame = synthetic_frame(n_cards=30, days=60)
    by_fraction = split_by_recency(frame, test_days=10)
    by_days = split_by_recency(frame, test_days=10, valid_days=5)
    t = by_days.valid["trans_date_trans_time"]
    assert t.max() - t.min() <= pd.Timedelta(days=5)
    assert len(by_days.train) > len(by_fraction.train)  # recent days go to training now
    assert by_days.train["trans_date_trans_time"].max() <= t.min()
