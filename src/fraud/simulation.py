"""Replay `fraudTest.csv` day by day as if it were new production traffic, optionally drifted.

Each scenario injects one kind of drift from `onset_day` on, so the monitor's rules can
be seen firing (or staying quiet for `none`). Labels are only used `label_delay_days`
after the transaction, as in production.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from .features import build_features
from .monitoring import delayed_performance, drift_metrics, evaluate_rules
from .pipeline import FraudModel

SCENARIOS: Final[tuple[str, ...]] = (
    "none",
    "amount_inflation",  # covariate drift: every amount x1.8 (inflation / new merchant mix)
    "new_fraud_pattern",  # concept drift: fraud becomes small, daytime -> model goes blind
    "category_shift",  # covariate drift: 40% of traffic jumps to one category (shopping_net)
)


def apply_scenario(frame: pd.DataFrame, scenario: str, onset: pd.Timestamp, seed: int = 41):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}.")
    out = frame.copy()
    rng = np.random.default_rng(seed)
    late = out["trans_date_trans_time"] >= onset
    if scenario == "amount_inflation":
        out.loc[late, "amt"] = out.loc[late, "amt"] * 1.8
    elif scenario == "new_fraud_pattern":
        m = late & (out["is_fraud"] == 1)
        day = out.loc[m, "trans_date_trans_time"].dt.normalize()
        offset = pd.to_timedelta(rng.integers(9 * 60, 18 * 60, m.sum()), unit="m")
        out.loc[m, "trans_date_trans_time"] = day + offset
        out.loc[m, "amt"] = rng.uniform(20, 80, m.sum()).round(2)
    elif scenario == "category_shift":
        m = late & (rng.random(len(out)) < 0.4)
        out.loc[m, "category"] = "shopping_net"
    return out.sort_values(["cc_num", "trans_date_trans_time"], ignore_index=True)


def build_features_frozen(day_rows: pd.DataFrame, past: pd.DataFrame) -> pd.DataFrame:
    """Features as the online path sees them with a card state refreshed once a day.

    Each row sees only `past` (everything before today), never earlier transactions of the
    same day: that is what a daily-refreshed state serves. Done in rounds of one row per
    card (the k-th transaction of each card today), so no row has a same-day sibling in
    its batch. Slow by design; it exists to measure the cost of staleness, not to serve.
    """

    ordinal = day_rows.groupby("cc_num").cumcount()
    rounds = [
        build_features(day_rows[ordinal == k], history=past) for k in range(int(ordinal.max()) + 1)
    ]
    return pd.concat(rounds, ignore_index=True)


def simulate(
    model: FraudModel,
    history: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_days: int = 45,
    scenario: str = "none",
    onset_day: int = 21,
    start_day: int = 0,
    label_delay_days: int = 7,
    perf_window_days: int = 14,
    state_freeze: bool = False,  # True: score with a card state refreshed once a day (see above)
    drift_window_days: int = 7,  # one full week, or Mon/Tue volume spikes read as drift
    seed: int = 41,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (daily_metrics, alerts, scored). Features are built once over the whole window.

    That is equivalent to day-by-day scoring because every feature is causal (a row never
    sees later rows), which `tests/test_features.py` enforces.
    """

    day0 = test["trans_date_trans_time"].min().normalize() + pd.Timedelta(days=start_day)
    end = day0 + pd.Timedelta(days=n_days)
    ts = test["trans_date_trans_time"]
    before = test.loc[ts < day0]
    window = test.loc[(ts >= day0) & (ts < end)]
    sim = apply_scenario(window, scenario, day0 + pd.Timedelta(days=onset_day), seed)

    base = pd.concat([history, before])
    if state_freeze:
        sim_day = (sim["trans_date_trans_time"].dt.normalize() - day0).dt.days
        feat = pd.concat(
            [
                build_features_frozen(sim[sim_day == d], pd.concat([base, sim[sim_day < d]]))
                for d in range(n_days)
                if (sim_day == d).any()
            ],
            ignore_index=True,
        )
    else:
        feat = build_features(sim, history=base)
    proba = model.score_features(feat)
    scored = feat[["trans_num", "cc_num", "trans_date_trans_time", "amt", "is_fraud"]].assign(
        fraud_probability=proba, flagged=proba >= model.threshold
    )

    day_idx = (feat["trans_date_trans_time"].dt.normalize() - day0).dt.days.to_numpy()
    rows, alerts = [], []
    for d in range(drift_window_days - 1, n_days):  # warm-up: need a full window
        mask = (day_idx > d - drift_window_days) & (day_idx <= d)
        if not (day_idx == d).any():
            continue
        drift = drift_metrics(model.reference, feat[mask], proba[mask], threshold=model.threshold)
        hi = d - label_delay_days  # newest day whose labels have arrived
        matured = (day_idx > hi - perf_window_days) & (day_idx <= hi)
        perf = delayed_performance(
            scored[matured], threshold=model.threshold, fp_cost=model.fp_cost
        )
        date = (day0 + pd.Timedelta(days=d)).date()
        rows.append({"date": date, "scenario": scenario, **drift, **(perf or {})})
        alerts += [{"date": date, **a} for a in evaluate_rules(model.reference, drift, perf)]
    cols = ["date", "rule", "severity", "value", "limit"]
    return pd.DataFrame(rows), pd.DataFrame(alerts, columns=cols), scored
