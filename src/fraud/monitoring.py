"""Post-deployment monitoring with numpy/pandas only: drift, data quality, delayed performance.

Labels arrive days after the decision, so there are two clocks: input/score drift
is checkable the same day (no labels needed); recall and net savings are
checkable only on transactions old enough to have labels.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from .evaluation import binary_metrics
from .modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .threshold import threshold_sweep

PSI_WARN: Final[float] = 0.10
PSI_CRIT: Final[float] = 0.25  # conventional PSI bands: <0.1 stable, >0.25 major shift
ALERT_RATE_BAND: Final[tuple[float, float]] = (0.5, 2.0)  # x the validation alert rate
RECALL_DROP: Final[float] = 0.10  # absolute, vs validation recall
SAVINGS_DROP: Final[float] = 0.20  # relative, vs validation share of fraud dollars saved net
MIN_FRAUD_FOR_PERF: Final[int] = 30  # below this, recall is noise: report "insufficient labels"
UNSEEN = "__unseen__"


def psi(ref_props: np.ndarray, cur_props: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index between two bin-proportion vectors."""

    r = np.clip(np.asarray(ref_props, dtype=float), eps, None)
    c = np.clip(np.asarray(cur_props, dtype=float), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _numeric_props(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges[1:-1], x, side="right")
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def _categorical_props(x: pd.Series, levels: list[str]) -> np.ndarray:
    counts = x.astype(str).value_counts()
    seen = np.array([counts.get(level, 0) for level in levels], dtype=float)
    unseen = counts.sum() - seen.sum()
    out = np.append(seen, unseen)
    return out / max(out.sum(), 1.0)


def _savings_ratio(sweep: pd.DataFrame, y: np.ndarray, amt: pd.Series | np.ndarray) -> float:
    """Net savings as a share of all fraud dollars: unlike savings per transaction it does not
    move with prevalence (0.59% in validation vs 0.39% in the holdout)."""

    fraud_usd = float(np.asarray(amt, dtype=float)[y == 1].sum())
    return float(sweep["net_savings_usd"].iloc[0] / fraud_usd) if fraud_usd else 0.0


def build_reference(
    feat: pd.DataFrame, proba: np.ndarray, *, threshold: float, fp_cost: float, bins: int = 10
) -> dict[str, Any]:
    """Freeze validation distributions and performance; stored inside the model artifact."""

    ref: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for col in NUMERIC_FEATURES:
        x = feat[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        if len(edges) >= 3:  # constant/binary-like columns carry no PSI signal
            ref["numeric"][col] = {
                "edges": edges.tolist(),
                "props": _numeric_props(x, edges).tolist(),
            }
    for col in CATEGORICAL_FEATURES:
        levels = sorted(feat[col].astype(str).unique())
        ref["categorical"][col] = {
            "levels": levels,
            "props": _categorical_props(feat[col], levels).tolist(),
        }
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, bins + 1)))
    ref["score"] = {"edges": edges.tolist(), "props": _numeric_props(proba, edges).tolist()}

    y = feat["is_fraud"].to_numpy()
    m = binary_metrics(y, proba, threshold=threshold)
    sweep = threshold_sweep(y, proba, feat["amt"], fp_cost=fp_cost, grid=np.array([threshold]))
    ref["alert_rate"] = float((proba >= threshold).mean())
    ref["recall"] = m["recall"]
    ref["precision"] = m["precision"]
    ref["savings_ratio"] = _savings_ratio(sweep, y, feat["amt"])
    return ref


def drift_metrics(
    ref: dict[str, Any], feat: pd.DataFrame, proba: np.ndarray, *, threshold: float
) -> dict[str, float]:
    """Same-day, label-free checks: per-feature PSI, score PSI, alert rate, data quality."""

    out: dict[str, float] = {"n": float(len(feat))}
    for col, r in ref["numeric"].items():
        cur = _numeric_props(feat[col].to_numpy(dtype=float), np.asarray(r["edges"]))
        out[f"psi_{col}"] = psi(np.asarray(r["props"]), cur)
    for col, r in ref["categorical"].items():
        cur = _categorical_props(feat[col], r["levels"])
        out[f"psi_{col}"] = psi(np.asarray(r["props"]), cur)
        out[f"unseen_{col}"] = float(cur[-1])
    r = ref["score"]
    out["psi_score"] = psi(np.asarray(r["props"]), _numeric_props(proba, np.asarray(r["edges"])))
    out["alert_rate"] = float((proba >= threshold).mean())
    out["null_rate"] = float(feat[NUMERIC_FEATURES].isna().to_numpy().mean())
    return out


def delayed_performance(
    labeled: pd.DataFrame, *, threshold: float, fp_cost: float
) -> dict[str, float] | None:
    """Recall/precision/savings on matured transactions (`is_fraud`, `fraud_probability`, `amt`).

    Returns None when there are too few frauds for the numbers to mean anything.
    """

    y = labeled["is_fraud"].to_numpy()
    if y.sum() < MIN_FRAUD_FOR_PERF:
        return None
    proba = labeled["fraud_probability"].to_numpy()
    m = binary_metrics(y, proba, threshold=threshold)
    sweep = threshold_sweep(y, proba, labeled["amt"], fp_cost=fp_cost, grid=np.array([threshold]))
    return {
        "n_labeled": float(len(y)),
        "n_fraud": float(y.sum()),
        "pr_auc": m["pr_auc"],
        "recall": m["recall"],
        "precision": m["precision"],
        "savings_ratio": _savings_ratio(sweep, y, labeled["amt"]),
    }


def evaluate_rules(
    ref: dict[str, Any], drift: dict[str, float], perf: dict[str, float] | None
) -> list[dict[str, Any]]:
    """Turn metrics into alerts: [{rule, severity, value, limit}]. Empty list = healthy."""

    alerts: list[dict[str, Any]] = []

    def add(rule: str, severity: str, value: float, limit: float) -> None:
        alerts.append({"rule": rule, "severity": severity, "value": value, "limit": limit})

    for key, value in drift.items():
        if key.startswith("psi_"):
            if value > PSI_CRIT:
                add(key, "critical", value, PSI_CRIT)
            elif value > PSI_WARN:
                add(key, "warning", value, PSI_WARN)
    lo, hi = ALERT_RATE_BAND
    ratio = drift["alert_rate"] / ref["alert_rate"] if ref["alert_rate"] else float("inf")
    if not lo <= ratio <= hi:
        add("alert_rate_ratio", "critical", ratio, hi if ratio > hi else lo)
    if drift["null_rate"] > 0:
        add("null_rate", "warning", drift["null_rate"], 0.0)
    if perf is not None:
        if perf["recall"] < ref["recall"] - RECALL_DROP:
            add("recall", "critical", perf["recall"], ref["recall"] - RECALL_DROP)
        floor = ref["savings_ratio"] * (1 - SAVINGS_DROP)
        if perf["savings_ratio"] < floor:
            add("savings_ratio", "critical", perf["savings_ratio"], floor)
    return alerts
