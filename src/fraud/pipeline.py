"""Single orchestrator: raw CSVs in, a scored/calibrated/thresholded artifact out."""

from __future__ import annotations

import hashlib
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from .data import TemporalSplit, make_split
from .evaluation import binary_metrics
from .features import build_features
from .modeling import FEATURE_COLUMNS, build_pipeline
from .monitoring import build_reference
from .threshold import FP_COST, select_threshold_by_savings


@dataclass
class FraudModel:
    """A fitted pipeline plus the calibrator and threshold it was shipped with.

    A model's probabilities are only meaningful together with the calibrator
    that maps them to real fraud rates, and a threshold is only meaningful
    together with the model it was tuned against -- so the three travel as
    one artifact instead of three files that can drift apart.
    """

    pipeline: Pipeline
    calibrator: IsotonicRegression
    threshold: float
    fp_cost: float
    metadata: dict[str, Any] = field(default_factory=dict)  # lineage: version, git sha, data hash
    reference: dict[str, Any] = field(default_factory=dict)  # validation distributions for drift

    def predict_proba(self, frame: pd.DataFrame, history: pd.DataFrame | None = None) -> np.ndarray:
        """Calibrated fraud probability. `history` = the card traffic seen before `frame`."""

        return self.score_features(build_features(frame, history=history))

    def score_features(self, feat: pd.DataFrame) -> np.ndarray:
        """Calibrated probability from an already-built feature frame."""

        raw = self.pipeline.predict_proba(feat[FEATURE_COLUMNS])[:, 1]
        return self.calibrator.transform(raw)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> FraudModel:
        return joblib.load(path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _lib_versions() -> dict[str, str]:
    out = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm"):
        with suppress(PackageNotFoundError):
            out[name] = version(name)
    return out


def run_train(
    train_path: str | Path,
    test_path: str | Path,
    *,
    model_name: str,
    valid_fraction: float = 0.2,
    fp_cost: float = FP_COST,
    device: str | None = None,
) -> tuple[FraudModel, dict[str, Any]]:
    """Fit `model_name` on the temporal train split, calibrate and threshold on validation.

    Returns the artifact and a metrics dict for both validation and the
    untouched holdout, so a single call produces everything `cli.py train`
    needs to report and save.
    """

    split = make_split(train_path, test_path, valid_fraction=valid_fraction)
    hashes = {Path(p).name: _sha256(p) for p in (train_path, test_path)}
    return fit_from_split(
        split, model_name=model_name, fp_cost=fp_cost, device=device, data_sha256=hashes
    )


def recency_weights(times: pd.Series, half_life_days: float) -> np.ndarray:
    """Weight 1 for the newest row, 0.5 one half-life older, and so on.

    After a drift, a model fit on the full history is dominated by the old pattern: measured here,
    ~7,500 old-pattern frauds drown ~180 new ones (PR-AUC 0.07 vs 0.50 fit on recent data only).
    Decaying weights adapt faster without throwing the history away; the cost is forgetting
    patterns that stop appearing, which the simulation cannot measure.
    """

    age_days = (times.max() - times).dt.total_seconds() / 86400
    return (0.5 ** (age_days / half_life_days)).to_numpy()


def fit_from_split(
    split: TemporalSplit,
    *,
    model_name: str,
    fp_cost: float = FP_COST,
    device: str | None = None,
    data_sha256: dict[str, str] | None = None,
    half_life_days: float | None = None,
) -> tuple[FraudModel, dict[str, Any]]:
    """The fitting core of `run_train`, on frames already split by time.

    Retraining from production tables enters here with a recency split
    (`data.split_by_recency`) instead of the two Sparkov CSVs.
    """

    train_feat = build_features(split.train)
    # Validation and holdout are scored the way production would: with the cards'
    # earlier traffic as history, not as a cold start.
    valid_feat = build_features(split.valid, history=split.train)

    prevalence = float(train_feat["is_fraud"].mean())
    pipeline = build_pipeline(model_name, prevalence=prevalence, device=device)
    fit_params = {}
    if half_life_days:
        weights = recency_weights(train_feat["trans_date_trans_time"], half_life_days)
        fit_params["model__sample_weight"] = weights
    pipeline.fit(train_feat[FEATURE_COLUMNS], train_feat["is_fraud"], **fit_params)

    valid_raw = pipeline.predict_proba(valid_feat[FEATURE_COLUMNS])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(valid_raw, valid_feat["is_fraud"])
    valid_proba = calibrator.transform(valid_raw)

    threshold = select_threshold_by_savings(
        valid_feat["is_fraud"], valid_proba, valid_feat["amt"], fp_cost=fp_cost
    )
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    metadata = {
        "model": model_name,
        "version": f"{trained_at}-{_git_sha()}",
        "trained_at": trained_at,
        "git_sha": _git_sha(),
        "data_sha256": data_sha256 or {},
        "feature_columns": FEATURE_COLUMNS,
        "libs": _lib_versions(),
    }
    reference = build_reference(valid_feat, valid_proba, threshold=threshold, fp_cost=fp_cost)
    model = FraudModel(
        pipeline=pipeline,
        calibrator=calibrator,
        threshold=threshold,
        fp_cost=fp_cost,
        metadata=metadata,
        reference=reference,
    )

    test_proba = model.predict_proba(split.test, history=pd.concat([split.train, split.valid]))
    metrics = {
        "model": model_name,
        "version": metadata["version"],
        "half_life_days": half_life_days,
        "prevalence_train": prevalence,
        "threshold": threshold,
        "fp_cost": fp_cost,
        "valid": binary_metrics(valid_feat["is_fraud"], valid_proba, threshold=threshold),
        "test": binary_metrics(split.test["is_fraud"], test_proba, threshold=threshold),
    }
    return model, metrics


SCORE_COLUMNS = ["trans_num", "cc_num", "trans_date_trans_time", "amt"]


def run_score(
    model: FraudModel, frame: pd.DataFrame, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Score transactions and flag those above the frozen threshold.

    Returns only ids, amount, score and model version: the raw frame carries PII
    (name, street, dob) that downstream consumers of scores don't need.
    """

    proba = model.predict_proba(frame, history=history)
    return frame[SCORE_COLUMNS].assign(
        fraud_probability=proba,
        flagged=proba >= model.threshold,
        model_version=model.metadata.get("version", "unknown"),
    )
