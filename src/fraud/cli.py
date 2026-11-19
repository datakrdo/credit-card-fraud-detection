#!/usr/bin/env python3
"""Command-line entry points: `fraud train`, `fraud score`, `fraud evaluate`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_transactions
from .evaluation import binary_metrics
from .modeling import MODEL_SPECS
from .pipeline import FraudModel, run_score, run_train
from .simulation import SCENARIOS, simulate
from .threshold import FP_COST


def _train(args: argparse.Namespace) -> None:
    model, metrics = run_train(
        args.data, args.test_data, model_name=args.model, fp_cost=args.fp_cost, device=args.device
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.model}.joblib"
    model.save(model_path)
    (out_dir / f"{args.model}-metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {model_path}")
    print(json.dumps(metrics, indent=2))


def _score(args: argparse.Namespace) -> None:
    model = FraudModel.load(args.model)
    frame = load_transactions(args.transactions, require_label=False)
    history = load_transactions(args.history) if args.history else None
    scored = run_score(model, frame, history)
    scored.to_csv(args.out, index=False)
    print(f"Scored {len(scored)} transactions -> {args.out} ({scored['flagged'].sum()} flagged)")


def _evaluate(args: argparse.Namespace) -> None:
    model = FraudModel.load(args.model)
    frame = load_transactions(args.data)
    history = load_transactions(args.history) if args.history else None
    proba = model.predict_proba(frame, history)
    metrics = binary_metrics(frame["is_fraud"], proba, threshold=model.threshold)
    print(json.dumps(metrics, indent=2))


def _simulate(args: argparse.Namespace) -> None:
    model = FraudModel.load(args.model)
    history = load_transactions(args.data)
    test = load_transactions(args.test_data)
    daily, alerts, _ = simulate(
        model,
        history,
        test,
        n_days=args.days,
        scenario=args.scenario,
        onset_day=args.onset_day,
        start_day=args.start_day,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / f"daily_metrics_{args.scenario}.csv", index=False)
    alerts.to_csv(out / f"alerts_{args.scenario}.csv", index=False)
    print(f"{len(daily)} days simulated, {len(alerts)} alerts -> {out}/")
    print(alerts.value_counts(["rule", "severity"]).to_string() if len(alerts) else "no alerts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Fit a model and freeze its threshold on validation.")
    train_p.add_argument("--data", default="data/raw/fraudTrain.csv")
    train_p.add_argument("--test-data", default="data/raw/fraudTest.csv")
    train_p.add_argument("--model", default="xgboost", choices=sorted(MODEL_SPECS))
    train_p.add_argument("--fp-cost", type=float, default=FP_COST)
    train_p.add_argument("--device", choices=["cpu", "cuda"], help="override booster GPU default")
    train_p.add_argument("--out", default="models")
    train_p.set_defaults(func=_train)

    score_p = sub.add_parser("score", help="Batch-score a CSV of raw Sparkov-schema transactions.")
    score_p.add_argument("transactions")
    score_p.add_argument("--model", required=True)
    score_p.add_argument("--history", help="earlier labeled/unlabeled traffic for card history")
    score_p.add_argument("-o", "--out", default="scores.csv")
    score_p.set_defaults(func=_score)

    eval_p = sub.add_parser("evaluate", help="Report metrics for a saved model on a labeled CSV.")
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--data", default="data/raw/fraudTest.csv")
    eval_p.add_argument("--history", default="data/raw/fraudTrain.csv")
    eval_p.set_defaults(func=_evaluate)

    sim_p = sub.add_parser("simulate", help="Replay fraudTest as new traffic and run the monitor.")
    sim_p.add_argument("--model", required=True)
    sim_p.add_argument("--data", default="data/raw/fraudTrain.csv", help="history")
    sim_p.add_argument("--test-data", default="data/raw/fraudTest.csv")
    sim_p.add_argument("--scenario", default="none", choices=SCENARIOS)
    sim_p.add_argument("--days", type=int, default=45)
    sim_p.add_argument("--onset-day", type=int, default=21)
    sim_p.add_argument("--start-day", type=int, default=0)
    sim_p.add_argument("--out", default="outputs/sim")
    sim_p.set_defaults(func=_simulate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
