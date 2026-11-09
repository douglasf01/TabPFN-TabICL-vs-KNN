"""Baseline accuracy, balanced accuracy, F1, MCC and ROC AUC of every model on every dataset.

Usage:
    uv run python scripts/run_baseline.py --models knn
    uv run python scripts/run_baseline.py --models knn tabpfn tabicl --seeds 0 1 2
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from tabknn.data import DATASETS, get_split
from tabknn.evaluate import full_context_metrics
from tabknn.models import MODEL_NAMES

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=[d for d, i in DATASETS.items() if i is not None])
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=1)
    p.add_argument("--output", type=Path, default=RESULTS / "baseline.csv")
    args = p.parse_args()
    if args.n_estimators < 1:
        p.error("--n-estimators must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.datasets:
        for seed in args.seeds:
            X_train, X_test, y_train, y_test = get_split(name, seed=seed)
            for model in args.models:
                metrics = full_context_metrics(model, X_train, X_test, y_train, y_test,
                                               k=args.k, seed=seed, n_estimators=args.n_estimators)
                rows.append(dict(dataset=name, seed=seed, model=model, **metrics,
                                 n_estimators=1 if model == "knn" else args.n_estimators))
                # Save each completed case so a failed run retains its results.
                pd.DataFrame(rows).to_csv(args.output, index=False)
                print(f"{name:22s} seed={seed} {model:7s} acc={metrics['accuracy']:.4f} "
                      f"f1_macro={metrics['f1_macro']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(args.output, index=False)
    metric_columns = ["accuracy", "balanced_accuracy", "f1_macro", "f1_minority", "mcc", "roc_auc"]
    summary = df.groupby(["dataset", "model"])[metric_columns].agg(["mean", "std", "count"])
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"))
    metadata = dict(seeds=args.seeds, models=args.models, datasets=args.datasets,
                    foundation_n_estimators=args.n_estimators, knn_k=args.k,
                    completed_runs=len(df),
                    versions={name: version(name) for name in ["tabpfn", "tabicl", "torch", "scikit-learn"]})
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(df.groupby(["dataset", "model"])[["accuracy", "f1_macro"]].mean().unstack().round(4))


if __name__ == "__main__":
    main()
