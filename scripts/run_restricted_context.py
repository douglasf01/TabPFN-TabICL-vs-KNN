"""Accuracy when TabPFN/TabICL only see the k nearest training points.

Usage:
    uv run python scripts/run_restricted_context.py --models tabpfn --ks 5 10 25 50
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from tabknn.data import DATASETS, get_split
from tabknn.evaluate import restricted_context_accuracy
from tabknn.models import cached_model_weights

RESULTS = Path(__file__).resolve().parent.parent / "results"


def initialize_worker():
    import torch
    torch.set_num_threads(1)
    global weight_cache
    weight_cache = cached_model_weights()
    weight_cache.__enter__()


def run_cases(task):
    name, seed, model, ks, n_estimators, output_dir = task
    X_train, X_test, y_train, y_test = get_split(name, seed=seed)
    rows = []
    for k in ks:
        case = f"{name}_seed{seed}_{model}_k{k}"
        r = restricted_context_accuracy(model, X_train, X_test, y_train, y_test,
            k=k, seed=seed, n_estimators=n_estimators,
            checkpoint_path=output_dir / "restricted_checkpoints" / f"{case}.npz")
        rows.append(dict(dataset=name, seed=seed, model=model, k=k, n_estimators=n_estimators, **r))
        print(f"Completed {case}: accuracy={r['accuracy']:.4f}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=[d for d, i in DATASETS.items() if i is not None])
    p.add_argument("--models", nargs="+", default=["tabpfn", "tabicl"])
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10, 25, 50])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--n-estimators", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=RESULTS / "final_8seeds")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "restricted_context.csv"
    rows = pd.read_csv(output).to_dict("records") if output.exists() else []
    if any(r["n_estimators"] != args.n_estimators for r in rows):
        raise ValueError("Output contains a different estimator count")
    done = {(r["dataset"], r["seed"], r["model"], r["k"]) for r in rows}
    tasks = []
    for name in args.datasets:
        for seed in args.seeds:
            for model in args.models:
                ks = [k for k in args.ks if (name, seed, model, k) not in done]
                if ks:
                    tasks.append((name, seed, model, ks, args.n_estimators, args.output_dir))
    if args.workers == 1:
        initialize_worker()
        for task in tasks:
            rows.extend(run_cases(task))
            pd.DataFrame(rows).to_csv(output, index=False)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=initialize_worker) as pool:
            for future in as_completed([pool.submit(run_cases, task) for task in tasks]):
                rows.extend(future.result())
                pd.DataFrame(rows).to_csv(output, index=False)
                print(f"Saved {len(rows)} restricted-context cases", flush=True)
    pd.DataFrame(rows).groupby(["dataset", "model", "k"]).agg(
        mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        seeds=("seed", "nunique"), model_used=("frac_model_used", "mean")
    ).to_csv(args.output_dir / "restricted_context_summary.csv")
    (args.output_dir / "restricted_context.json").write_text(json.dumps(dict(
        seeds=args.seeds, n_estimators=args.n_estimators, ks=args.ks, completed_runs=len(rows),
        fixed_weight_cache=True, versions={p: version(p) for p in ["tabpfn", "tabicl", "torch"]}), indent=2))


if __name__ == "__main__":
    main()
