"""Attention/KNN comparison from the project report, with resumable artifacts.

Usage: uv run python scripts/run_attention.py
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

from tabknn.attention import extract_attention
from tabknn.data import DATASETS, get_split
from tabknn.models import make_model
from tabknn.neighbors import knn_indices, overlap_at_k, top_k_indices

RESULTS = Path(__file__).resolve().parent.parent / "results" / "attention"


def rank_correlation(weights, distances):
    """Per-query Spearman correlation: high attention versus small distance."""
    a = rankdata(weights, axis=1)
    b = rankdata(-distances, axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    norm = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return np.divide((a * b).sum(axis=1), norm,
                     out=np.full(len(a), np.nan), where=norm > 0)


def make_outputs():
    frames = [pd.read_csv(p) for p in sorted(RESULTS.glob("*/queries.csv"))]
    if not frames:
        return
    queries = pd.concat(frames, ignore_index=True)
    estimator_counts = sorted({count for path in RESULTS.glob("*/metadata.json")
                               for count in json.loads(path.read_text())["ensemble_members_per_layer"].values()})
    group = ["dataset", "seed", "model", "layer", "k"]
    metrics = ["overlap", "random_overlap", "knn_attention_mass", "attention_label_purity",
               "knn_label_purity", "distance_rank_correlation", "model_correct", "knn_correct"]
    summary = queries.groupby(group)[metrics].mean().reset_index()
    summary["overlap_above_random"] = summary.overlap - summary.random_overlap
    summary.to_csv(RESULTS / "summary.csv", index=False)
    aggregate = summary.groupby(["dataset", "model", "layer", "k"])[
        ["overlap", "random_overlap", "overlap_above_random", "knn_attention_mass",
         "distance_rank_correlation", "attention_label_purity", "knn_label_purity"]
    ].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join(filter(None, c)) for c in aggregate.columns]
    aggregate.to_csv(RESULTS / "across_seeds.csv", index=False)
    last = summary[summary.layer == summary.groupby("model").layer.transform("max")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, model in zip(axes, ["tabpfn", "tabicl"]):
        subset = last[last.model == model]
        for dataset, rows in subset.groupby("dataset"):
            means = rows.groupby("k").overlap.agg(["mean", "std"])
            ax.errorbar(means.index, means["mean"], yerr=means["std"].fillna(0),
                        marker="o", capsize=3, label=dataset)
        random = subset.groupby("k").random_overlap.mean()
        ax.plot(random.index, random, "k--", label="Random (dataset mean)")
        ax.set(title=f"{model}: final ICL layer", xlabel="k", ylabel="Mean overlap")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "overlap_by_k.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, model in zip(axes, ["tabpfn", "tabicl"]):
        subset = summary[(summary.model == model) & (summary.k == 10)]
        for dataset, rows in subset.groupby("dataset"):
            means = rows.groupby("layer").overlap.agg(["mean", "std"])
            ax.plot(means.index, means["mean"], label=dataset)
            ax.fill_between(means.index, means["mean"] - means["std"].fillna(0),
                            means["mean"] + means["std"].fillna(0), alpha=0.12)
        ax.set(title=f"{model}: k=10", xlabel="ICL layer (zero based)", ylabel="Mean overlap")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "overlap_by_layer.png", dpi=180)
    plt.close(fig)

    final_queries = queries[queries.layer == queries.groupby("model").layer.transform("max")].copy()
    final_queries["outcome"] = np.select(
        [final_queries.model_correct.astype(bool) & ~final_queries.knn_correct.astype(bool),
         ~final_queries.model_correct.astype(bool) & final_queries.knn_correct.astype(bool),
         final_queries.model_correct.astype(bool) & final_queries.knn_correct.astype(bool)],
        ["model_only_correct", "knn_only_correct", "both_correct"], default="both_wrong")
    divergence = final_queries.groupby(["dataset", "model", "k", "outcome"]).agg(
        n_queries=("overlap", "size"), overlap=("overlap", "mean"),
        knn_attention_mass=("knn_attention_mass", "mean"))
    divergence.to_csv(RESULTS / "divergence.csv")

    # Quantify the duplicate/tie caveat instead of relying only on prose.
    diagnostics = []
    for (dataset, seed), runs in summary.groupby(["dataset", "seed"]):
        X, T, _, _ = get_split(dataset, seed=int(seed))
        train_rows = {row.tobytes() for row in X}
        exact_match = np.mean([row.tobytes() in train_rows for row in T])
        distances = np.sort(pairwise_distances(T, X), axis=1)
        for model, model_runs in runs.groupby("model"):
            target = RESULTS / f"{dataset}_seed{seed}_{model}"
            with np.load(target / "weights.npz") as archive:
                weights = archive[f"layer_{int(model_runs.layer.max())}"]
            ranked = np.sort(weights, axis=1)[:, ::-1]
            for k in model_runs.k.unique():
                k = int(k)
                diagnostics.append(dict(dataset=dataset, seed=seed, model=model, k=k,
                                        n_train=len(X), n_test=len(T),
                                        duplicate_train_rows=len(X) - len(train_rows),
                                        fraction_test_exact_train_match=exact_match,
                                        fraction_knn_boundary_tie=float(np.isclose(
                                            distances[:, k-1], distances[:, k], rtol=1e-6, atol=1e-7).mean()) if k < len(X) else 0,
                                        fraction_attention_boundary_tie=float((
                                            ranked[:, k-1] == ranked[:, k]).mean()) if k < len(X) else 0))
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(RESULTS / "tie_diagnostics.csv", index=False)

    table = last[last.k == 10].groupby(["dataset", "model"])[
        ["overlap", "random_overlap", "knn_attention_mass", "distance_rank_correlation"]].mean().reset_index()
    lines = ["# Attention versus nearest neighbors", "",
             f"Completed {summary[['dataset', 'seed', 'model']].drop_duplicates().shape[0]} dataset/seed/model runs.",
             "", "All available test rows; stratified 80/20 splits; preprocessing fitted on training data.",
             f"Seeds: {sorted(summary.seed.unique().tolist())}. Estimator counts per run: {estimator_counts}. Same splits as the matching baseline. ICL attention is averaged over heads and",
             "ensemble members per layer. Top-k is selected after averaging. Feature attention is excluded.",
             "", "## Final-layer results at k=10", "",
             "Means across available seeds. Attention mass is the probability assigned to the 10 nearest rows.",
             "Positive rank correlation means higher attention tends to coincide with smaller distances.", "",
             "| Dataset | Model | Overlap | Random expectation | KNN attention mass | Distance rank correlation |",
             "|---|---|---:|---:|---:|---:|"]
    for row in table.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.model} | {row.overlap:.3f} | {row.random_overlap:.3f} | {row.knn_attention_mass:.3f} | {row.distance_rank_correlation:.3f} |")
    lines += ["", "## Findings", ""]
    for model, rows in table.groupby("model"):
        lines.append(f"- {model}: final-layer top-10 overlap ranges from {rows.overlap.min():.1%} to {rows.overlap.max():.1%} across datasets; the nearest ten rows receive {rows.knn_attention_mass.min():.1%} to {rows.knn_attention_mass.max():.1%} of attention mass.")
        model_layers = summary[(summary.model == model) & (summary.k == 10)]
        first = model_layers[model_layers.layer == model_layers.layer.min()].overlap.mean()
        final = model_layers[model_layers.layer == model_layers.layer.max()].overlap.mean()
        lines.append(f"- {model}: mean top-10 overlap is {first:.1%} in the first ICL layer and {final:.1%} in the final layer (equal weight per dataset/seed). Intermediate layers vary non-monotonically.")
    if (table.overlap > table.random_overlap).all():
        lines.append("- Every dataset/model final-layer top-10 mean is above the random-selection expectation. This is descriptive evidence of locality, without a formal significance test.")
    lines += ["- Ranking agreement and attention concentration measure different things: high top-k overlap does not imply the model restricts attention to those k rows.",
              "- The layer curves show whether locality persists through the network. Final-layer results were fixed in advance as the main summary; all layers remain available for inspection.",
              "", "## Duplicate and tie diagnostics", "",
              "Means across seeds; distance boundary ties use rtol=1e-6 and atol=1e-7.", "",
              "| Dataset | Duplicate training rows | Test rows exactly matching training | KNN top-10 boundary ties |",
              "|---|---:|---:|---:|"]
    diag = diagnostics[(diagnostics.model == "tabpfn") & (diagnostics.k == 10)]
    for dataset, rows in diag.groupby("dataset"):
        lines.append(f"| {dataset} | {rows.duplicate_train_rows.mean():.1f} | {rows.fraction_test_exact_train_match.mean():.1%} | {rows.fraction_knn_boundary_tie.mean():.1%} |")
    lines += ["", "## Where predictions diverge from KNN", "",
              "Final-layer overlap at k=5, matching the baseline KNN. Pooled query counts across all datasets",
              "and seeds; repeated observations across overlapping splits are included. These groups are",
              "descriptive and confounded by dataset composition, not evidence of a causal explanation.", "",
              "| Model | Prediction outcome | Query count | Mean top-5 overlap |",
              "|---|---|---:|---:|"]
    divergence_rows = divergence.reset_index()
    for (model, outcome), rows in divergence_rows[divergence_rows.k == 5].groupby(["model", "outcome"]):
        count = rows.n_queries.sum()
        mean = (rows.overlap * rows.n_queries).sum() / count
        lines.append(f"| {model} | {outcome} | {count} | {mean:.3f} |")
    lines += ["", "## Interpretation and limits", "",
              "Random expected overlap is k / n_train, from independent uniformly selected sets of size k.",
              "Above-random overlap supports an association between attention and Euclidean locality;",
              "it does not show that attention causes the prediction or measures training-point importance.",
              "Attention rankings depend on layer and head aggregation. See overlap_by_layer.png.",
              "Error bars/bands are standard deviations across seeds, not confidence intervals.",
              "The seed splits overlap, so their observations are not independent replications.",
              "", "Exact duplicates and equal-distance ties can make index-set overlap ambiguous, especially",
              "for Maternal Health Risk. No duplicates were removed, to retain the baseline splits.",
              "KNN ties use sklearn's ordering; attention ties use stable training-index order.",
              "This is an exploratory comparison; do not select a favorable layer using test labels.",
              "divergence.csv groups final-layer overlap by model/KNN correctness; its KNN uses the listed k,",
              "whereas the original baseline uses k=5. Label purity uses test labels only for analysis.",
              "", "## Artifacts", "",
              "- summary.csv: per-seed, per-layer, per-k metrics.",
              "- across_seeds.csv: means and seed standard deviations.",
              "- divergence.csv: model/KNN disagreement groups.",
              "- tie_diagnostics.csv: exact duplicates and top-k boundary ambiguity.",
              "- Each run directory: weights.npz, queries.csv, metadata.json.",
              "- weights.npz preserves test/train ordering from get_split(dataset, seed).",
              "- metadata.json records versions, architecture, accuracy, and prediction checks."]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Updated {RESULTS / 'report.md'}", flush=True)


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=["tabpfn", "tabicl"], choices=["tabpfn", "tabicl"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 25, 50])
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    RESULTS = args.output_dir
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        make_outputs()
        return
    for dataset in args.datasets:
        for seed in args.seeds:
            X, T, y, truth = get_split(dataset, seed=seed)
            if any(k < 1 or k > len(X) for k in args.ks):
                raise ValueError("k must be between 1 and n_train")
            distances = pairwise_distances(T, X)
            neighbors = {k: knn_indices(X, T, k) for k in args.ks}
            knn_pred = {k: KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(T) for k in args.ks}
            for name in args.models:
                target = RESULTS / f"{dataset}_seed{seed}_{name}"
                if (target / "metadata.json").exists():
                    saved = json.loads((target / "metadata.json").read_text())
                    if saved["ks"] != args.ks:
                        raise ValueError(f"Existing run uses different k values: {target}")
                    counts = set(saved["ensemble_members_per_layer"].values())
                    if counts != {args.n_estimators}:
                        raise ValueError("Existing attention uses a different estimator count. "
                                         "Use --output-dir for a separate experiment.")
                    print(f"Already completed: {target.name}", flush=True)
                    continue
                print(f"Extracting {dataset} seed={seed} {name}", flush=True)
                model = make_model(name, seed=seed, n_estimators=args.n_estimators)
                scores, probs, metadata = extract_attention(model, X, y, T, verify_predictions=True)
                pred = model.classes_[probs.argmax(axis=1)]
                rows = []
                for layer, weights in scores.items():
                    correlation = rank_correlation(weights, distances)
                    for k in args.ks:
                        attn_idx = top_k_indices(weights, k)
                        nn_idx = neighbors[k]
                        overlap = overlap_at_k(nn_idx, attn_idx)
                        mass = np.take_along_axis(weights, nn_idx, axis=1).sum(axis=1)
                        attn_purity = (y[attn_idx] == truth[:, None]).mean(axis=1)
                        knn_purity = (y[nn_idx] == truth[:, None]).mean(axis=1)
                        for i in range(len(T)):
                            rows.append(dict(dataset=dataset, seed=seed, model=name, layer=layer, k=k,
                                             test_index=i, overlap=overlap[i], random_overlap=k / len(X),
                                             knn_attention_mass=mass[i], attention_label_purity=attn_purity[i],
                                             knn_label_purity=knn_purity[i], distance_rank_correlation=correlation[i],
                                             model_correct=int(pred[i] == truth[i]),
                                             knn_correct=int(knn_pred[k][i] == truth[i])))
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / "weights.npz", **{f"layer_{i}": w for i, w in scores.items()})
                pd.DataFrame(rows).to_csv(target / "queries.csv", index=False)
                metadata.update(dataset=dataset, seed=seed, model=name, ks=args.ks,
                                n_train=len(X), n_test=len(T), accuracy=float((pred == truth).mean()),
                                aggregation="mean heads and ensemble members per ICL layer",
                                versions={p: version(p) for p in ["tabpfn", "tabicl", "torch", "numpy", "scikit-learn"]})
                (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"Completed {target.name}: accuracy={metadata['accuracy']:.4f}; {len(scores)} layers", flush=True)
    make_outputs()


if __name__ == "__main__":
    main()
