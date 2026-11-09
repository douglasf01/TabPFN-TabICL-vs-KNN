"""Validate and summarize the final eight-seed, single-estimator experiments."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from tabknn.data import DATASETS, get_split
from tabknn.evaluate import classification_metrics

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "final_8seeds"
SEEDS = list(range(8))
KS = [5, 10, 25, 50]
MODELS = ["tabpfn", "tabicl"]
COLORS = {"tabpfn": "#1665a7", "tabicl": "#d66b21"}
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_minority", "mcc"]


def knn_grid():
    RESULTS.mkdir(parents=True, exist_ok=True)
    for suffix in [".csv", ".json", "_summary.csv"]:
        shutil.copyfile(ROOT / "results" / f"baseline_8seeds_single_estimator{suffix}",
                        RESULTS / f"baseline{suffix}")
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            X, T, y, truth = get_split(dataset, seed=seed)
            for k in KS:
                knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
                rows.append(dict(dataset=dataset, seed=seed, model="knn", k=k, n_estimators=1,
                                 **classification_metrics(truth, knn.predict(T), knn.predict_proba(T))))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "knn_grid.csv", index=False)
    frame.groupby(["dataset", "k"]).accuracy.agg(["mean", "std", "count"]).to_csv(RESULTS / "knn_grid_summary.csv")
    print("Saved baseline and 160 KNN neighborhood runs", flush=True)


def class_balance():
    """Class counts and the scores of always predicting the majority class (seed-0 test split)."""
    rows = []
    for dataset in DATASETS:
        _, _, y, truth = get_split(dataset, seed=0)
        counts = np.bincount(np.concatenate([y, truth]))
        majority = classification_metrics(truth, np.full_like(truth, counts.argmax()))
        rows.append(dict(dataset=dataset, n_classes=len(counts), class_counts=" / ".join(map(str, counts)),
                         majority_share=counts.max() / counts.sum(),
                         majority_balanced_accuracy=majority["balanced_accuracy"],
                         majority_f1_macro=majority["f1_macro"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "class_balance.csv", index=False)
    return frame


def validate():
    baseline = pd.read_csv(RESULTS / "baseline.csv")
    restricted = pd.read_csv(RESULTS / "restricted_context.csv")
    knn = pd.read_csv(RESULTS / "knn_grid.csv")
    assert len(baseline) == 120 and len(restricted) == 320 and len(knn) == 160
    for frame, keys in [(baseline, ["dataset", "model"]),
                        (restricted, ["dataset", "model", "k"]), (knn, ["dataset", "k"])]:
        assert not frame.duplicated(keys + ["seed"]).any()
        assert set(frame.seed) == set(SEEDS)
        assert set(frame.dataset) == set(DATASETS)
        assert frame.groupby(keys).seed.nunique().eq(8).all()
        assert frame.n_estimators.eq(1).all()
        assert frame.accuracy.between(0, 1).all()
    accuracy = baseline.set_index(["dataset", "seed", "model"]).accuracy
    attention = RESULTS / "attention"
    summary = pd.read_csv(attention / "summary.csv")
    assert len(summary) == 5760
    checkpoints = 0
    undefined_correlations = 0
    restricted_metrics = []
    for dataset in DATASETS:
        for seed in SEEDS:
            X, T, _, truth = get_split(dataset, seed=seed)
            for model in MODELS:
                case = attention / f"{dataset}_seed{seed}_{model}"
                metadata = json.loads((case / "metadata.json").read_text())
                assert set(metadata["ensemble_members_per_layer"].values()) == {1}
                assert metadata["prediction_max_abs_delta"] <= 1e-6
                assert abs(metadata["accuracy"] - accuracy.loc[(dataset, seed, model)]) < 1e-12
                query = pd.read_csv(case / "queries.csv", usecols=["layer", "k", "knn_correct", "distance_rank_correlation"])
                undefined_correlations += int(query[query.k == 5].distance_rank_correlation.isna().sum())
                for k in KS:
                    checkpoint = RESULTS / "restricted_checkpoints" / f"{dataset}_seed{seed}_{model}_k{k}.npz"
                    with np.load(checkpoint) as saved:
                        assert int(saved["completed"]) == len(T)
                        np.testing.assert_array_equal(saved["signature"], [len(X), len(T), k, seed, 1])
                        measured = np.mean(saved["preds"] == truth)
                        restricted_metrics.append(dict(dataset=dataset, seed=seed, model=model, k=k,
                                                       **classification_metrics(truth, saved["preds"])))
                    row = restricted[(restricted.dataset == dataset) & (restricted.seed == seed)
                                     & (restricted.model == model) & (restricted.k == k)].iloc[0]
                    assert abs(row.accuracy - measured) < 1e-12
                    assert abs(row.frac_one_class + row.frac_constant_features + row.frac_model_used - 1) < 1e-12
                    knn_accuracy = knn[(knn.dataset == dataset) & (knn.seed == seed) & (knn.k == k)].accuracy.iloc[0]
                    assert abs(query[(query.layer == 0) & (query.k == k)].knn_correct.mean() - knn_accuracy) < 1e-12
                    checkpoints += 1
    original_knn = baseline[baseline.model == "knn"].set_index(["dataset", "seed"]).accuracy
    grid_knn = knn[knn.k == 5].set_index(["dataset", "seed"]).accuracy
    pd.testing.assert_series_equal(original_knn.sort_index(), grid_knn.sort_index())
    manifest = dict(status="complete", validated_utc=datetime.now(timezone.utc).isoformat(),
                    seeds=SEEDS, n_estimators=1, datasets=list(DATASETS), ks=KS,
                    baseline_runs=120, knn_grid_runs=160, attention_runs=80,
                    attention_layer_k_summaries=5760, restricted_context_runs=320,
                    completed_query_checkpoints=checkpoints,
                    undefined_query_layer_rank_correlations=undefined_correlations,
                    protocol="shared stratified 80/20 splits; train-only preprocessing; no test-set tuning",
                    limitations=["Overlapping splits are not independent replications",
                                 "Duplicates retained", "Single-class/constant-context fallbacks counted",
                                 "Attention measures association, not causal importance"])
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Imbalance-aware metrics from the saved predictions; the fallbacks give no probabilities.
    restricted = restricted.drop(columns=METRICS[1:], errors="ignore").merge(
        pd.DataFrame(restricted_metrics).drop(columns="accuracy"),
        on=["dataset", "seed", "model", "k"], validate="one_to_one")
    restricted.to_csv(RESULTS / "restricted_context.csv", index=False)
    return baseline, restricted, knn


def analyze(baseline, restricted, knn):
    baseline_models = baseline[baseline.model != "knn"][["dataset", "seed", "model", "accuracy"]]
    merged = restricted.merge(baseline_models.rename(columns={"accuracy": "full_accuracy"}),
                              on=["dataset", "seed", "model"], validate="many_to_one")
    merged["accuracy_change"] = merged.accuracy - merged.full_accuracy
    full_f1 = baseline.set_index(["dataset", "seed", "model"]).f1_macro
    merged["f1_macro_change"] = merged.f1_macro - full_f1.loc[list(zip(merged.dataset, merged.seed, merged.model))].values
    merged.to_csv(RESULTS / "restricted_vs_full.csv", index=False)
    stats = merged.groupby(["dataset", "model", "k"]).agg(
        restricted_mean=("accuracy", "mean"), restricted_std=("accuracy", "std"),
        full_mean=("full_accuracy", "mean"), change_mean=("accuracy_change", "mean"),
        change_std=("accuracy_change", "std"), f1_macro_change_mean=("f1_macro_change", "mean"),
        f1_macro_change_std=("f1_macro_change", "std"), model_used=("frac_model_used", "mean"), seeds=("seed", "nunique"))
    stats.to_csv(RESULTS / "restricted_vs_full_summary.csv")
    figures = RESULTS / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        for model in MODELS:
            part = stats.loc[(dataset, model)]
            ax.errorbar(part.index, part.restricted_mean, yerr=part.restricted_std,
                        marker="o", capsize=2, color=COLORS[model], label=model + " local")
            ax.axhline(part.full_mean.iloc[0], color=COLORS[model], linestyle="--", label=model + " full")
        part = knn[knn.dataset == dataset].groupby("k").accuracy.mean()
        ax.plot(part.index, part, color="#555555", marker="s", label="KNN")
        ax.set(title=dataset, xlabel="k", ylabel="Accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0.4, 1)
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Does nearby context preserve full-context accuracy?")
    fig.text(0.02, 0.015, "One estimator; eight shared seeds. Error bars: seed standard deviation. Single-class and constant contexts use the recorded fallbacks.", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(figures / f"restricted_context_accuracy.{extension}", dpi=190)
    plt.close(fig)
    for column, stem, label in [("change_mean", "restricted_context_change", "Accuracy change (percentage points)"),
                                ("f1_macro_change_mean", "restricted_context_f1_change", "Macro F1 change (points ×100)")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, model in zip(axes, MODELS):
            values = stats.reset_index().query("model == @model").pivot(index="dataset", columns="k", values=column).reindex(list(DATASETS)) * 100
            im = ax.imshow(values, cmap="RdBu", vmin=-20, vmax=20, aspect="auto")
            ax.set_xticks(range(4), KS)
            ax.set_yticks(range(5), list(DATASETS))
            ax.set(title=model, xlabel="k")
            for i in range(5):
                for j in range(4):
                    ax.text(j, i, f"{values.iloc[i,j]:+.1f}", ha="center", va="center", fontsize=9)
            fig.colorbar(im, ax=ax, label=label)
        fig.suptitle("Local context minus full context, paired by seed" + (" (macro F1)" if "f1" in column else ""))
        fig.text(0.02, 0.015, "Positive values favor local context. Means across eight seeds; one estimator. No test-set tuning.", fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 0.92))
        for extension in ["png", "svg", "pdf"]:
            fig.savefig(figures / f"{stem}.{extension}", dpi=190)
        plt.close(fig)
    lines = ["# Final experiments", "", "All experiments use seeds 0 through 7 and one estimator per foundation model.",
             "The same stratified 80/20 splits and training-only preprocessing are used throughout.", "",
             "| Experiment | Completed cases |", "|---|---:|", "| Baseline, three models | 120 |",
             "| KNN at four k values | 160 |", "| Attention, two models | 80 |", "| Restricted context, two models and four k values | 320 |", "",
             "## Baseline accuracy", "", "Means and standard deviations over eight seeds.", "",
             "| Dataset | KNN | TabPFN | TabICL |", "|---|---:|---:|---:|"]
    base = baseline.groupby(["dataset", "model"]).accuracy.agg(["mean", "std"])
    for dataset in DATASETS:
        values = [f"{base.loc[(dataset,m),'mean']:.1%} ± {base.loc[(dataset,m),'std']:.1%}" for m in ["knn", *MODELS]]
        lines.append("| " + dataset + " | " + " | ".join(values) + " |")
    lines += ["", "## Class balance and imbalance-aware metrics", "",
              "Every dataset is imbalanced, so accuracy alone overstates performance.",
              "Majority share is the accuracy of always predicting the most common class.",
              "Balanced accuracy is mean per-class recall (chance = 1/classes). Macro F1 averages per-class F1.",
              "Minority F1 is the F1 of the least frequent class. MCC is 0 for constant predictions.",
              "ROC AUC uses predicted probabilities (macro one-vs-rest for three classes).", "",
              "| Dataset | Class counts | Majority share | Majority macro F1 |", "|---|---|---:|---:|"]
    balance = class_balance()
    for row in balance.itertuples():
        lines.append(f"| {row.dataset} | {row.class_counts} | {row.majority_share:.1%} | {row.majority_f1_macro:.3f} |")
    lines += ["", "Means and standard deviations over eight seeds.", "",
              "| Dataset | Model | Accuracy | Balanced acc. | Macro F1 | Minority F1 | MCC | ROC AUC |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    full = baseline.groupby(["dataset", "model"])[METRICS + ["roc_auc"]].agg(["mean", "std"])
    for dataset in DATASETS:
        for model in ["knn", *MODELS]:
            values = [f"{full.loc[(dataset, model), (m, 'mean')]:.3f} ± {full.loc[(dataset, model), (m, 'std')]:.3f}"
                      for m in METRICS + ["roc_auc"]]
            lines.append(f"| {dataset} | {model} | " + " | ".join(values) + " |")
    lines += ["", "KNN at four k values, macro F1:", "", "| Dataset | k=5 | k=10 | k=25 | k=50 |", "|---|---:|---:|---:|---:|"]
    grid = knn.groupby(["dataset", "k"]).f1_macro.mean()
    for dataset in DATASETS:
        lines.append("| " + dataset + " | " + " | ".join(f"{grid.loc[(dataset, k)]:.3f}" for k in KS) + " |")
    lines += ["", "## Restricted context", "", "Mean changes from full context, in percentage points. These include fallback predictions.", "",
              "| Dataset | Model | k=5 | k=10 | k=25 | k=50 |", "|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for model in MODELS:
            values = [f"{100*stats.loc[(dataset,model,k),'change_mean']:+.1f}" for k in KS]
            lines.append("| " + dataset + " | " + model + " | " + " | ".join(values) + " |")
    lines += ["", "Mean changes in macro F1 from full context, in points (×100). ROC AUC is omitted because fallbacks give no probabilities.", "",
              "| Dataset | Model | k=5 | k=10 | k=25 | k=50 |", "|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for model in MODELS:
            values = [f"{100*stats.loc[(dataset,model,k),'f1_macro_change_mean']:+.1f}" for k in KS]
            lines.append("| " + dataset + " | " + model + " | " + " | ".join(values) + " |")
    lines += ["", "## Reading the results", "",
              "Compare overlap and attention mass separately. Similar rankings do not mean only nearby rows matter.",
              "Context changes are paired by seed. They test the whole restricted-context pipeline, including refitted preprocessing.",
              "Shared preprocessing and neighbor search use the full training set. Only the model input context is restricted.",
              "Single-class contexts use their class directly. Identical-feature contexts use a majority vote. Fallback rates are reported separately.",
              "KNN k values are descriptive comparisons, not selected using test accuracy. The baseline remains k=5.",
              "Seed splits overlap. Standard deviations are not confidence intervals. Duplicate rows remain in the data.", "",
              "See attention/report.md for attention findings, figures/ for context plots, and manifest.json for completion checks."]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
    print("Validated and finalized all experiments", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knn-only", action="store_true")
    args = parser.parse_args()
    if not (RESULTS / "knn_grid.csv").exists():
        knn_grid()
    if not args.knn_only:
        analyze(*validate())


if __name__ == "__main__":
    main()
