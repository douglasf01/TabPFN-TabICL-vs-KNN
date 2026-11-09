"""Create explanatory figures from saved attention runs; no model inference.

Run: uv run python scripts/plot_attention.py
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from tabknn.data import DATASETS, get_split

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "final_8seeds" / "attention"
OUTPUT = SOURCE / "figures"
MODELS = ["tabpfn", "tabicl"]
NAMES = {"tabpfn": "TabPFN", "tabicl": "TabICL"}
COLORS = {"tabpfn": "#1665a7", "tabicl": "#d66b21"}
DATASET_NAMES = {"blood-transfusion": "Blood Transfusion", "diabetes": "Diabetes",
                 "maternal-health-risk": "Maternal Health Risk", "credit-g": "German Credit",
                 "qsar-biodeg": "QSAR Biodegradation"}
DATASET_ORDER = list(DATASETS)


def footer(fig, text):
    fig.text(0.02, 0.018, text, fontsize=9, color="#555555", va="bottom")


def save(fig, stem, pdf):
    fig.savefig(OUTPUT / f"{stem}.png", dpi=190, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.svg", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    print(f"Created {stem}", flush=True)


def overview(last, pdf):
    specs = [("overlap", "Top-10 overlap with KNN", "Blues", 0, 0.5),
             ("knn_attention_mass", "Attention mass on 10 nearest rows", "Oranges", 0, 0.12),
             ("distance_rank_correlation", "Attention vs. proximity (Spearman ρ)", "PuOr", -1, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, (metric, title, cmap, lo, hi) in zip(axes, specs):
        data = last[last.k == 10].groupby(["dataset", "model"])[metric].mean().unstack()
        values = data.reindex(index=DATASET_ORDER, columns=MODELS).to_numpy()
        im = ax.imshow(values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        ax.set_xticks(range(2), [NAMES[m] for m in MODELS])
        ax.set_yticks(range(5), [DATASET_NAMES[d] for d in DATASET_ORDER] if ax is axes[0] else [])
        ax.set_title(title, fontsize=11, pad=14)
        for row in range(5):
            for col in range(2):
                value = values[row, col]
                label = f"{value:.1%}" if metric != "distance_rank_correlation" else f"{value:.2f}"
                ax.text(col, row, label, ha="center", va="center", fontsize=12,
                        color="white" if metric != "distance_rank_correlation" and value > (lo + hi) * 0.6 else "#222222")
        fig.colorbar(im, ax=ax, shrink=0.8, format=PercentFormatter(1) if metric != "distance_rank_correlation" else None)
    fig.suptitle("Attention favors nearby rows, but most attention goes elsewhere", fontsize=15, y=0.98)
    footer(fig, "Final ICL layer; one estimator; means over seeds 0–7. Random top-10 overlap is 1.2–1.7%. Each panel has its own scale.")
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    save(fig, "01_locality_overview", pdf)


def concentration(pdf):
    fractions = np.linspace(0, 1, 101)
    records = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, dataset in zip(axes.flat, DATASET_ORDER):
        for model in MODELS:
            curves, ranked_curves = [], []
            for target in sorted(SOURCE.glob(f"{dataset}_seed*_{model}")):
                seed = int(target.name.split("_seed")[1].split("_")[0])
                X, T, _, _ = get_split(dataset, seed=seed)
                order = np.argsort(pairwise_distances(T, X), axis=1, kind="stable")
                with np.load(target / "weights.npz") as archive:
                    layer = max(int(key.split("_")[1]) for key in archive.files)
                    weights = archive[f"layer_{layer}"]
                assert weights.shape == order.shape
                np.testing.assert_allclose(weights.sum(axis=1), 1, atol=2e-5)
                nearest_mass = np.take_along_axis(weights, order, axis=1).cumsum(axis=1).mean(axis=0)
                highest_mass = np.sort(weights, axis=1)[:, ::-1].cumsum(axis=1).mean(axis=0)
                x = np.arange(len(X) + 1) / len(X)
                curve = np.interp(fractions, x, np.r_[0, nearest_mass])
                ranked = np.interp(fractions, x, np.r_[0, highest_mass])
                curves.append(curve)
                ranked_curves.append(ranked)
                for fraction, mass, upper in zip(fractions, curve, ranked):
                    records.append(dict(dataset=dataset, model=model, seed=seed,
                                        fraction_train_rows=fraction, nearest_attention_mass=mass,
                                        highest_attention_mass=upper))
            curves, ranked_curves = np.array(curves), np.array(ranked_curves)
            mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
            ax.plot(fractions, mean, color=COLORS[model], label=f"{NAMES[model]}: nearest first", linewidth=2)
            ax.fill_between(fractions, np.clip(mean-std, 0, 1), np.clip(mean+std, 0, 1), color=COLORS[model], alpha=0.12)
            ax.plot(fractions, ranked_curves.mean(axis=0), color=COLORS[model], linestyle=":", linewidth=1.5,
                    label=f"{NAMES[model]}: highest attention first")
        ax.plot([0, 1], [0, 1], "--", color="#777777", label="Uniform attention")
        ax.set(title=DATASET_NAMES[dataset], xlim=(0, 1), ylim=(0, 1))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.18)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 1), frameon=False, fontsize=9)
    axes.flat[-1].text(0.04, 0.06, "Above the diagonal: more attention\nthan uniform allocation.\n\nDotted curves show concentration when\nrows are ordered by attention itself.", transform=axes.flat[-1].transAxes, fontsize=10)
    fig.suptitle("How much attention is captured as more training rows are included?", fontsize=15)
    fig.supxlabel("Fraction of training rows included", y=0.07)
    fig.supylabel("Cumulative attention mass", x=0.015)
    footer(fig, "Final ICL layer; average over queries within each seed, then across seeds. Shading: ±1 seed standard deviation.")
    fig.tight_layout(rect=(0.03, 0.10, 1, 0.94))
    pd.DataFrame(records).to_csv(OUTPUT / "concentration_curves.csv", index=False)
    save(fig, "02_attention_concentration", pdf)


def layers(summary, pdf):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.7))
    for ax, model in zip(axes, MODELS):
        data = summary[(summary.model == model) & (summary.k == 10)].groupby(["dataset", "layer"]).overlap.mean().unstack().reindex(DATASET_ORDER)
        im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=0.75)
        ax.set_yticks(range(5), [DATASET_NAMES[d] for d in DATASET_ORDER])
        ax.set_xticks(range(len(data.columns)), data.columns)
        ax.set(title=f"{NAMES[model]} · {len(data.columns)} ICL layers", xlabel="Layer index (zero based)")
        fig.colorbar(im, ax=ax, format=PercentFormatter(1), label="Top-10 overlap")
    fig.suptitle("Locality changes across layers and differs between architectures", fontsize=15)
    footer(fig, "Means over seeds 0–7; one estimator; heads averaged before ranking. Shared color scale; layer widths differ.")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save(fig, "03_layer_heatmaps", pdf)


def distributions(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    descriptive = []
    for ax, model in zip(axes, MODELS):
        values = []
        for dataset in DATASET_ORDER:
            samples = []
            for path in sorted(SOURCE.glob(f"{dataset}_seed*_{model}/queries.csv")):
                data = pd.read_csv(path, usecols=["layer", "k", "overlap"])
                samples.extend(data[(data.layer == data.layer.max()) & (data.k == 10)].overlap.to_numpy())
            values.append(samples)
            descriptive.append(dict(dataset=dataset, model=model, n=len(samples),
                                    zero_overlap_fraction=float(np.mean(np.array(samples) == 0)),
                                    median=float(np.median(samples))))
        artists = ax.violinplot(values, showmedians=True, showextrema=False, widths=0.8)
        for body in artists["bodies"]:
            body.set_facecolor(COLORS[model]); body.set_alpha(0.35)
        artists["cmedians"].set_color("#222222")
        ax.scatter(range(1, 6), [np.mean(v) for v in values], color=COLORS[model], marker="D", s=28, label="Mean")
        ax.set_xticks(range(1, 6), [DATASET_NAMES[d].replace(" ", "\n") for d in DATASET_ORDER], fontsize=9)
        ax.set(title=NAMES[model], ylim=(-0.03, 1.03), ylabel="Top-10 overlap")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper right")
    fig.suptitle("Average overlap hides large differences between test points", fontsize=15)
    footer(fig, "Final layer, k=10. Violin density pools queries across overlapping seed splits; black line: median; diamond: mean. Not uncertainty intervals.")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    pd.DataFrame(descriptive).to_csv(OUTPUT / "query_distribution_summary.csv", index=False)
    save(fig, "04_query_overlap_distributions", pdf)


def disagreements(pdf):
    data = pd.read_csv(SOURCE / "divergence.csv")
    data = data[data.k == 5]
    outcomes = ["both_correct", "model_only_correct", "knn_only_correct", "both_wrong"]
    labels = ["Both correct", "Foundation model only correct", "KNN only correct", "Both wrong"]
    colors = ["#ccd7df", "#218866", "#bd5a33", "#555866"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        counts = data[data.model == model].pivot(index="dataset", columns="outcome", values="n_queries").reindex(index=DATASET_ORDER, columns=outcomes).fillna(0)
        fractions = counts.div(counts.sum(axis=1), axis=0)
        left = np.zeros(5)
        for outcome, label, color in zip(outcomes, labels, colors):
            vals = fractions[outcome].to_numpy()
            ax.barh(range(5), vals, left=left, color=color, label=label, height=0.65)
            for i, value in enumerate(vals):
                if value > 0.035:
                    ax.text(left[i] + value/2, i, f"{value:.0%}", ha="center", va="center", fontsize=9,
                            color="#222222" if outcome == "both_correct" else "white")
            left += vals
        ax.set(title=f"{NAMES[model]} versus KNN", xlim=(0, 1), xlabel="Share of test predictions")
        ax.set_yticks(range(5), [DATASET_NAMES[d] for d in DATASET_ORDER])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].invert_yaxis()
    fig.suptitle("Foundation models correct more KNN errors than they introduce", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=2, frameon=False)
    footer(fig, "KNN k=5, matching the baseline. Counts pool seeds 0–7; overlapping splits repeat observations. This does not establish a causal role for attention.")
    fig.tight_layout(rect=(0, 0.19, 1, 0.92))
    save(fig, "05_prediction_disagreements", pdf)


def ties(pdf):
    data = pd.read_csv(SOURCE / "tie_diagnostics.csv")
    data = data[(data.model == "tabpfn") & (data.k == 10)].copy()
    data["duplicate_train_fraction"] = data.duplicate_train_rows / data.n_train
    specs = [("duplicate_train_fraction", "Duplicate training rows", "#647c92"),
             ("fraction_test_exact_train_match", "Test rows matching training exactly", "#b77838"),
             ("fraction_knn_boundary_tie", "Ambiguous top-10 distance boundary", "#9b566f")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for ax, (metric, title, color) in zip(axes, specs):
        stats = data.groupby("dataset")[metric].agg(["mean", "std"]).reindex(DATASET_ORDER)
        ax.barh(range(5), stats["mean"], xerr=stats["std"], color=color, capsize=3, height=0.6)
        for i, value in enumerate(stats["mean"]):
            ax.text(value + stats["std"].iloc[i] + 0.025, i, f"{value:.1%}", va="center", fontsize=10)
        ax.set(title=title, xlim=(0, 0.95), xlabel="Fraction")
        ax.set_yticks(range(5), [DATASET_NAMES[d] for d in DATASET_ORDER])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle("Duplicate rows and ties complicate row-index overlap", fontsize=15)
    footer(fig, "Means ±1 seed standard deviation. Duplicate rows count copies beyond the first. Boundary ties compare the 10th and 11th distances (rtol 1e−6, atol 1e−7).")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    save(fig, "06_duplicates_and_ties", pdf)


def main():
    global SOURCE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    SOURCE = args.source
    OUTPUT = SOURCE / "figures"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    summary = pd.read_csv(SOURCE / "summary.csv")
    assert summary[["dataset", "seed", "model"]].drop_duplicates().shape[0] == 80
    last = summary[summary.layer == summary.groupby("model").layer.transform("max")]
    with PdfPages(OUTPUT / "attention_visualizations.pdf") as pdf:
        overview(last, pdf)
        concentration(pdf)
        layers(summary, pdf)
        distributions(pdf)
        disagreements(pdf)
        ties(pdf)
    captions = [
        ("01_locality_overview", "Locality overview", "Compare top-10 neighbor overlap, attention mass on those neighbors, and attention–distance rank correlation. These are distinct measurements: matching rankings does not imply concentrating all attention on nearby rows."),
        ("02_attention_concentration", "Where attention goes", "Solid curves add rows from nearest to farthest. Dotted curves add rows from highest to lowest attention. The diagonal represents uniform attention. The gap between solid and dotted curves shows how much attention concentrates on rows outside the nearest-neighbor ordering."),
        ("03_layer_heatmaps", "How locality changes across layers", "Colors use the same scale for both models. TabPFN begins with low overlap and has peaks and troughs through later layers; TabICL starts with stronger locality and ends with lower overlap. Layer indices refer to different architectures and are not directly equivalent."),
        ("04_query_overlap_distributions", "Variation between test points", "Each violin shows the distribution of final-layer top-10 overlap. Black horizontal lines mark medians; diamonds mark means. The density pools eight overlapping seed splits and is not a confidence interval."),
        ("05_prediction_disagreements", "Which predictions differ from KNN", "Green marks cases where the foundation model is correct and KNN is wrong; orange marks the reverse. KNN uses k=5, matching the baseline. These descriptive comparisons do not show that attention caused an improvement."),
        ("06_duplicates_and_ties", "Why exact row-index comparisons need care", "Duplicate training rows, exact train/test feature matches, and ties around the kth distance are especially prevalent in Blood Transfusion and Maternal Health Risk. Equivalent rows can receive different indices, making overlap sensitive to tie ordering.")]
    lines = ["# Attention analysis: visual guide", "",
             "80 completed runs: five datasets × eight seeds × two models, with one estimator per run.",
             "Attention is averaged over heads. All figures use the final ICL layer",
             "except the layer heatmaps. Attention describes association, not causal importance.", "",
             "[Download all six figures as a PDF](attention_visualizations.pdf). Each figure also has an editable SVG.", ""]
    for stem, title, caption in captions:
        lines += [f"## {title}", "", caption, "", f"![{title}]({stem}.png)", "", f"[SVG]({stem}.svg)", ""]
    lines += ["## Reproduce", "", "From the project root:", "", "```bash",
              "uv run python scripts/plot_attention.py", "```", "",
              "Cumulative-mass source data: concentration_curves.csv. Per-query distribution summaries: query_distribution_summary.csv.",
              "Other source tables are in the parent directory. Error bands/bars indicate seed standard deviations, not confidence intervals."]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Visual guide: {OUTPUT / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
