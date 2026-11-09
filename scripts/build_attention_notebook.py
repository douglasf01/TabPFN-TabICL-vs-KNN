"""Build and execute a self-contained notebook using the saved experiment data.

Uses standard Python to store Jupyter outputs, so building does not require a
Jupyter server. The notebook itself also runs in a normal Python Jupyter kernel.
"""

import ast
from contextlib import redirect_stdout, redirect_stderr
import html
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "notebooks" / "attention_analysis.ipynb"


def definitions(relative, names=None):
    """Embed function definitions, without importing project-only modules."""
    source = (ROOT / relative).read_text()
    tree = ast.parse(source)
    return "\n\n".join("\n".join(source.splitlines()[min([node.lineno] + [d.lineno for d in node.decorator_list])-1:node.end_lineno])
                        for node in tree.body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                        and (names is None or node.name in names))


def main():
    cells = []

    def markdown(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text.strip() + "\n"))

    def code(text):
        cells.append(dict(cell_type="code", metadata={}, source=text.strip() + "\n",
                          execution_count=None, outputs=[]))

    markdown("""
# Attention and nearest neighbors

We compare KNN, TabPFN, and TabICL on five datasets and eight seeds.
Each foundation model uses one estimator. KNN uses one classifier.
We ask whether high attention goes to nearby training rows.

The notebook includes the code and saved outputs. Run cells from top to bottom.
Saved runs are reused. Missing runs are computed when RUN_MISSING is True.
The full run needs model weights and may need internet access and a TabPFN token.
Set the token in a local .env file. Do not put it in this notebook.

Python packages: numpy, pandas, scipy, scikit-learn, openml, matplotlib, torch,
tabpfn, and tabicl. Use Jupyter and nbconvert to open and export this file.
The tested package versions are shown below. The attention adapter uses model internals.

Install missing packages in the notebook environment with:

    python -m pip install numpy pandas scipy scikit-learn openml matplotlib torch tabpfn==9.1.0 tabicl==2.2.0 jupyter nbconvert
""")
    code('''
from pathlib import Path
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
from importlib.metadata import version
import base64
import importlib
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (pairwise_distances, accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    if "display" not in globals():
        from IPython.display import display
except ImportError:
    # The build script supplies display when Jupyter is not installed.
    if "display" not in globals():
        def display(value, **kwargs):
            print(value)

# Find the project root. A standalone copy uses the current directory.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "pyproject.toml").exists()), Path.cwd())
DATA_DIR = ROOT / "data"
FINAL_RESULTS = ROOT / "results" / "final_8seeds"
RESULTS = FINAL_RESULTS / "attention"
SOURCE = RESULTS
OUTPUT = RESULTS / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(8))
N_ESTIMATORS = 1
KS = [5, 10, 25, 50]
MODELS = ["tabpfn", "tabicl"]
RUN_MISSING = True
RERUN_BASELINE = False
RERUN_ATTENTION = False
RERUN_RESTRICTED = False

DATASETS = {"blood-transfusion": 1464, "diabetes": 37,
            "maternal-health-risk": 46941, "credit-g": 31, "qsar-biodeg": 1494}
DATASET_ORDER = list(DATASETS)
DATASET_NAMES = {"blood-transfusion": "Blood Transfusion", "diabetes": "Diabetes",
                 "maternal-health-risk": "Maternal Health Risk", "credit-g": "German Credit",
                 "qsar-biodeg": "QSAR Biodegradation"}
NAMES = {"tabpfn": "TabPFN", "tabicl": "TabICL"}
COLORS = {"tabpfn": "#1665a7", "tabicl": "#d66b21"}
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_minority", "mcc", "roc_auc"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})

def display_png(path):
    # Store the picture in the notebook for export.
    display({"image/png": base64.b64encode(Path(path).read_bytes()).decode()}, raw=True)

display(pd.DataFrame({"package": ["tabpfn", "tabicl", "torch", "numpy", "pandas", "scikit-learn"],
                      "version": [version(p) for p in ["tabpfn", "tabicl", "torch", "numpy", "pandas", "scikit-learn"]]}))
''')
    markdown("""
## Data and model setup

We use an 80/20 split with the same class balance and seed for each model.
Missing values, scaling, and category encoding are fitted on training data only.
KNN uses Euclidean distance. The foundation models also apply their own preprocessing.

Every dataset is class-imbalanced, so we report more than accuracy.
Balanced accuracy is mean per-class recall. Macro F1 weights every class equally.
Minority F1 is the F1 of the least frequent class. MCC is 0 for constant predictions.
ROC AUC uses class probabilities, with a macro one-vs-rest average for three classes.
""")
    env_source = definitions("tabknn/__init__.py", {"_load_env"}).replace(
        'Path(__file__).resolve().parent.parent / ".env"', 'ROOT / ".env"')
    code(env_source + "\n\n_load_env()\n\n" + definitions("tabknn/data.py") + "\n\n" + definitions("tabknn/models.py") + "\n\n" + definitions("tabknn/neighbors.py")
         + "\n\n" + definitions("tabknn/evaluate.py", {"classification_metrics"}))
    markdown("""
## Read attention

We read attention from test rows to training rows in every ICL layer.
We average heads and ensemble members before choosing the top k rows.
Feature attention is excluded. We check that reading attention does not change predictions.
This code supports TabPFN v3.5 and TabICLv2. Run it in one thread.
""")
    code(definitions("tabknn/attention.py"))
    markdown("""
## Run or load the baseline

KNN uses k=5. Saved baseline rows are reused unless RERUN_BASELINE is True.
""")
    code('''
baseline_path = FINAL_RESULTS / "baseline.csv"
baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
baseline_rows = [] if RERUN_BASELINE else baseline.to_dict("records")
known = {(r["dataset"], r["seed"], r["model"]) for r in baseline_rows}
for dataset in DATASET_ORDER:
    for seed in SEEDS:
        missing = [name for name in ["knn", *MODELS] if (dataset, seed, name) not in known]
        if not missing:
            continue
        if not RUN_MISSING and not RERUN_BASELINE:
            raise RuntimeError("Missing baseline runs. Set RUN_MISSING=True.")
        X, T, y, truth = get_split(dataset, seed=seed)
        for name in missing:
            model = make_model(name, seed=seed, n_estimators=N_ESTIMATORS)
            model.fit(X, y)
            scores = classification_metrics(truth, model.predict(T), model.predict_proba(T))
            baseline_rows.append(dict(dataset=dataset, seed=seed, model=name, **scores, n_estimators=1))
            print(dataset, seed, name, round(scores["accuracy"], 4), round(scores["f1_macro"], 4))
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
baseline = pd.DataFrame(baseline_rows)
assert set(METRICS) <= set(baseline.columns), "Old baseline file without F1 columns. Set RERUN_BASELINE=True."

# What always predicting the most common class scores, on the seed-0 test split.
balance_rows = []
for dataset in DATASET_ORDER:
    _, _, y, truth = get_split(dataset, seed=0)
    counts = np.bincount(np.concatenate([y, truth]))
    majority = classification_metrics(truth, np.full_like(truth, counts.argmax()))
    balance_rows.append(dict(dataset=dataset, class_counts=" / ".join(map(str, counts)),
                             majority_share=counts.max() / counts.sum(),
                             majority_balanced_accuracy=majority["balanced_accuracy"],
                             majority_f1_macro=majority["f1_macro"]))
balance = pd.DataFrame(balance_rows).set_index("dataset")
display(balance.round(3))
baseline_means = baseline.groupby(["dataset", "model"])[METRICS].mean().reindex(DATASET_ORDER, level=0)
display(baseline_means.round(3))
''')
    markdown("""
Both foundation models have higher mean accuracy than KNN on all five datasets.
The largest gap is on Maternal Health Risk. Duplicate rows make that dataset harder to assess.
""")
    markdown("""
## Run or load attention comparisons

Overlap is the share of rows shared by the attention top k and the nearest k.
Random selection has expected overlap k / number of training rows.
Attention mass is the total weight on the nearest k rows.
Positive rank correlation means nearby rows tend to receive more attention.
""")
    code(definitions("scripts/run_attention.py", {"rank_correlation"}))
    code('''
# Each completed case has weights, query metrics, and metadata.
for dataset in DATASET_ORDER:
    for seed in SEEDS:
        for name in MODELS:
            target = RESULTS / f"{dataset}_seed{seed}_{name}"
            complete = all((target / f).exists() for f in ["metadata.json", "weights.npz", "queries.csv"])
            if complete and not RERUN_ATTENTION:
                saved = json.loads((target / "metadata.json").read_text())
                if saved["ks"] != KS:
                    raise ValueError("Saved k values differ. Set RERUN_ATTENTION=True.")
                if set(saved["ensemble_members_per_layer"].values()) != {N_ESTIMATORS}:
                    raise ValueError("Saved attention uses a different estimator count.")
                continue
            if not RUN_MISSING and not RERUN_ATTENTION:
                raise RuntimeError(f"Missing {target.name}. Set RUN_MISSING=True.")
            print("Running", dataset, seed, name)
            X, T, y, truth = get_split(dataset, seed=seed)
            distances = pairwise_distances(T, X)
            neighbors = {k: knn_indices(X, T, k) for k in KS}
            knn_pred = {k: KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(T) for k in KS}
            model = make_model(name, seed=seed, n_estimators=N_ESTIMATORS)
            scores, probs, metadata = extract_attention(model, X, y, T, verify_predictions=True)
            pred = model.classes_[probs.argmax(axis=1)]
            rows = []
            for layer, weights in scores.items():
                correlation = rank_correlation(weights, distances)
                for k in KS:
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
            metadata.update(dataset=dataset, seed=seed, model=name, ks=KS,
                            n_train=len(X), n_test=len(T), accuracy=float((pred == truth).mean()),
                            aggregation="mean heads and ensemble members per ICL layer",
                            versions={p: version(p) for p in ["tabpfn", "tabicl", "torch", "numpy", "scikit-learn"]})
            (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\\n")

paths = [RESULTS / f"{d}_seed{s}_{m}" for d in DATASET_ORDER for s in SEEDS for m in MODELS]
queries = pd.concat([pd.read_csv(p / "queries.csv") for p in paths], ignore_index=True)
metrics = ["overlap", "random_overlap", "knn_attention_mass", "attention_label_purity",
           "knn_label_purity", "distance_rank_correlation", "model_correct", "knn_correct"]
summary = queries.groupby(["dataset", "seed", "model", "layer", "k"])[metrics].mean().reset_index()
summary["overlap_above_random"] = summary.overlap - summary.random_overlap
summary.to_csv(RESULTS / "summary.csv", index=False)
last = summary[summary.layer == summary.groupby("model").layer.transform("max")]
final_queries = queries[queries.layer == queries.groupby("model").layer.transform("max")].copy()
final_queries["outcome"] = np.select(
    [final_queries.model_correct.astype(bool) & ~final_queries.knn_correct.astype(bool),
     ~final_queries.model_correct.astype(bool) & final_queries.knn_correct.astype(bool),
     final_queries.model_correct.astype(bool) & final_queries.knn_correct.astype(bool)],
    ["model_only_correct", "knn_only_correct", "both_correct"], default="both_wrong")
divergence = final_queries.groupby(["dataset", "model", "k", "outcome"]).agg(
    n_queries=("overlap", "size"), overlap=("overlap", "mean"), knn_attention_mass=("knn_attention_mass", "mean"))
divergence.to_csv(RESULTS / "divergence.csv")

metadata = [json.loads((p / "metadata.json").read_text()) for p in paths]
checks = pd.DataFrame(metadata)[["dataset", "seed", "model", "accuracy", "prediction_max_abs_delta"]]
expected = baseline.set_index(["dataset", "seed", "model"]).accuracy
for row in checks.itertuples(index=False):
    assert abs(row.accuracy - expected.loc[(row.dataset, row.seed, row.model)]) < 1e-12
    assert row.prediction_max_abs_delta <= 1e-6
print(f"Checked {len(checks)} attention runs. Accuracies match the baseline.")
print("Largest prediction change:", checks.prediction_max_abs_delta.max())

# KNN comparisons use the same test points and k values as attention.
knn_rows = []
for dataset in DATASET_ORDER:
    for seed in SEEDS:
        X, T, y, truth = get_split(dataset, seed=seed)
        for k in KS:
            knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
            knn_rows.append(dict(dataset=dataset, seed=seed, model="knn", k=k, n_estimators=1,
                                 **classification_metrics(truth, knn.predict(T), knn.predict_proba(T))))
knn_grid = pd.DataFrame(knn_rows)
assert len(knn_grid) == 160
expected = final_queries.groupby(["dataset", "seed", "k"]).knn_correct.mean()
assert np.allclose(knn_grid.set_index(["dataset", "seed", "k"]).accuracy.loc[expected.index], expected)
knn_grid.to_csv(FINAL_RESULTS / "knn_grid.csv", index=False)
display(knn_grid.groupby(["dataset", "k"])[["accuracy", "f1_macro"]].mean().unstack().round(3))
''')
    markdown("""
## Duplicate rows and ties

Several rows can have the same features or distance. This can change index overlap.
We keep the baseline splits and measure how often this happens.
""")
    code('''
diagnostics = []
for dataset in DATASET_ORDER:
    for seed in SEEDS:
        X, T, _, _ = get_split(dataset, seed=seed)
        train_rows = {row.tobytes() for row in X}
        exact_match = np.mean([row.tobytes() in train_rows for row in T])
        distances = np.sort(pairwise_distances(T, X), axis=1)
        for name in MODELS:
            target = RESULTS / f"{dataset}_seed{seed}_{name}"
            with np.load(target / "weights.npz") as archive:
                layer = max(int(key.split("_")[1]) for key in archive.files)
                ranked = np.sort(archive[f"layer_{layer}"], axis=1)[:, ::-1]
            for k in KS:
                diagnostics.append(dict(dataset=dataset, seed=seed, model=name, k=k,
                    n_train=len(X), n_test=len(T), duplicate_train_rows=len(X)-len(train_rows),
                    fraction_test_exact_train_match=exact_match,
                    fraction_knn_boundary_tie=float(np.isclose(distances[:, k-1], distances[:, k], rtol=1e-6, atol=1e-7).mean()),
                    fraction_attention_boundary_tie=float((ranked[:, k-1] == ranked[:, k]).mean())))
pd.DataFrame(diagnostics).to_csv(RESULTS / "tie_diagnostics.csv", index=False)
''')
    markdown("""
## Plot functions

The following code creates every figure from the result tables and saved weights.
All pictures are saved as PNG and SVG. A PDF collects the figures below.
""")
    code(definitions("scripts/plot_attention.py", {"footer", "save", "overview", "concentration", "layers", "distributions", "disagreements", "ties"}))
    code('''
pdf = PdfPages(OUTPUT / "notebook_figures.pdf")
''')
    markdown("## Class imbalance")
    code('''
# Accuracy next to metrics that do not reward predicting the majority class.
panels = [("accuracy", "Accuracy", "majority_share"), ("balanced_accuracy", "Balanced accuracy", "majority_balanced_accuracy"),
          ("f1_macro", "Macro F1", "majority_f1_macro"), ("roc_auc", "ROC AUC", None)]
model_colors = {"knn": "#5b6770", **COLORS}
model_names = {"knn": "KNN (k=5)", **NAMES}
fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
for ax, (column, title, reference) in zip(axes, panels):
    for j, dataset in enumerate(DATASET_ORDER):
        ax.scatter(balance.loc[dataset, reference] if reference else 0.5, j, marker="|", s=160, color="#7b8790", zorder=4,
                   label="Majority class / chance" if j == 0 else None)
        for i, name in enumerate(["knn", *MODELS]):
            values = baseline[(baseline.dataset == dataset) & (baseline.model == name)][column]
            ax.errorbar(values.mean(), j + (i - 1) * 0.24, xerr=values.std(), fmt="s" if name == "knn" else "o",
                        color=model_colors[name], capsize=3, label=model_names[name] if j == 0 else None)
    ax.set(title=title, xlim=(0.15, 1.02))
    ax.set_xticks([0.25, 0.5, 0.75, 1])
    ax.grid(axis="x", alpha=0.2)
axes[0].set_yticks(range(5), [DATASET_NAMES[d] for d in DATASET_ORDER])
axes[0].invert_yaxis()
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Full-context performance against the majority-class predictor")
footer(fig, "Eight seeds; one estimator. Error bars: seed standard deviation. Gray ticks: always predicting the majority class (0.5 for ROC AUC).")
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
save(fig, "12_imbalance_metrics", pdf)
display_png(OUTPUT / "12_imbalance_metrics.png")
''')
    markdown("""
Accuracy hides the class imbalance. On Blood Transfusion, 76% of rows are in one class,
and both foundation models score only about two points above always predicting it.
There, KNN has higher balanced accuracy, macro F1 and minority F1 than either foundation model,
although the foundation models have the higher ROC AUC. They rank cases better, but their predicted labels favor the majority class.
On the other four datasets the foundation models lead on every metric.
""")
    sections = [
        ("Locality and attention mass", "overview(last, pdf)", "01_locality_overview", "TabPFN has more final-layer top-10 overlap than TabICL on every dataset.\nBoth exceed random overlap. The nearest ten rows receive only a small part of total attention."),
        ("Where attention goes", "concentration(pdf)", "02_attention_concentration", "Nearby rows receive more weight than uniform attention would give them.\nThe curves rise gradually, so attention also reaches distant rows. The dotted curves show that some highly weighted rows are not the nearest ones."),
        ("Layers", "layers(summary, pdf)", "03_layer_heatmaps", "TabPFN starts with low overlap and changes sharply between layers.\nTabICL starts with stronger overlap and ends lower. The pattern is not a steady rise or fall."),
        ("Differences between test points", "distributions(pdf)", "04_query_overlap_distributions", "Overlap varies a lot between test points. Some have no shared neighbors.\nA single mean hides this spread, especially for Maternal Health Risk."),
        ("Prediction differences", "disagreements(pdf)", "05_prediction_disagreements", "Both foundation models correct more KNN errors than they add.\nThis describes predictions. It does not show that attention causes the accuracy gain."),
        ("Duplicates and ties", "ties(pdf)", "06_duplicates_and_ties", "Blood Transfusion and Maternal Health Risk have many repeated rows and distance ties.\nMany Maternal Health Risk test rows exactly match a training row. Treat its accuracy and index overlap with care.")]
    for title, call, stem, note in sections:
        markdown("## " + title)
        code(call + f'\ndisplay_png(OUTPUT / "{stem}.png")')
        markdown(note)
        if title == "Layers":
            code('''
# Lines show the seed variation that the heatmap does not show.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, name in zip(axes, MODELS):
    part = summary[(summary.model == name) & (summary.k == 10)]
    for dataset, rows in part.groupby("dataset"):
        values = rows.groupby("layer").overlap.agg(["mean", "std"])
        mean, std = values["mean"], values["std"]
        ax.plot(values.index, mean, label=DATASET_NAMES[dataset])
        ax.fill_between(values.index, np.clip(mean-std, 0, 1), np.clip(mean+std, 0, 1), alpha=0.12)
    ax.set(title=NAMES[name], xlabel="ICL layer (zero based)", ylabel="Top-10 overlap", ylim=(0, 0.8))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8)
fig.suptitle("Layer changes across eight seeds")
footer(fig, "k=10; one estimator. Lines show seed means; shading shows one seed standard deviation, not confidence intervals.")
fig.tight_layout(rect=(0, 0.09, 1, 0.92))
save(fig, "08_notebook_layer_curves", pdf)
display_png(OUTPUT / "08_notebook_layer_curves.png")
''')
    markdown("""
## Choice of k

Larger k gives larger sets. Random overlap also rises, so compare against the dashed line.
Bars show variation across seeds, not confidence intervals.
""")
    code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, name in zip(axes, MODELS):
    part = last[last.model == name]
    for dataset, rows in part.groupby("dataset"):
        values = rows.groupby("k").overlap.agg(["mean", "std"])
        ax.errorbar(values.index, values["mean"], yerr=values["std"], marker="o", capsize=3,
                    label=DATASET_NAMES[dataset])
    random = part.groupby("k").random_overlap.mean()
    ax.plot(random.index, random, "k--", label="Random (dataset mean)")
    ax.set(title=NAMES[name], xlabel="k", ylabel="Overlap", ylim=(0, 0.8))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8)
fig.suptitle("Top-k overlap increases with neighborhood size")
footer(fig, "Final layer; one estimator. Means and standard deviations across seeds 0–7.")
fig.tight_layout(rect=(0, 0.09, 1, 0.92))
save(fig, "07_notebook_overlap_by_k", pdf)
display_png(OUTPUT / "07_notebook_overlap_by_k.png")
''')
    markdown("""
## Restricted context

Each test row gets only its nearest k training rows. Preprocessing is fitted again on that context.
Shared scaling and neighbor search still use the full training set. Only the model context is restricted.
Single-class contexts predict their class directly. Constant-feature contexts use a majority vote.
The fallback rates are reported, so these results are not pure model predictions for every row.
""")
    code(definitions("tabknn/evaluate.py", {"restricted_context_accuracy"}))
    code('''
restricted_path = FINAL_RESULTS / "restricted_context.csv"
restricted_rows = pd.read_csv(restricted_path).to_dict("records") if restricted_path.exists() else []
if RERUN_RESTRICTED:
    restricted_rows = []
known = {(r["dataset"], r["seed"], r["model"], r["k"]) for r in restricted_rows}
assert all(r["n_estimators"] == N_ESTIMATORS for r in restricted_rows)
with cached_model_weights():
    for dataset in DATASET_ORDER:
        for seed in SEEDS:
            missing = [(name,k) for name in MODELS for k in KS if (dataset,seed,name,k) not in known]
            if not missing:
                continue
            if not RUN_MISSING and not RERUN_RESTRICTED:
                raise RuntimeError("Missing restricted runs. Set RUN_MISSING=True.")
            X, T, y, truth = get_split(dataset, seed=seed)
            for name,k in missing:
                # Fixed weights are reused; data preprocessing is refitted for each context.
                result = restricted_context_accuracy(name,X,T,y,truth,k,seed=seed,n_estimators=N_ESTIMATORS)
                restricted_rows.append(dict(dataset=dataset,seed=seed,model=name,k=k,n_estimators=N_ESTIMATORS,**result))
                pd.DataFrame(restricted_rows).to_csv(restricted_path,index=False)
restricted = pd.DataFrame(restricted_rows)
assert len(restricted) == 320
assert "f1_macro" in restricted, "Old restricted file without F1 columns. Run scripts/finalize_experiments.py."
local_full = restricted.merge(baseline[baseline.model != "knn"][["dataset","seed","model","accuracy","f1_macro"]].rename(
    columns={"accuracy":"full_accuracy","f1_macro":"full_f1_macro"}), on=["dataset","seed","model"], validate="many_to_one")
local_full["change_pp"] = 100 * (local_full.accuracy-local_full.full_accuracy)
local_full["f1_change"] = 100 * (local_full.f1_macro-local_full.full_f1_macro)
display(local_full.groupby(["dataset","model","k"])[["accuracy","change_pp","f1_macro","f1_change","frac_model_used"]].mean().round(3))
''')
    code('''
# Compare local accuracy with the matching full-context prediction.
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
for ax,dataset in zip(axes.flat,DATASET_ORDER):
    for name in MODELS:
        part=local_full[(local_full.dataset==dataset)&(local_full.model==name)]
        values=part.groupby("k").accuracy.agg(["mean","std"])
        ax.errorbar(values.index,values["mean"],yerr=values["std"],marker="o",capsize=2,color=COLORS[name],label=NAMES[name]+" local")
        ax.axhline(part.full_accuracy.mean(),color=COLORS[name],linestyle="--",label=NAMES[name]+" full")
    ax.set(title=DATASET_NAMES[dataset],xlabel="k",ylabel="Accuracy",ylim=(0.4,1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
axes.flat[-1].axis("off")
handles,labels=axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(handles,labels,loc="center",frameon=False)
fig.suptitle("Restricted context compared with full context")
footer(fig,"Eight seeds; one estimator. Error bars: seed standard deviation. Fallback predictions are included.")
fig.tight_layout(rect=(0,0.07,1,0.94))
save(fig,"09_restricted_context",pdf)
display_png(OUTPUT/"09_restricted_context.png")

for column,stem,label,title in [("change_pp","11_restricted_accuracy_change","Accuracy change (percentage points)","Restricted context minus full context"),
                                ("f1_change","13_restricted_f1_change","Macro F1 change (points ×100)","Restricted context minus full context, macro F1")]:
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,name in zip(axes,MODELS):
        changes=local_full[local_full.model==name].groupby(["dataset","k"])[column].mean().unstack().reindex(DATASET_ORDER)
        im=ax.imshow(changes,cmap="RdBu",vmin=-20,vmax=20,aspect="auto")
        ax.set_xticks(range(4),KS)
        ax.set_yticks(range(5),[DATASET_NAMES[d] for d in DATASET_ORDER])
        ax.set(title=NAMES[name],xlabel="k")
        for i in range(5):
            for j in range(4):
                ax.text(j,i,f"{changes.iloc[i,j]:+.1f}",ha="center",va="center",fontsize=9)
        fig.colorbar(im,ax=ax,label=label)
    fig.suptitle(title)
    footer(fig,"Paired by seed, then averaged across eight seeds. Positive values favor restricted context. Fallbacks are included.")
    fig.tight_layout(rect=(0,0.07,1,0.92))
    save(fig,stem,pdf)
    display_png(OUTPUT/f"{stem}.png")

fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
for ax,(column,label) in zip(axes,[("accuracy","Accuracy"),("f1_macro","Macro F1")]):
    for dataset,part in knn_grid.groupby("dataset"):
        values=part.groupby("k")[column].agg(["mean","std"])
        ax.errorbar(values.index,values["mean"],yerr=values["std"],marker="o",capsize=3,label=DATASET_NAMES[dataset])
    ax.set(xlabel="k",ylabel=label,title=label,ylim=(0.4,1))
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=8)
fig.suptitle("KNN for the four neighborhood sizes")
footer(fig,"Eight shared splits. Error bars: seed standard deviation. This is a descriptive comparison, not test-set tuning.")
fig.tight_layout(rect=(0,0.09,1,0.94))
save(fig,"10_knn_neighborhood_accuracy",pdf)
display_png(OUTPUT/"10_knn_neighborhood_accuracy.png")
pdf.close()
''')
    markdown("""
The dashed lines use the same full-context splits. A gap below a dashed line means local context lowers accuracy.
This tests the full context-selection pipeline, including its preprocessing and fallback rules.
Full context has higher mean accuracy in all tested dataset, model, and k combinations.
Larger contexts often reduce the gap, but the change is not always steady.

Macro F1 mostly tells the same story, with one exception. On Blood Transfusion, local context
lowers TabPFN's accuracy but slightly raises its mean macro F1, within one seed standard deviation.
On German Credit, TabPFN's macro F1 loss is larger than its accuracy loss.
""")
    markdown("""
## What we learned

Attention and Euclidean neighbors are related, but they do not select the same rows.
Most attention lies outside the nearest ten rows. The pattern depends on layer and dataset.
Attention alone cannot explain why a prediction is correct.

The seeds reuse some observations. Error bars are not evidence of statistical significance.
Duplicate rows and head averaging also limit the comparison.
Every dataset is imbalanced. Accuracy alone flatters the foundation models on Blood Transfusion.
All baseline, attention, and restricted-context cases use eight seeds and one estimator.
KNN is also compared at k=5, 10, 25, and 50. These values were not selected using test accuracy.

## Export

This notebook has saved tables and figures. Export it to HTML with Jupyter, or run:

    jupyter nbconvert --to html notebooks/attention_analysis.ipynb

For PDF, print the HTML to PDF in a browser. Hide code during HTML export with:

    jupyter nbconvert --to html --no-input notebooks/attention_analysis.ipynb
""")
    for i, cell in enumerate(cells):
        cell["id"] = f"attention-{i:02d}"
    notebook = dict(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"}}, nbformat=4, nbformat_minor=5)

    namespace = {"__name__": "__main__"}
    active_outputs = []

    def display(value, raw=False, **kwargs):
        if raw:
            data = value
        elif hasattr(value, "_repr_html_"):
            data = {"text/html": value._repr_html_(), "text/plain": repr(value)}
        else:
            data = {"text/plain": repr(value)}
        active_outputs.append(dict(output_type="display_data", data=data, metadata={}))

    namespace["display"] = display
    count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        count += 1
        active_outputs = cell["outputs"]
        buffer = io.StringIO()
        print(f"Executing notebook cell {count}", flush=True)
        with redirect_stdout(buffer), redirect_stderr(buffer):
            exec(compile(cell["source"], f"notebook-cell-{count}", "exec"), namespace)
        if buffer.getvalue():
            active_outputs.append(dict(output_type="stream", name="stdout", text=buffer.getvalue()))
        cell["execution_count"] = count
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    export_html(notebook, TARGET.with_suffix(".html"))
    print(f"Saved executed notebook: {TARGET}")


def export_html(notebook, target):
    """Write a portable HTML copy with embedded pictures and collapsible code."""
    pieces = ['<!doctype html><html lang="en"><meta charset="utf-8">',
              '<title>Attention and nearest neighbors</title>',
              '<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#20252b}img{width:100%;height:auto}pre{overflow:auto;background:#f4f6f8;padding:16px;font-size:13px}table{border-collapse:collapse;margin:16px 0}td,th{padding:6px 12px;border-bottom:1px solid #ddd}details{margin:12px 0}summary{cursor:pointer;color:#426584}h1,h2{line-height:1.25}@media print{details{display:none}img{break-inside:avoid}h2{break-after:avoid}body{margin:0;max-width:none}}</style>']
    for cell in notebook["cells"]:
        source = cell["source"]
        if cell["cell_type"] == "markdown":
            for paragraph in source.strip().split("\n\n"):
                escaped = html.escape(paragraph)
                if paragraph.startswith("# "):
                    pieces.append("<h1>" + escaped[2:] + "</h1>")
                elif paragraph.startswith("## "):
                    pieces.append("<h2>" + escaped[3:] + "</h2>")
                elif paragraph.startswith("    "):
                    pieces.append("<pre>" + escaped + "</pre>")
                else:
                    pieces.append("<p>" + escaped.replace("\n", " ") + "</p>")
            continue
        pieces.append('<details><summary>Show code</summary><pre>' + html.escape(source) + '</pre></details>')
        for output in cell["outputs"]:
            data = output.get("data", {})
            if "image/png" in data:
                pieces.append('<img alt="Attention analysis figure" src="data:image/png;base64,' + data["image/png"] + '">')
            elif "text/html" in data:
                pieces.append(data["text/html"])
            elif "text/plain" in data:
                pieces.append("<pre>" + html.escape(data["text/plain"]) + "</pre>")
            elif output.get("output_type") == "stream":
                pieces.append("<pre>" + html.escape(output["text"]) + "</pre>")
    pieces.append("</html>")
    target.write_text("\n".join(pieces))


if __name__ == "__main__":
    main()
