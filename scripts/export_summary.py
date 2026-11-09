"""Copy the final experiment figures and summary tables into summary/."""

import json
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "final_8seeds"
SUMMARY = ROOT / "summary"


def main():
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    plots = RESULTS / "attention" / "figures"
    names = ["01_locality_overview", "02_attention_concentration", "03_layer_heatmaps",
             "04_query_overlap_distributions", "05_prediction_disagreements", "06_duplicates_and_ties"]
    mapping = {plots / f"{name}.png": f"{name}.png" for name in names}
    mapping.update({plots / "07_notebook_overlap_by_k.png": "07_overlap_by_k.png",
                    plots / "08_notebook_layer_curves.png": "08_layer_curves.png",
                    RESULTS / "figures" / "restricted_context_accuracy.png": "09_restricted_context.png",
                    RESULTS / "figures" / "restricted_context_change.png": "10_restricted_context_change.png",
                    plots / "10_knn_neighborhood_accuracy.png": "11_knn_neighborhood_accuracy.png",
                    plots / "12_imbalance_metrics.png": "12_imbalance_metrics.png",
                    RESULTS / "figures" / "restricted_context_f1_change.png": "13_restricted_context_f1_change.png"})
    figures = SUMMARY / "figures"
    tables = SUMMARY / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for path, name in mapping.items():
        shutil.copyfile(path, figures / name)
    for name in ["baseline_summary.csv", "knn_grid_summary.csv", "restricted_vs_full_summary.csv", "class_balance.csv"]:
        shutil.copyfile(RESULTS / name, tables / name)
    attention = pd.read_csv(RESULTS / "attention" / "summary.csv")
    final = attention[(attention.layer == attention.groupby("model").layer.transform("max")) & (attention.k == 10)]
    final.groupby(["dataset", "model"])[["overlap", "random_overlap", "knn_attention_mass", "distance_rank_correlation"]].agg(["mean", "std"]).to_csv(tables / "attention_final_k10.csv")
    shutil.copyfile(RESULTS / "manifest.json", SUMMARY / "manifest.json")
    print(f"Updated summary: {len(mapping)} figures, five tables, and the completion manifest")


if __name__ == "__main__":
    main()
