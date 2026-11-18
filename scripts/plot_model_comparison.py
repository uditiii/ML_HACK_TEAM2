"""Model comparison plots for the report, drawn from the committed experiment logs.

Reads  results/model_results.csv     (5 distance features, "v1")
       results/model_results_v2.csv  (12 features incl. heading, "v2")
Writes results/figures/model_top1_v1_vs_v2.png   student-test Top-1 per model, both feature sets
       results/figures/model_cv_f1_v2.png        CV macro-F1 (mean +- sd) per model, 12 features

Usage (from anywhere): python scripts/plot_model_comparison.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
BLUE, GREY, INK, INK2, GRID = "#2a78d6", "#a8a7a2", "#0b0b0b", "#52514e", "#e6e5e1"
NAMES = {"logreg": "Logistic regression", "gaussian_nb": "Gaussian Naive Bayes", "knn": "k-NN",
         "decision_tree": "Decision tree", "random_forest": "Random Forest",
         "random_forest_tuned": "Random Forest, tuned", "random_forest_tuned_v2": "Random Forest, tuned"}


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


v1 = pd.read_csv(RES / "model_results.csv").drop_duplicates("model", keep="last")
v2 = pd.read_csv(RES / "model_results_v2.csv").drop_duplicates("model", keep="last")
v1["name"], v2["name"] = v1["model"].map(NAMES), v2["model"].map(NAMES)
both = v1.merge(v2, on="name", suffixes=("_v1", "_v2")).sort_values("test_top1_v2")
FIG.mkdir(parents=True, exist_ok=True)

# 1. Student-test Top-1, 5 vs 12 features (dumbbell: one row per model)
gain_lo, gain_hi = (both["test_top1_v2"] - both["test_top1_v1"]).agg(["min", "max"])
fig, ax = plt.subplots(figsize=(8, 4.6))
y = range(len(both))
ax.hlines(y, both["test_top1_v1"], both["test_top1_v2"], color=GREY, linewidth=2, zorder=1)
ax.scatter(both["test_top1_v1"], y, s=70, color=GREY, zorder=2, label="5 distance features")
ax.scatter(both["test_top1_v2"], y, s=70, color=BLUE, zorder=3, label="12 features (with heading)")
for yi, (a, b) in zip(y, zip(both["test_top1_v1"], both["test_top1_v2"])):
    ax.text(a - 0.012, yi, f"{a:.3f}", ha="right", va="center", fontsize=9, color=INK2)
    ax.text(b + 0.012, yi, f"{b:.3f}", ha="left", va="center", fontsize=9, color=INK)
ax.set_yticks(list(y), both["name"], color=INK)
ax.set_xlim(0.45, 0.95)
ax.set_xlabel("Student-test Top-1 accuracy (classifier only)", color=INK2)
style(ax)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
ax.set_title(f"Heading features lift every model's Top-1 by {gain_lo:.2f}–{gain_hi:.2f}",
             loc="left", color=INK, fontsize=12, fontweight="bold", pad=28)
fig.text(0.01, 0.01, "Source: results/model_results.csv, results/model_results_v2.csv "
         "(HistGradientBoosting was run on 5 features only)", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "model_top1_v1_vs_v2.png", dpi=160)
plt.close(fig)

# 2. CV macro-F1 on 12 features (dot + error bar; the final model in blue)
d = v2.sort_values("cv_macro_f1_mean")
final = d["model"] == "random_forest_tuned_v2"
fig, ax = plt.subplots(figsize=(8, 3.8))
y = range(len(d))
for yi, (m, s, f) in zip(y, zip(d["cv_macro_f1_mean"], d["cv_macro_f1_std"], final)):
    c = BLUE if f else GREY
    ax.errorbar(m, yi, xerr=s, fmt="o", color=c, ecolor=c, elinewidth=2, capsize=0, markersize=8)
    ax.text(m + s + 0.006, yi, f"{m:.3f} ± {s:.3f}", va="center", fontsize=9, color=INK if f else INK2)
ax.set_yticks(list(y), d["name"], color=INK)
ax.set_xlim(0.70, 0.98)
ax.set_xlabel("5-fold GroupKFold CV macro-F1 (mean ± sd over trip folds), train trips", color=INK2)
style(ax)
best = d[final].iloc[0]
ax.set_title(f"Tuned Random Forest has the best CV macro-F1 ({best['cv_macro_f1_mean']:.3f}) on 12 features",
             loc="left", color=INK, fontsize=12, fontweight="bold")
fig.text(0.01, 0.01, "Source: results/model_results_v2.csv", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "model_cv_f1_v2.png", dpi=160)
plt.close(fig)
print("saved results/figures/model_top1_v1_vs_v2.png and results/figures/model_cv_f1_v2.png")
