"""Task 3 check: label quality of v1 (nearest route arc) vs v2 (monotone DP), per partition.

Usage (from the repo root): python scripts/compare_labels.py
Prints, per partition and version: % unlabelled, mean label distance (m), share of backward
label steps, and the % of labelled points where v1 and v2 choose a different arc.
Saves the per-trip table to results/label_versions_comparison.csv.
Uses the .route file ONLY to create labels (this is label creation, not features).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import PARTITIONS
from scripts.build_candidates import load_projected
from src.candidates.labels import label_points, label_quality
from src.data.loader import list_trips

rows = []
for partition in PARTITIONS:
    for trip_id in list_trips(partition):
        trip, clean, geom, _ = load_projected(partition, trip_id)
        res = {}
        for method in ["v1", "v2"]:
            true_arc, pos = label_points(clean, geom, trip["route"], method=method, return_positions=True)
            res[method] = true_arc
            rows.append({"partition": partition, "trip_id": trip_id, "method": method,
                         **label_quality(clean, geom, true_arc, pos)})
        lab = res["v1"] >= 0
        rows[-1]["pct_disagree_v1_v2"] = rows[-2]["pct_disagree_v1_v2"] = (
            float(100 * (res["v1"][lab] != res["v2"][lab]).mean()) if lab.any() else np.nan)
        print(f"{partition} {trip_id} done", flush=True)

df = pd.DataFrame(rows)
(ROOT / "results").mkdir(exist_ok=True)
df.to_csv(ROOT / "results" / "label_versions_comparison.csv", index=False)

# point-weighted averages over all trips of a partition
summary = []
for (partition, method), g in df.groupby(["partition", "method"], sort=False):
    w = g["n_labelled"]
    summary.append({"partition": partition, "method": method, "n_points": int(g["n_points"].sum()),
                    "pct_unlabelled": 100 * (1 - w.sum() / g["n_points"].sum()),
                    "mean_label_dist_m": float(np.average(g["mean_label_dist_m"], weights=w)),
                    "backward_step_share": float(np.average(g["backward_step_share"].fillna(0), weights=w)),
                    "pct_disagree_v1_v2": float(np.average(g["pct_disagree_v1_v2"].fillna(0), weights=w))})
print()
print("Point-weighted over all trips (per-trip values in results/label_versions_comparison.csv):")
print(pd.DataFrame(summary).to_string(index=False))
