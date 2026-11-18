"""Visual check: draw one GPS point, its 50 m circle, all nearby arcs, its candidates and its label.

Usage (from the repo root):
  python scripts/plot_candidates.py                                   # student_test 00000000, obs 100
  python scripts/plot_candidates.py --partition train --trip 00000001 --obs 250
Saves results/figures/candidates_<partition>_<trip>_<obs>.png and prints the candidate table.
What you should see: every blue (candidate) arc touches the dashed 50 m circle or lies inside
it; no grey arc enters the circle; the green arc (label) is one of the blue ones.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import RADIUS_M
from scripts.build_candidates import load_projected
from src.candidates.generate import generate_candidates
from src.candidates.labels import label_points

ap = argparse.ArgumentParser()
ap.add_argument("--partition", default="student_test")
ap.add_argument("--trip", default="00000000")
ap.add_argument("--obs", type=int, default=100)
ap.add_argument("--label-method", choices=["v1", "v2"], default="v2")
args = ap.parse_args()

trip, clean, geom, _ = load_projected(args.partition, args.trip)
true_arc = label_points(clean, geom, trip["route"], method=args.label_method)
o = args.obs
px, py = clean["x"].to_numpy()[o], clean["y"].to_numpy()[o]
cand = generate_candidates(args.trip, [px], [py], [o], geom)
print(cand.to_string(index=False))
print(f"label ({args.label_method}) true_arc = {true_arc[o]}")

view = 2.5 * RADIUS_M
near = geom[(np.maximum(geom.ax, geom.bx) >= px - view) & (np.minimum(geom.ax, geom.bx) <= px + view)
            & (np.maximum(geom.ay, geom.by) >= py - view) & (np.minimum(geom.ay, geom.by) <= py + view)]
fig, ax = plt.subplots(figsize=(7, 7))
for _, r in near.iterrows():
    ax.plot([r.ax, r.bx], [r.ay, r.by], color="0.75", lw=1, zorder=1)
for a in cand["arc_id"]:
    r = geom.loc[a]
    ax.plot([r.ax, r.bx], [r.ay, r.by], color="tab:blue", lw=2.5, zorder=2)
if true_arc[o] >= 0:
    r = geom.loc[true_arc[o]]
    ax.plot([r.ax, r.bx], [r.ay, r.by], color="tab:green", lw=5, alpha=0.7, zorder=3)
nb = slice(max(0, o - 15), o + 16)
ax.plot(clean["x"].to_numpy()[nb], clean["y"].to_numpy()[nb], ".", color="tab:orange", zorder=4)
ax.plot(px, py, "x", color="red", ms=12, mew=3, zorder=5)
ax.add_patch(plt.Circle((px, py), RADIUS_M, fill=False, ls="--", color="red"))
ax.plot([], [], color="0.75", label="other arcs")
ax.plot([], [], color="tab:blue", lw=2.5, label=f"candidates ({len(cand)})")
ax.plot([], [], color="tab:green", lw=5, alpha=0.7, label=f"label {args.label_method} (arc {true_arc[o]})")
ax.plot([], [], ".", color="tab:orange", label="neighbouring GPS points")
ax.plot([], [], "x", color="red", label=f"GPS point obs {o}")
ax.set_xlim(px - view, px + view)
ax.set_ylim(py - view, py + view)
ax.set_aspect("equal")
ax.set_xlabel("x (m, east)")
ax.set_ylabel("y (m, north)")
ax.set_title(f"{args.partition} trip {args.trip}, obs {o}: candidates within {RADIUS_M} m")
ax.legend(loc="upper right", fontsize=8)
out = ROOT / "results" / "figures" / f"candidates_{args.partition}_{args.trip}_{o}.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")
