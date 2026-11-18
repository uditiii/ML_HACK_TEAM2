"""Person B: classifier-only vs classifier + HMM/Viterbi (terminal version of notebooks/05_hmm.ipynb).

Usage (from anywhere):
  python scripts/run_hmm_experiment.py                  # search on 15 train trips, then student_test
  python scripts/run_hmm_experiment.py --n-train-trips 5   # quicker search
  python scripts/run_hmm_experiment.py --skip-search    # DEFAULT_PARAMS, student_test only

Writes results/hmm_param_search.csv, results/hmm_chosen_params.json, results/hmm_comparison.csv.
Evaluator data is never touched: only train (search) and student_test (one final evaluation).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from config import RANDOM_STATE
from src.data.loader import list_trips
from src.hmm.experiment import (compare, error_runs, load_scorer, load_trip_for_hmm, param_search,
                                plot_trip_comparison, score_trips)
from src.hmm.viterbi import DEFAULT_PARAMS, decode_trip

ap = argparse.ArgumentParser()
ap.add_argument("--n-train-trips", type=int, default=15)
ap.add_argument("--skip-search", action="store_true")
ap.add_argument("--no-oof", action="store_true", help="score train trips in-sample (not recommended)")
args = ap.parse_args()
RES = ROOT / "results"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

scorer = load_scorer()
params = dict(DEFAULT_PARAMS)

# ---- 1. parameter search on a subsample of TRAIN trips (full-rate points) --------------------
if not args.skip_search:
    rng = np.random.default_rng(RANDOM_STATE)
    all_train = list_trips("train")
    picked = sorted(rng.choice(all_train, size=min(args.n_train_trips, len(all_train)), replace=False).tolist())
    print(f"\n[1] parameter search on {len(picked)} train trips: {picked}", flush=True)
    t0 = time.time()
    trips = [load_trip_for_hmm("train", t, scorer) for t in picked]
    score_trips(trips, scorer, out_of_fold=not args.no_oof)
    search, params = param_search(trips)
    search.insert(0, "model", scorer["source"])
    search.insert(1, "oof_scores", not args.no_oof)
    search.to_csv(RES / "hmm_param_search.csv", index=False)
    print(search.sort_values("top1", ascending=False).head(10).to_string(index=False))
    print(f"chosen (best pooled Top-1 on train): {params}   [{time.time() - t0:.1f}s]", flush=True)
(RES / "hmm_chosen_params.json").write_text(json.dumps(
    {"params": params, "model": scorer["source"], "searched": not args.skip_search}, indent=2))

# ---- 2. one evaluation on student_test with the chosen parameters ---------------------------
print("\n[2] student_test: classifier-only vs HMM", flush=True)
t0 = time.time()
test = [load_trip_for_hmm("student_test", t, scorer) for t in list_trips("student_test")]
score_trips(test, scorer)
comp = compare(test, params)
comp.insert(1, "model", scorer["source"])
comp.to_csv(RES / "hmm_comparison.csv", index=False)
print(comp[["trip_id", "n_points", "n_labelled", "top1_clf", "top1_hmm", "cont_clf", "cont_hmm",
            "reverse_err_clf", "reverse_err_hmm", "pct_points_changed_by_hmm"]].to_string(index=False))
print(f"[{time.time() - t0:.1f}s]")

# ---- 3. where the HMM helped most / least --------------------------------------------------
per = comp[comp["trip_id"] != "OVERALL"].copy()
per["top1_gain"] = per["top1_hmm"] - per["top1_clf"]
per["cont_gain"] = per["cont_hmm"] - per["cont_clf"]
cols = ["trip_id", "top1_clf", "top1_hmm", "top1_gain", "cont_clf", "cont_hmm", "cont_gain", "reverse_err_clf",
        "reverse_err_hmm", "n_segments", "mean_candidates", "pct_points_changed_by_hmm", "n_labelled"]
print("\n[3] HMM helped most (top 2 by Top-1 gain):")
print(per.sort_values("top1_gain", ascending=False).head(2)[cols].to_string(index=False))
print("HMM helped least / hurt (bottom 1):")
print(per.sort_values("top1_gain").head(1)[cols].to_string(index=False))

# ---- 4. evidence for the report: longest wrong runs + side-by-side maps -----------------------
by_id = {d["trip_id"]: d for d in test}
helped = per.sort_values("top1_gain", ascending=False)["trip_id"].head(2).tolist()
least = per.sort_values("top1_gain")["trip_id"].head(1).tolist()
worst = per.sort_values("top1_hmm")["trip_id"].head(1).tolist()
print("\n[4] longest runs of wrong HMM points (least-helped trip and lowest-HMM-accuracy trip):")
for tid in dict.fromkeys(least + worst):
    d = by_id[tid]
    runs, n_wrong, n_rev = error_runs(d, decode_trip(d["cand"], d["proba"], d["arcs"], d["clean"], params))
    print(f"trip {tid}: {n_wrong} wrong labelled points, {n_rev} of them the reverse twin of the true arc")
    print(runs.to_string(index=False))
for tid in dict.fromkeys(helped + least + worst):
    print("saved", plot_trip_comparison(by_id[tid], params, RES / "figures" / f"hmm_student_test_{tid}.png"))
