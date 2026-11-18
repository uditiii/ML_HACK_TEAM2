"""Sarvani's quick model run, as a terminal script (same code as the notebook cell).

Usage (from anywhere):
  python scripts/run_quick_models.py
  python scripts/run_quick_models.py --models decision_tree,logreg      # only some models
Quick features straight from the candidate table (no route, no labels used as inputs).
Each model prints one line; rows are appended to results/model_results.csv.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                      # so results/ and data/ paths work from any folder

import pandas as pd

from src.models.experiment import make_folds, run_experiment
from src.models.models import get_models

DEFAULT_MODELS = "decision_tree,logreg,gaussian_nb,hist_gb,random_forest"


def quick_feats(df):
    g = df.groupby(["trip_id", "obs_idx"])["dist_m"]
    df["dist_rank"] = g.rank(method="min")
    df["dist_minus_min"] = df["dist_m"] - g.transform("min")
    df["n_candidates"] = g.transform("size")
    return df


ap = argparse.ArgumentParser()
ap.add_argument("--models", default=DEFAULT_MODELS)
ap.add_argument("--results", default="results/model_results.csv")
args = ap.parse_args()

train = quick_feats(pd.read_parquet("data/processed/candidates_train.parquet"))
test = quick_feats(pd.read_parquet("data/processed/candidates_student_test.parquet"))
obs_tr = pd.read_parquet("data/processed/obs_train.parquet")
obs_te = pd.read_parquet("data/processed/obs_student_test.parquet")
for d in (train, test, obs_tr, obs_te):
    d["trip_id"] = d["trip_id"].astype(str)
FEATURES = ["dist_m", "proj_t", "dist_rank", "dist_minus_min", "n_candidates"]

folds = make_folds(train)
wanted = args.models.split(",")
for name, fam, model, cfg, mx in get_models():
    if name in wanted:
        print(f"running {name} ...", flush=True)
        run_experiment(model, name, fam, cfg, train, test, obs_tr, obs_te, FEATURES, folds,
                       results_path=args.results, max_train_rows=mx)
print(f"done; results appended to {args.results}")
