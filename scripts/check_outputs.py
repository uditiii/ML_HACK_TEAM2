"""Check that Person B's outputs exist and are consistent, before Person C builds features.

Usage (from the repo root): python scripts/check_outputs.py
Prints OK / FAIL per check; exits with code 1 if anything fails.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
CAND_COLS = ["trip_id", "obs_idx", "arc_id", "dist_m", "proj_t", "label"]
OBS_COLS = ["trip_id", "obs_idx", "true_arc", "n_candidates", "has_true_candidate"]
CLEAN_COLS = ["trip_id", "obs_idx", "lon", "lat", "t", "x", "y", "dt", "segment_id"]
LEAK_COLS = {"label", "true_arc", "has_true_candidate"}

failures = 0


def check(name, ok):
    global failures
    print(f"{'OK  ' if ok else 'FAIL'} {name}")
    failures += 0 if ok else 1


for part in ["train", "student_test"]:
    files = {k: P / f"{k}_{part}.parquet" for k in ["candidates", "obs", "clean_tracks"]}
    for k, f in files.items():
        check(f"{part}: {f.name} exists", f.exists())
    if not all(f.exists() for f in files.values()):
        continue
    cand, obs, clean = (pd.read_parquet(files[k]) for k in ["candidates", "obs", "clean_tracks"])
    check(f"{part}: candidates columns {CAND_COLS}", list(cand.columns) == CAND_COLS)
    check(f"{part}: obs columns {OBS_COLS}", list(obs.columns) == OBS_COLS)
    check(f"{part}: clean_tracks columns {CLEAN_COLS}", list(clean.columns) == CLEAN_COLS)
    check(f"{part}: clean_tracks has no label columns", not (LEAK_COLS & set(clean.columns)))
    check(f"{part}: dist_m within [0, 50]", bool(cand["dist_m"].between(0, 50).all()))
    check(f"{part}: proj_t within [0, 1]", bool(cand["proj_t"].between(0, 1).all()))
    check(f"{part}: no duplicate (trip_id, obs_idx, arc_id)",
          not cand.duplicated(["trip_id", "obs_idx", "arc_id"]).any())
    check(f"{part}: at most one positive per obs",
          int(cand.groupby(["trip_id", "obs_idx"])["label"].sum().max()) <= 1)
    keys = cand[["trip_id", "obs_idx"]].drop_duplicates()
    check(f"{part}: every candidate obs is in obs file",
          len(keys.merge(obs[["trip_id", "obs_idx"]])) == len(keys))
    check(f"{part}: every candidate obs is in clean_tracks",
          len(keys.merge(clean[["trip_id", "obs_idx"]])) == len(keys))
    n_c = cand.groupby(["trip_id", "obs_idx"]).size().rename("n").reset_index()
    m = obs.merge(n_c, how="left").fillna({"n": 0})
    check(f"{part}: obs.n_candidates matches candidate rows", bool((m["n"] == m["n_candidates"]).all()))
    if part == "train":
        check("train: obs_idx are multiples of 5", bool((obs["obs_idx"] % 5 == 0).all()))
    else:
        per_trip = clean.groupby("trip_id").size()
        check("student_test: obs covers every cleaned point",
              bool((obs.groupby("trip_id").size() == per_trip).all()))
    print(f"     {part}: {len(cand)} candidate rows, {len(obs)} obs, {len(clean)} cleaned points, "
          f"{obs['trip_id'].nunique()} trips")

print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
