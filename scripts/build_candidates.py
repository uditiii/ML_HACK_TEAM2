"""Task 4 + 5: build candidate and obs parquet files, then the summary.

Usage (from the repo root):
  python scripts/build_candidates.py                         # all trips, both partitions, labels v2 (default)
  python scripts/build_candidates.py --max-train-trips 30    # first 30 train trips (sorted) + all test
  python scripts/build_candidates.py --label-method v1   # simpler nearest-route-arc labels
  python scripts/build_candidates.py --overwrite             # recompute trips already done

Each trip is saved as its own part file under data/processed/parts/<method>/<partition>/, so a
partial run can be continued later; only missing trips are computed. At the end all existing
parts are merged into data/processed/candidates_<partition>.parquet, obs_<partition>.parquet and
clean_tracks_<partition>.parquet (every cleaned GPS point: trip_id, obs_idx, lon, lat, t, x, y,
dt, segment_id; used by Person C's features).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import PARTITIONS, RADIUS_M, TRAIN_STEP
from src.candidates.generate import generate_candidates
from src.candidates.geometry import arc_geometry, trip_projector
from src.candidates.labels import label_points, label_quality
from src.data.loader import list_trips, load_trip
from src.data.preprocess import clean_track

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_projected(partition, trip_id):
    """Load + clean one trip and project GPS points and arcs with ONE projector (same centre).

    Returns (trip, clean, geom, report). No labels are computed here.
    """
    trip = load_trip(partition, trip_id)
    clean, report = clean_track(trip["track"])
    proj = trip_projector(clean, report)
    clean = clean.copy()
    clean["x"], clean["y"] = proj(clean["lon"].to_numpy(), clean["lat"].to_numpy())
    geom = arc_geometry(trip["nodes"], trip["arcs"], proj)
    return trip, clean, geom, report


def prepare_trip(partition, trip_id, label_method="v2"):
    """load_projected() + label every cleaned point.

    Returns (clean, geom, true_arc, label_stats, report). Used by the build and the check scripts.
    """
    trip, clean, geom, report = load_projected(partition, trip_id)
    true_arc, route_pos = label_points(clean, geom, trip["route"], method=label_method,
                                       return_positions=True)
    stats = label_quality(clean, geom, true_arc, route_pos)
    return clean, geom, true_arc, stats, report


def selected_obs(partition, n_points):
    """Train: every TRAIN_STEP-th cleaned point; student_test: all points."""
    obs = np.arange(n_points)
    return obs[obs % TRAIN_STEP == 0] if partition == "train" else obs


def build_trip(partition, trip_id, label_method):
    """Candidates + labels for one trip. Returns (cand_df, obs_df, info dict)."""
    clean, geom, true_arc, stats, report = prepare_trip(partition, trip_id, label_method)
    obs = selected_obs(partition, len(clean))
    cand = generate_candidates(trip_id, clean["x"].to_numpy()[obs], clean["y"].to_numpy()[obs],
                               obs, geom, RADIUS_M)
    cand["label"] = (cand["arc_id"].to_numpy() == true_arc[cand["obs_idx"].to_numpy()]).astype(np.int8)

    n_cand = np.bincount(cand["obs_idx"].to_numpy(), minlength=len(clean))[obs]
    n_pos = np.bincount(cand["obs_idx"].to_numpy(), weights=cand["label"].to_numpy(),
                        minlength=len(clean))[obs]
    assert n_pos.max(initial=0) <= 1, "an obs has more than one positive candidate"
    obs_df = pd.DataFrame({"trip_id": trip_id, "obs_idx": obs, "true_arc": true_arc[obs],
                           "n_candidates": n_cand.astype(np.int64), "has_true_candidate": n_pos > 0})
    info = {"trip_id": trip_id, "n_arcs": int(len(geom)), **report, **stats,
            "n_obs": int(len(obs)), "n_rows": int(len(cand))}
    # ALL cleaned points (not only the selected obs), so features can use neighbouring points.
    # No label columns here: this file is safe as a feature input.
    if "obs_idx" in clean.columns:   # Person A's clean_track already numbers the rows
        assert (clean["obs_idx"].to_numpy() == np.arange(len(clean))).all(), "obs_idx must be 0..n-1"
    clean_df = clean[["lon", "lat", "t", "x", "y", "dt", "segment_id"]].copy()
    clean_df.insert(0, "obs_idx", np.arange(len(clean), dtype=np.int64))
    clean_df.insert(0, "trip_id", trip_id)
    return cand, obs_df, info, clean_df


def merge_parts(part_dir, partition):
    """Concatenate all per-trip part files of a partition into the two contract parquet files."""
    cands = sorted(part_dir.glob("*_cand.parquet"))
    if not cands:
        return None, None
    cand = pd.concat([pd.read_parquet(p) for p in cands], ignore_index=True)
    obs = pd.concat([pd.read_parquet(p) for p in sorted(part_dir.glob("*_obs.parquet"))],
                    ignore_index=True)
    cand = cand.astype({"trip_id": str, "obs_idx": np.int64, "arc_id": np.int64,
                        "dist_m": np.float64, "proj_t": np.float64, "label": np.int8})
    obs = obs.astype({"trip_id": str, "obs_idx": np.int64, "true_arc": np.int64,
                      "n_candidates": np.int64, "has_true_candidate": bool})
    cand.to_parquet(PROCESSED / f"candidates_{partition}.parquet", index=False)
    obs.to_parquet(PROCESSED / f"obs_{partition}.parquet", index=False)
    clean = pd.concat([pd.read_parquet(p) for p in sorted(part_dir.glob("*_clean.parquet"))],
                      ignore_index=True)
    clean.to_parquet(PROCESSED / f"clean_tracks_{partition}.parquet", index=False)
    return cand, obs


def summarise(partition, cand, obs, n_trips_total):
    """Task 5: one summary row per partition."""
    labelled = obs["true_arc"] >= 0
    return {"partition": partition,
            "n_trips": int(obs["trip_id"].nunique()), "n_trips_total": n_trips_total,
            "n_obs": int(len(obs)),
            "pct_unlabelled": float(100 * (~labelled).mean()),
            "mean_candidates_per_obs": float(obs["n_candidates"].mean()),
            "pct_obs_zero_candidates": float(100 * (obs["n_candidates"] == 0).mean()),
            "total_rows": int(len(cand)),
            "positive_rate": float(cand["label"].mean()),
            "candidate_miss_rate": float((~obs.loc[labelled, "has_true_candidate"]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=PARTITIONS)
    ap.add_argument("--label-method", choices=["v1", "v2"], default="v2")
    ap.add_argument("--max-train-trips", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for partition in args.partitions:
        trips = list_trips(partition)
        todo = trips[:args.max_train_trips] if (partition == "train" and args.max_train_trips) else trips
        part_dir = PROCESSED / "parts" / args.label_method / partition
        part_dir.mkdir(parents=True, exist_ok=True)
        t_part = time.time()
        for i, trip_id in enumerate(todo, 1):
            cpath, opath = part_dir / f"{trip_id}_cand.parquet", part_dir / f"{trip_id}_obs.parquet"
            kpath = part_dir / f"{trip_id}_clean.parquet"
            if cpath.exists() and opath.exists() and kpath.exists() and not args.overwrite:
                print(f"[{partition} {i}/{len(todo)}] {trip_id} already built, skipped", flush=True)
                continue
            t0 = time.time()
            cand, obs_df, info, clean_df = build_trip(partition, trip_id, args.label_method)
            cand.to_parquet(cpath, index=False)
            obs_df.to_parquet(opath, index=False)
            clean_df.to_parquet(kpath, index=False)
            (part_dir / f"{trip_id}_info.json").write_text(json.dumps(info))
            print(f"[{partition} {i}/{len(todo)}] {trip_id}: points={info['n_points']} obs={info['n_obs']} "
                  f"arcs={info['n_arcs']} rows={info['n_rows']} unlabelled={info['pct_unlabelled']:.1f}% "
                  f"time={time.time() - t0:.1f}s", flush=True)
        print(f"{partition}: loop time {time.time() - t_part:.1f}s", flush=True)

        cand, obs = merge_parts(part_dir, partition)
        if cand is None:
            continue
        rows.append(summarise(partition, cand, obs, len(trips)))
        infos = pd.DataFrame([json.loads(p.read_text()) for p in sorted(part_dir.glob("*_info.json"))])
        infos.to_csv(RESULTS / f"label_quality_{args.label_method}_{partition}.csv", index=False)
        print(f"{partition} labels {args.label_method}: mean label dist (mean over trips) = "
              f"{infos['mean_label_dist_m'].mean():.2f} m, backward step share (mean over trips) = "
              f"{infos['backward_step_share'].mean():.4f}", flush=True)

    if rows:
        summary = pd.DataFrame(rows)
        summary.insert(1, "label_method", args.label_method)
        summary.to_csv(RESULTS / "candidates_summary.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
