"""Build data/processed/train_table.parquet and test_table.parquet.

Reads   data/processed/candidates_{train,student_test}.parquet     (Person B)
        data/processed/clean_tracks_{train,student_test}.parquet   (Person B; EVERY cleaned point, has x, y, segment_id)
        nodes + arcs of each trip via src.data.loader.load_trip     (Person A)
Adds    the features from src/features/build_features.add_features, trip by trip
Writes  data/processed/train_table.parquet
        data/processed/test_table.parquet                           (= the 10 student-test trips)

The "route" returned by load_trip is dropped here, so it can never reach feature code.
Evaluator-testing trips do not exist on our side and are never touched.

Run from the REPO ROOT (load_trip uses the relative path data/raw/Student folder):
    python scripts/build_tables.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_features as bf  # noqa: E402
from src.features.build_features import ALL_FEATURES, add_features, assert_features_clean  # noqa: E402

OUTPUTS = {"train": "train_table", "student_test": "test_table"}   # partition -> output file stem
TRAIN_STEP = 5                                                    # config.TRAIN_STEP (informational check only)

_TRACKS: dict = {}


def _clean_tracks(root: Path, partition: str) -> dict:
    """{trip_id: DataFrame of that trip's cleaned track}, read once per partition."""
    if partition not in _TRACKS:
        df = pd.read_parquet(root / "data" / "processed" / f"clean_tracks_{partition}.parquet")
        _TRACKS[partition] = {k: g.reset_index(drop=True) for k, g in df.groupby("trip_id", sort=False)}
    return _TRACKS[partition]


def load_trip_assets(root: Path, partition: str, trip_id: str):
    """Return (trip, clean_track) for one trip.

    trip         {"nodes": DataFrame(lon, lat), "arcs": DataFrame(from_node, to_node)}  (row index = id)
    clean_track  Person B's cleaned track for this trip (obs_idx, lon, lat, t, x, y, dt, segment_id),
                 containing EVERY cleaned point even for train, so speed/heading use the 1 Hz data.
    """
    from src.data.loader import load_trip            # Person A's loader

    t = load_trip(partition, trip_id)
    trip = {"nodes": t["nodes"], "arcs": t["arcs"]}   # t["route"] deliberately NOT passed on
    tracks = _clean_tracks(root, partition)
    if trip_id not in tracks:
        raise KeyError(f"{trip_id} missing from clean_tracks_{partition}.parquet")
    return trip, tracks[trip_id]


def verify_against_b_geometry(partition: str, trip_id: str, trip_rows: pd.DataFrame) -> None:
    """Optional, non-fatal: compare our arc_length_m with Person B's geom.length_m for one trip."""
    try:
        from scripts.build_candidates import load_projected  # Person B
        _, _, geom, _ = load_projected(partition, trip_id)
        sub = trip_rows[["arc_id", "arc_length_m"]].drop_duplicates("arc_id")
        theirs = geom["length_m"].reindex(sub["arc_id"]).to_numpy()
        diff = np.abs(sub["arc_length_m"].to_numpy() - theirs)
        print(f"[{partition}] arc_length_m vs B's geom.length_m on {trip_id}: max abs diff = {np.nanmax(diff):.4f} m "
              f"over {len(sub)} arcs (expect ~0)")
    except Exception as e:  # noqa: BLE001 -- verification must never break the build
        print(f"[{partition}] (skipped geometry cross-check: {type(e).__name__}: {e})")


def build_partition(root: Path, partition: str, loader=load_trip_assets, verify: bool = True) -> pd.DataFrame:
    proc = root / "data" / "processed"
    src = proc / f"candidates_{partition}.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found -- run scripts/build_candidates.py first")
    cand = pd.read_parquet(src)
    need = {"trip_id", "obs_idx", "arc_id", "dist_m", "proj_t", "label"}
    if need - set(cand.columns):
        raise KeyError(f"{src.name} is missing columns {sorted(need - set(cand.columns))}")
    print(f"[{partition}] read {src.name}: {cand.shape}, trips={cand['trip_id'].nunique()}")

    if partition == "train":
        off = (cand["obs_idx"] % TRAIN_STEP != 0).mean()
        if off > 0:
            print(f"[{partition}] WARNING: {off:.1%} of candidate rows have obs_idx % {TRAIN_STEP} != 0")

    parts, t0, checked = [], time.time(), False
    for i, (trip_id, g) in enumerate(cand.groupby("trip_id", sort=False), 1):
        trip, trk = loader(root, partition, trip_id)
        feats = add_features(g.reset_index(drop=True), trip, trk)
        parts.append(feats)
        if verify and not checked:
            verify_against_b_geometry(partition, trip_id, feats)
            checked = True
        if i % 10 == 0:
            print(f"   ... {i} trips ({time.time() - t0:.1f}s)")
    table = pd.concat(parts, ignore_index=True)
    assert_features_clean(table, ALL_FEATURES)
    assert len(table) == len(cand), "row count changed during feature building"

    out = proc / f"{OUTPUTS[partition]}.parquet"
    table.to_parquet(out, index=False)

    grp = table.groupby(["trip_id", "obs_idx"])["label"]
    print(f"[{partition}] wrote {out.relative_to(root)}  shape={table.shape}")
    print(f"[{partition}] positive rate (label==1): {table['label'].mean():.4f}   obs with candidates={grp.ngroups}   "
          f"obs whose true arc is among candidates: {grp.max().mean():.4f}   "
          f"rows with a valid heading: {table['heading_valid'].mean():.4f}   ({time.time() - t0:.1f}s)")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--no-verify", action="store_true", help="skip the cross-check against Person B's geometry")
    a = ap.parse_args()
    print("projection:", bf.PROJECTOR_SOURCE)
    for partition in OUTPUTS:
        build_partition(a.root, partition, verify=not a.no_verify)


if __name__ == "__main__":
    main()
