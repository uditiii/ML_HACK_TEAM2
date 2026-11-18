"""Shared trip loader — the single entry point every teammate uses to read raw data.

Contract
--------
list_trips(partition) -> list[str]                 sorted trip ids, e.g. ["00000001", ...]
load_trip(partition, trip_id) -> dict with keys:
    "trip_id", "partition"
    "track": DataFrame[lon float64, lat float64, t float64]   raw, NOT cleaned
    "nodes": DataFrame[lon float64, lat float64]   index = node_id (0-based row number)
    "arcs" : DataFrame[from_node int64, to_node int64]        index = arc_id (0-based row number)
    "route": DataFrame[arc_id int64] or None if no .route file
iter_trips(partitions=PARTITIONS) -> yields (partition, trip_id)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_ROOT, PARTITIONS  # noqa: E402

_READ = dict(sep=r"\s+", header=None, engine="c")


def trip_dir(partition: str, trip_id: str) -> Path:
    return Path(DATA_ROOT) / partition / trip_id


def list_trips(partition: str) -> list[str]:
    root = Path(DATA_ROOT) / partition
    if not root.is_dir():
        raise FileNotFoundError(f"No partition folder: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def iter_trips(partitions=PARTITIONS):
    for part in partitions:
        for tid in list_trips(part):
            yield part, tid


def _read(path: Path, names, dtypes) -> pd.DataFrame:
    df = pd.read_csv(path, names=names, dtype=dtypes, **_READ)
    df.index = pd.RangeIndex(len(df))          # 0-based row number = id
    return df


def load_trip(partition: str, trip_id: str) -> dict:
    d = trip_dir(partition, trip_id)
    f = lambda ext: d / f"{trip_id}.{ext}"  # noqa: E731

    # track: read as float64; NaN-tolerant (cleaning happens in preprocess.clean_track)
    track = _read(f("track"), ["lon", "lat", "t"], {"lon": np.float64, "lat": np.float64, "t": np.float64})
    nodes = _read(f("nodes"), ["lon", "lat"], {"lon": np.float64, "lat": np.float64})
    nodes.index.name = "node_id"
    arcs = _read(f("arcs"), ["from_node", "to_node"], {"from_node": np.int64, "to_node": np.int64})
    arcs.index.name = "arc_id"
    route = None
    if f("route").exists():
        route = _read(f("route"), ["arc_id"], {"arc_id": np.int64})

    return {"trip_id": trip_id, "partition": partition,
            "track": track, "nodes": nodes, "arcs": arcs, "route": route}


def validate_trip(trip: dict) -> None:
    """Referential-integrity checks; raises AssertionError on failure."""
    n_nodes, n_arcs = len(trip["nodes"]), len(trip["arcs"])
    a = trip["arcs"]
    assert (a[["from_node", "to_node"]].to_numpy() >= 0).all(), "negative node id in arcs"
    assert (a[["from_node", "to_node"]].to_numpy() < n_nodes).all(), "arc references node >= len(nodes)"
    if trip["route"] is not None:
        r = trip["route"]["arc_id"].to_numpy()
        assert ((r >= 0) & (r < n_arcs)).all(), "route arc_id >= len(arcs)"


if __name__ == "__main__":
    part = "train"
    tid = list_trips(part)[0]
    trip = load_trip(part, tid)
    for k in ("track", "nodes", "arcs", "route"):
        v = trip[k]
        print(f"{k:6s} shape={None if v is None else v.shape}  dtypes={None if v is None else dict(v.dtypes.astype(str))}")
    validate_trip(trip)
    print(f"loader.py OK | {part}/{tid}: {len(list_trips('train'))} train, "
          f"{len(list_trips('student_test'))} student_test trips; integrity asserts passed")
