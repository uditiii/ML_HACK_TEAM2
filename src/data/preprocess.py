"""GPS track cleaning. Uses ONLY the track itself — never the road network or the route —
so the same function can run unchanged on unseen/evaluator trips (no leakage)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GAP_THRESHOLD_S  # noqa: E402
from src.data.geo import make_projector  # noqa: E402


def clean_track(track: pd.DataFrame, gap_threshold_s: float = GAP_THRESHOLD_S):
    """Clean a raw track (columns lon, lat, t).

    Returns (clean_df, report). clean_df columns:
        obs_idx, lon, lat, t, x, y, dt, segment_id
    report: dict of issue counts + projection origin (lon0, lat0).
    """
    df = track[["lon", "lat", "t"]].copy()
    rep = {"raw_points": int(len(df))}

    # a. invalid: NaN or out-of-range coordinates
    bad = df.isna().any(axis=1) | ~df["lon"].between(-180, 180) | ~df["lat"].between(-90, 90)
    rep["dropped_invalid"] = int(bad.sum())
    df = df.loc[~bad]

    # b. exact duplicate rows (same lon, lat, t)
    dup = df.duplicated(keep="first")
    rep["dropped_duplicates"] = int(dup.sum())
    df = df.loc[~dup]

    # c. stable sort by time; count rows that were not in their sorted position
    order = np.argsort(df["t"].to_numpy(), kind="stable")
    rep["out_of_order"] = int((order != np.arange(len(order))).sum())
    df = df.iloc[order]

    # d. same timestamp, different position -> keep first
    dupt = df["t"].duplicated(keep="first")
    rep["duplicate_timestamps"] = int(dupt.sum())
    df = df.loc[~dupt]

    # e. project to local metres around the trip's mean lon/lat
    if len(df):
        lon0, lat0 = float(df["lon"].mean()), float(df["lat"].mean())
    else:
        lon0 = lat0 = float("nan")
    to_xy, _ = make_projector(lon0, lat0)
    x, y = to_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    df = df.assign(x=x, y=y)
    rep["lon0"], rep["lat0"] = lon0, lat0

    # f. time step, first = 0
    dt = np.diff(df["t"].to_numpy(), prepend=df["t"].to_numpy()[:1]) if len(df) else np.array([])
    df = df.assign(dt=dt)

    # g. segment id: new segment after every gap > threshold
    gaps = dt > gap_threshold_s
    df = df.assign(segment_id=np.cumsum(gaps).astype(np.int64))
    rep["n_gaps_over_threshold"] = int(gaps.sum())
    rep["max_gap_s"] = float(dt.max()) if len(df) else 0.0
    rep["n_segments"] = int(gaps.sum() + 1) if len(df) else 0

    # h. fresh 0..n-1 index
    df = df.reset_index(drop=True)
    df.insert(0, "obs_idx", np.arange(len(df), dtype=np.int64))
    rep["clean_points"] = int(len(df))
    return df, rep


if __name__ == "__main__":
    # Synthetic test hitting every branch.
    raw = pd.DataFrame({
        "lon": [8.60, 8.60, np.nan, 8.61, 200.0, 8.62, 8.63, 8.64, 8.64],
        "lat": [49.86, 49.86, 49.86, 49.86, 49.86, 49.86, 49.86, 49.87, 49.88],
        "t":   [1.0,   1.0,   2.0,   3.0,   4.0,  2.5,  60.0, 61.0, 61.0],
    })
    c, r = clean_track(raw)
    assert r["dropped_invalid"] == 2 and r["dropped_duplicates"] == 1, r
    assert r["out_of_order"] == 2 and r["duplicate_timestamps"] == 1, r   # 2.5 jumps before 3.0
    assert r["n_segments"] == 2 and c["segment_id"].tolist() == [0, 0, 0, 1, 1], c
    assert c["t"].is_monotonic_increasing and c["obs_idx"].tolist() == list(range(len(c)))
    print("preprocess.py OK |", {k: v for k, v in r.items() if k not in ("lon0", "lat0")})
