"""Tiny local stand-in for B's candidate generator -- TESTING ONLY.

Builds a synthetic grid road network (both directions per street), a noisy GPS
track along a route (with a stationary stretch and a GPS dropout), and a
candidate table with the same columns B's generate_candidates produces:
    trip_id, obs_idx, arc_id, dist_m, proj_t, label, true_arc, has_true_candidate

Nothing here is real data. Never use its outputs for results or the report.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

LON0, LAT0 = 5.000, 52.000
DLON, DLAT = 0.0010, 0.0007           # ~68 m / ~78 m grid spacing at lat 52
M_PER_DEG_LAT = 111_195.0
M_PER_DEG_LON = 111_195.0 * np.cos(np.radians(LAT0))


def make_grid(n: int = 7):
    nodes = np.array([[LON0 + c * DLON, LAT0 + r * DLAT] for r in range(n) for c in range(n)])
    nid = lambda r, c: r * n + c
    arcs = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                arcs += [(nid(r, c), nid(r, c + 1)), (nid(r, c + 1), nid(r, c))]
            if r + 1 < n:
                arcs += [(nid(r, c), nid(r + 1, c)), (nid(r + 1, c), nid(r, c))]
    arcs = np.array(arcs, dtype=int)
    arc_of = {(a, b): i for i, (a, b) in enumerate(arcs)}
    return nodes, arcs, arc_of, nid


def _to_xy(lon, lat):
    return (np.asarray(lon) - LON0) * M_PER_DEG_LON, (np.asarray(lat) - LAT0) * M_PER_DEG_LAT


def make_trip(trip_id: str = "trip_000", seed: int = 0, noise_m: float = 6.0, radius_m: float = 45.0):
    rng = np.random.default_rng(seed)
    nodes, arcs, arc_of, nid = make_grid()

    # route: east along row 3 from col 0 to col 5, then north along col 5 to row 6
    node_path = [nid(3, c) for c in range(0, 6)] + [nid(r, 5) for r in range(4, 7)]
    route = np.array([arc_of[(a, b)] for a, b in zip(node_path[:-1], node_path[1:])])

    # true positions at 10 m/s, 1 Hz (arc-length parametrised), with a stationary pause
    px, py = _to_xy(nodes[node_path, 0], nodes[node_path, 1])
    seglen = np.hypot(np.diff(px), np.diff(py))
    cum = np.r_[0, np.cumsum(seglen)]
    s = np.arange(0, cum[-1], 10.0)
    pause_at = len(s) // 2
    s = np.r_[s[:pause_at], np.full(8, s[pause_at]), s[pause_at + 1:]]       # 8 s stationary
    tx, ty = np.interp(s, cum, px), np.interp(s, cum, py)
    true_arc_idx = np.minimum(np.searchsorted(cum, s, side="right") - 1, len(route) - 1)
    t = np.arange(len(s), dtype=float)
    # GPS dropout: skip 5 s after the first third -> new segment_id
    drop_at = len(s) // 3
    t[drop_at:] += 5.0
    segment_id = (np.arange(len(s)) >= drop_at).astype(int)

    jitter = np.where((np.arange(len(s)) >= pause_at) & (np.arange(len(s)) < pause_at + 8), 0.3, noise_m)
    gx, gy = tx + rng.normal(0, 1, len(s)) * jitter, ty + rng.normal(0, 1, len(s)) * jitter
    glon, glat = LON0 + gx / M_PER_DEG_LON, LAT0 + gy / M_PER_DEG_LAT

    clean_track = pd.DataFrame(
        {"obs_idx": np.arange(len(s)), "lon": glon, "lat": glat, "t": t, "segment_id": segment_id}
    )
    trip = SimpleNamespace(trip_id=trip_id, nodes=nodes, arcs=arcs, route=route)  # route: test labels only

    # ---- candidates: arcs within radius_m of each GPS point ----
    ax, ay = _to_xy(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1])
    bx, by = _to_xy(nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
    abx, aby = bx - ax, by - ay
    L2 = abx ** 2 + aby ** 2
    rows = []
    for k in range(len(s)):
        u = np.clip(((gx[k] - ax) * abx + (gy[k] - ay) * aby) / L2, 0, 1)
        d = np.hypot(gx[k] - (ax + u * abx), gy[k] - (ay + u * aby))
        for a in np.flatnonzero(d <= radius_m):
            rows.append((trip_id, k, a, d[k * 0 + a], u[a]))
    cand = pd.DataFrame(rows, columns=["trip_id", "obs_idx", "arc_id", "dist_m", "proj_t"])
    cand["true_arc"] = route[true_arc_idx][cand["obs_idx"].to_numpy()]
    cand["label"] = (cand["arc_id"] == cand["true_arc"]).astype(int)
    cand["has_true_candidate"] = cand.groupby("obs_idx")["label"].transform("max").astype(int)
    return trip, clean_track, cand
