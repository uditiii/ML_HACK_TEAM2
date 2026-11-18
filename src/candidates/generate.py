"""Task 2: fast candidate search (every arc within RADIUS_M of a GPS point).

Why not a KD-tree on arc midpoints? A 400 m arc can pass 5 m from a point while its
midpoint is 200 m away, so it would be missed. Instead we put sample points along every
arc, at most SPACING_M apart, endpoints included.

Correctness argument (viva): if an arc is within r of point P, let Q be the closest point
on the arc. Some sample S lies within SPACING_M / 2 = 5 m of Q along the segment, so
|P - S| <= r + 5 < r + PAD_M. A ball query of radius r + PAD_M therefore finds at least one
sample of every true candidate. Then we compute the EXACT point-to-segment distance and
keep only dist <= r, so the extra radius never adds wrong rows.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import RADIUS_M
from src.data.geo import point_segment_distance

SPACING_M = 10.0
PAD_M = 10.0
CAND_COLUMNS = ["trip_id", "obs_idx", "arc_id", "dist_m", "proj_t"]


def prefilter_arcs(geom, px, py, margin):
    """Keep arcs whose bounding box touches the GPS bounding box grown by `margin` metres.

    An arc that is within `margin` of some point must overlap this box, so this never
    removes a real candidate; it only shrinks big networks before densifying.
    """
    a = geom[["ax", "ay", "bx", "by"]].to_numpy()
    finite = np.isfinite(a).all(axis=1)
    xmin, xmax = px.min() - margin, px.max() + margin
    ymin, ymax = py.min() - margin, py.max() + margin
    keep = (finite
            & (np.maximum(a[:, 0], a[:, 2]) >= xmin) & (np.minimum(a[:, 0], a[:, 2]) <= xmax)
            & (np.maximum(a[:, 1], a[:, 3]) >= ymin) & (np.minimum(a[:, 1], a[:, 3]) <= ymax))
    return geom[keep]


def densify_arcs(geom, spacing=SPACING_M):
    """Sample points along every arc, <= `spacing` metres apart, both endpoints included.

    An arc of length L gets n = max(1, ceil(L / spacing)) equal pieces, i.e. n + 1 samples
    at fractions 0, 1/n, ..., 1. Fully vectorised: np.repeat expands per-arc values to
    per-sample values. Returns (sx, sy, sample_arc_id).
    """
    arc_ids = geom.index.to_numpy()
    ax, ay, bx, by, L = (geom[c].to_numpy() for c in ["ax", "ay", "bx", "by", "length_m"])
    n_seg = np.maximum(1, np.ceil(L / spacing)).astype(np.int64)
    counts = n_seg + 1
    owner = np.repeat(np.arange(len(arc_ids)), counts)            # which arc each sample belongs to
    first = np.cumsum(counts) - counts                              # index of each arc's first sample
    frac = (np.arange(counts.sum()) - first[owner]) / n_seg[owner]  # 0 ... 1 along the arc
    sx = ax[owner] + frac * (bx - ax)[owner]
    sy = ay[owner] + frac * (by - ay)[owner]
    return sx, sy, arc_ids[owner]


def generate_candidates(trip_id, px, py, obs_idx, geom, radius=RADIUS_M):
    """All (obs, arc) pairs with exact point-to-segment distance <= radius.

    px, py: projected GPS points (metres); obs_idx: their cleaned-track row numbers;
    geom: arc_geometry() output. Returns DataFrame trip_id, obs_idx, arc_id, dist_m, proj_t
    (proj_t = clamped position of the closest point along the arc: 0 = from node, 1 = to node).
    Points with no arc within radius simply have no rows.
    """
    px, py = np.asarray(px, np.float64), np.asarray(py, np.float64)
    obs_idx = np.asarray(obs_idx, np.int64)
    empty = pd.DataFrame({c: pd.Series(dtype=d) for c, d in
                          zip(CAND_COLUMNS, [object, np.int64, np.int64, np.float64, np.float64])})
    if len(px) == 0:
        return empty
    g = prefilter_arcs(geom, px, py, radius + PAD_M)
    if len(g) == 0:
        return empty
    sx, sy, s_arc = densify_arcs(g)
    tree = cKDTree(np.column_stack([sx, sy]))
    hits = tree.query_ball_point(np.column_stack([px, py]), r=radius + PAD_M, workers=-1)

    lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    if lens.sum() == 0:
        return empty
    pt = np.repeat(np.arange(len(px)), lens)                      # point position of each hit
    arc = s_arc[np.concatenate([np.asarray(h, np.int64) for h in hits if len(h)])]
    # one row per unique (point, arc): encode the pair as one integer and np.unique it
    base = np.int64(arc.max()) + 1
    key = np.unique(pt * base + arc)
    pt, arc = key // base, key % base

    gpos = geom.index.get_indexer(arc)
    d, t = point_segment_distance(px[pt], py[pt], geom["ax"].to_numpy()[gpos],
                                  geom["ay"].to_numpy()[gpos], geom["bx"].to_numpy()[gpos],
                                  geom["by"].to_numpy()[gpos])
    keep = d <= radius
    out = pd.DataFrame({"trip_id": trip_id, "obs_idx": obs_idx[pt[keep]], "arc_id": arc[keep],
                        "dist_m": d[keep], "proj_t": t[keep]})
    # sort by obs, then distance rounded to 1 micrometre, then arc_id: an arc and its reverse are
    # equally far, so the rounding + arc_id tie-break gives the same row order on every machine
    order = np.lexsort((out["arc_id"].to_numpy(), np.round(out["dist_m"].to_numpy(), 6),
                        out["obs_idx"].to_numpy()))
    return out.iloc[order].reset_index(drop=True)


def brute_force_candidates(trip_id, px, py, obs_idx, geom, radius=RADIUS_M):
    """Slow reference: exact distance from each point to EVERY arc. Only used to test the fast version."""
    a = [geom[c].to_numpy() for c in ["ax", "ay", "bx", "by"]]
    rows = []
    for x, y, o in zip(px, py, obs_idx):
        d, t = point_segment_distance(x, y, *a)
        k = np.flatnonzero(d <= radius)
        rows.append(pd.DataFrame({"trip_id": trip_id, "obs_idx": o, "arc_id": geom.index.to_numpy()[k],
                                  "dist_m": d[k], "proj_t": t[k]}))
    return pd.concat(rows, ignore_index=True)
