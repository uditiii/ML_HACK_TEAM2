"""Task 3: per-point reference labels. LABEL CREATION ONLY - never used for features or at inference.

The .route file lists the arcs driven, in order, but not WHEN each arc was driven, and it has
far fewer rows than the track. So we must decide which route arc each GPS point belongs to.

v1 (nearest route arc): for each point, the closest arc among the route's arcs, if <= 50 m.
   Limitation: ignores order, so at junctions, U-turns or parallel route arcs it can jump
   between arcs (including backwards along the route).
v2 (monotone DP): each point picks a route POSITION; the sequence of positions should move
   forward. Cost = sum of distances + 30 m per backward step + 15 m per forward jump > 10
   positions. Solved exactly by dynamic programming (Viterbi-style) with backtracking.
   Limitation: only the 8 closest route positions per point are considered, and the penalty
   values are hand-chosen, not learned.
Both: points with no route arc within 50 m get -1 (unlabelled). The route itself is treated as
reference, not perfect ground truth.
"""
import numpy as np

from config import RADIUS_M
from src.data.geo import point_segment_distance

LABEL_RADIUS_M = RADIUS_M
MAX_STATES = 8
BACKWARD_PENALTY = 30.0
JUMP_PENALTY = 15.0
JUMP_SIZE = 10
CHUNK = 1000
# Distances are rounded to 1 micrometre before choosing, so exact geometric ties (an arc and
# its reverse, both in the route, are equally close) are broken the same way on every machine:
# lowest arc_id (v1) / lowest route position (v2). Without this, last-digit floating-point
# differences between NumPy versions decided the winner.
TIE_DECIMALS = 6


def route_array(route):
    """Route as a 1-D int64 array of arc_ids. Accepts None (no .route file), a DataFrame with
    an arc_id column (Person A's loader), a Series, a list or an array."""
    if route is None:
        return np.empty(0, np.int64)
    if hasattr(route, "columns"):
        route = route["arc_id"]
    return np.asarray(route, dtype=np.int64).reshape(-1)


def _route_arc_coords(arc_geometry, arcs):
    pos = arc_geometry.index.get_indexer(arcs)
    if (pos < 0).any():
        raise ValueError("route contains arc_ids that are not in the trip's .arcs")
    g = arc_geometry.iloc[pos]
    return [g[c].to_numpy() for c in ["ax", "ay", "bx", "by"]]


def _dist_chunk(px, py, coords):
    """(n_points, n_arcs) matrix of point-to-segment distances, via broadcasting."""
    ax, ay, bx, by = coords
    d, _ = point_segment_distance(px[:, None], py[:, None], ax[None, :], ay[None, :],
                                  bx[None, :], by[None, :])
    return d


def _label_v1(px, py, arc_geometry, route):
    uarcs = np.unique(route)
    coords = _route_arc_coords(arc_geometry, uarcs)
    first_pos = {a: i for i, a in reversed(list(enumerate(route)))}  # first occurrence
    true_arc = np.full(len(px), -1, np.int64)
    for s in range(0, len(px), CHUNK):
        d = np.round(_dist_chunk(px[s:s + CHUNK], py[s:s + CHUNK], coords), TIE_DECIMALS)
        j = d.argmin(axis=1)          # first minimum = lowest arc_id (uarcs is sorted)
        ok = d[np.arange(len(j)), j] <= LABEL_RADIUS_M
        true_arc[s:s + CHUNK][ok] = uarcs[j[ok]]
    route_pos = np.array([first_pos[a] if a >= 0 else -1 for a in true_arc], np.int64)
    return true_arc, route_pos


def _label_v2(px, py, arc_geometry, route):
    uarcs, inv = np.unique(route, return_inverse=True)
    coords = _route_arc_coords(arc_geometry, uarcs)
    n, K = len(px), min(MAX_STATES, len(route))
    st_pos = np.full((n, K), -1, np.int64)       # candidate route positions per point
    st_cost = np.full((n, K), np.inf)            # their emission cost (distance in metres)
    for s in range(0, n, CHUNK):
        d = np.round(_dist_chunk(px[s:s + CHUNK], py[s:s + CHUNK], coords), TIE_DECIMALS)[:, inv]
        # K nearest route positions; stable sort -> among equal distances the lowest position wins
        idx = np.argsort(d, axis=1, kind="stable")[:, :K]
        dk = np.take_along_axis(d, idx, axis=1)
        ok = dk <= LABEL_RADIUS_M
        st_pos[s:s + CHUNK] = np.where(ok, idx, -1)
        st_cost[s:s + CHUNK] = np.where(ok, dk, np.inf)

    valid = np.flatnonzero(np.isfinite(st_cost).any(axis=1))   # points with >= 1 route arc in range
    true_arc = np.full(n, -1, np.int64)
    route_pos = np.full(n, -1, np.int64)
    if len(valid) == 0:
        return true_arc, route_pos
    # forward pass: acc[k] = cheapest total cost of any labelling ending in state k
    acc = st_cost[valid[0]].copy()
    back = np.zeros((len(valid), K), np.int64)
    for m in range(1, len(valid)):
        p_prev, p_cur = st_pos[valid[m - 1]], st_pos[valid[m]]
        step = p_cur[None, :] - p_prev[:, None]                     # (K_prev, K_cur)
        trans = np.where(step < 0, BACKWARD_PENALTY, np.where(step > JUMP_SIZE, JUMP_PENALTY, 0.0))
        tot = acc[:, None] + trans
        best = tot.argmin(axis=0)
        back[m] = best
        acc = tot[best, np.arange(K)] + st_cost[valid[m]]
    # backtracking: follow the stored best predecessors from the cheapest final state
    k = int(acc.argmin())
    for m in range(len(valid) - 1, -1, -1):
        i = valid[m]
        route_pos[i] = st_pos[i, k]
        k = back[m, k]
    true_arc[valid] = route[route_pos[valid]]
    return true_arc, route_pos


def label_points(clean_track, arc_geometry, route, method="v1", return_positions=False):
    """true_arc for every cleaned GPS point (-1 = unlabelled). method: "v1" or "v2".

    clean_track x, y and arc_geometry must come from the same projector.
    With return_positions=True also returns the route position of each label
    (v1: first occurrence of the arc in the route; v2: the DP position).
    """
    px, py = clean_track["x"].to_numpy(), clean_track["y"].to_numpy()
    route = route_array(route)
    if len(route) == 0:
        true_arc, route_pos = np.full(len(px), -1, np.int64), np.full(len(px), -1, np.int64)
    elif method == "v1":
        true_arc, route_pos = _label_v1(px, py, arc_geometry, route)
    elif method == "v2":
        true_arc, route_pos = _label_v2(px, py, arc_geometry, route)
    else:
        raise ValueError(f"unknown method {method!r}")
    return (true_arc, route_pos) if return_positions else true_arc


def label_quality(clean_track, arc_geometry, true_arc, route_pos):
    """Mean distance from labelled points to their label arc, and the share of backward steps
    (route position decreases) between consecutive labelled points."""
    lab = true_arc >= 0
    out = {"n_points": int(len(true_arc)), "n_labelled": int(lab.sum()),
           "pct_unlabelled": float(100 * (~lab).mean()) if len(lab) else np.nan,
           "mean_label_dist_m": np.nan, "backward_step_share": np.nan}
    if lab.any():
        ax, ay, bx, by = _route_arc_coords(arc_geometry, true_arc[lab])
        d, _ = point_segment_distance(clean_track["x"].to_numpy()[lab], clean_track["y"].to_numpy()[lab],
                                      ax, ay, bx, by)
        out["mean_label_dist_m"] = float(d.mean())
        steps = np.diff(route_pos[lab])
        if len(steps):
            out["backward_step_share"] = float((steps < 0).mean())
    return out
