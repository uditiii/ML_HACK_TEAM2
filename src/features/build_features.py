"""Candidate-level feature engineering for PathSnap (Person C).

LEAKAGE RULE: features use ONLY the cleaned GPS track and the road network
(nodes, arcs). They never read the .route file, `label`, `true_arc` or
`has_true_candidate` -- those encode the answer we are trying to predict, so
using them would be label leakage (the model would "see" the answer in training
and fail on the evaluator trips where it does not exist).

How that is enforced in code:
  * add_features() only *selects* INPUT_COLS from the candidate frame before any
    computation, so label columns are physically never read.
  * tests/test_build_features.py::test_no_label_leakage shuffles/drops the label
    columns and checks that every feature is bit-for-bit identical.

DISTANCES (locked team decision): every metre value here (arc_length_m, speed_mps)
comes from the equirectangular projection around the trip centre
    x = R*(lon-lon0)*cos(lat0),  y = R*(lat-lat0),  R = 6,371,008.8 m
with (lon0, lat0) = mean of the trip's CLEANED GPS points -- the same projection as dist_m.
We use our own copy of that formula (B's geo.make_projector returns a tuple) and PROVE it
matches B: add_features compares our x, y with the x, y stored in clean_tracks and raises
if they differ by more than 5 cm. A second guard compares projected track steps against
great-circle steps (catches radians/degrees mix-ups).

Input assumptions:
  cand_df     one trip; columns trip_id, obs_idx, arc_id, dist_m, proj_t
              (extra columns such as label/true_arc are passed through untouched)
  trip        object or dict with .nodes (N x 2 or DataFrame lon, lat) and .arcs
              (M x 2 or DataFrame from_node, to_node). arc_id == row index in arcs.
              A "route" entry, if present, is never accessed.
  clean_track ONE trip's rows of data/processed/clean_tracks_{partition}.parquet:
              obs_idx, lon, lat, t (seconds), x, y, dt, segment_id. (If obs_idx is
              absent, the row position is the obs_idx: obs_idx = row of the CLEANED track.)
              The FULL cleaned track must be passed, even for train trips whose rows
              are subsampled to every 5th point, so speed/bearing use the 1 Hz data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

EARTH_R_M = 6_371_008.8
OBS_KEY = ["trip_id", "obs_idx"]
INPUT_COLS = ["trip_id", "obs_idx", "arc_id", "dist_m", "proj_t"]
FORBIDDEN_COLS = ("label", "true_arc", "has_true_candidate", "route")

SPEED_MIN_MPS = 1.0          # below this the vehicle is "stationary": bearing is noise
NEUTRAL_HEADING_DEG = 90.0   # heading_diff_deg value when heading is meaningless
PROJECTION_TOLERANCE = 0.02  # projected vs great-circle step length must agree within 2 %

# Every model-input column this module guarantees (selection happens later).
ALL_FEATURES = [
    "dist_m",
    "dist_rank",
    "dist_minus_min",
    "n_candidates",
    "arc_length_m",
    "gps_bearing",
    "arc_bearing",
    "heading_diff_deg",
    "heading_rank",
    "speed_mps",
    "heading_valid",
    "proj_t",
]

LON_ALIASES = ("lon", "lng", "longitude")
LAT_ALIASES = ("lat", "latitude")
TIME_ALIASES = ("t", "time", "timestamp", "ts", "time_s")


# --------------------------------------------------------------------------- #
# Projection (locked team decision)
# --------------------------------------------------------------------------- #
def _local_make_projector(lon0_deg: float, lat0_deg: float):
    """Same formula as the contract's geo.make_projector: degrees in, metres out."""
    lon0, lat0 = np.radians(lon0_deg), np.radians(lat0_deg)
    c = np.cos(lat0)

    def project(lon, lat):
        lon = np.radians(np.asarray(lon, dtype=float))
        lat = np.radians(np.asarray(lat, dtype=float))
        return EARTH_R_M * (lon - lon0) * c, EARTH_R_M * (lat - lat0)

    return project


# NOTE: Person B's geo.make_projector returns a TUPLE (to_xy, to_lonlat), so it is deliberately not
# used here. Our own one-function copy of the locked formula is used instead, and
# check_projection_matches_clean_track() proves it equals B's projection (clean_tracks x, y).
make_projector = _local_make_projector
PROJECTOR_SOURCE = "local copy of the locked formula (verified against clean_tracks x, y)"


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance (m). Used ONLY as a sanity guard for the projection."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_from_xy(dx, dy) -> np.ndarray:
    """Compass bearing in [0, 360) of a displacement (dx = east, dy = north); 0 = north, 90 = east."""
    return (np.degrees(np.arctan2(np.asarray(dx, dtype=float), np.asarray(dy, dtype=float))) + 360.0) % 360.0


def angle_diff_deg(a, b) -> np.ndarray:
    """Smallest absolute angle between two bearings, in [0, 180]."""
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


# --------------------------------------------------------------------------- #
# Input coercion
# --------------------------------------------------------------------------- #
def _get(trip, name):
    if isinstance(trip, dict):
        return trip[name]
    return getattr(trip, name)


def _pick(df: pd.DataFrame, aliases, what: str) -> np.ndarray:
    for a in aliases:
        if a in df.columns:
            return df[a].to_numpy(dtype=float)
    raise KeyError(f"clean_track has no {what} column (looked for {aliases}); columns={list(df.columns)}")


def _nodes_array(nodes) -> np.ndarray:
    """-> (N, 2) float array of [lon, lat]; row position = node_id."""
    if isinstance(nodes, pd.DataFrame):
        if {"lon", "lat"} <= set(nodes.columns):
            nodes = nodes[["lon", "lat"]]
        nodes = nodes.iloc[:, :2].to_numpy()
    arr = np.asarray(nodes, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"nodes must be N x 2 (lon, lat); got shape {arr.shape}")
    if not np.isfinite(arr[:, :2]).all():
        raise ValueError("nodes contain NaN/inf")
    return arr[:, :2]


def _arcs_array(arcs) -> np.ndarray:
    """-> (M, 2) int array of [from_node, to_node]; row position = arc_id."""
    if isinstance(arcs, pd.DataFrame):
        if {"from_node", "to_node"} <= set(arcs.columns):
            arcs = arcs[["from_node", "to_node"]]
        arcs = arcs.iloc[:, :2].to_numpy()
    arr = np.asarray(arcs)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"arcs must be M x 2 (from_node, to_node); got shape {arr.shape}")
    return arr[:, :2].astype(np.int64)


def _sorted_track(clean_track: pd.DataFrame) -> pd.DataFrame:
    return clean_track.sort_values("obs_idx", kind="stable") if "obs_idx" in clean_track.columns else clean_track


def _track_arrays(clean_track: pd.DataFrame):
    """-> (obs_idx, lon, lat, t, segment_id) in obs_idx order."""
    trk = _sorted_track(clean_track)
    obs = trk["obs_idx"].to_numpy() if "obs_idx" in trk.columns else np.arange(len(trk))
    if len(np.unique(obs)) != len(obs):
        raise ValueError("clean_track obs_idx is not unique")
    lon, lat, t = _pick(trk, LON_ALIASES, "longitude"), _pick(trk, LAT_ALIASES, "latitude"), _pick(trk, TIME_ALIASES, "timestamp")
    if not (np.isfinite(lon).all() and np.isfinite(lat).all() and np.isfinite(t).all()):
        raise ValueError("clean_track still has NaN/inf in lon/lat/t -- clean it first")
    if "segment_id" in trk.columns:
        seg = trk["segment_id"].to_numpy()
    else:
        warnings.warn("clean_track has no segment_id; treating the whole track as one segment", stacklevel=3)
        seg = np.zeros(len(trk), dtype=int)
    return obs, lon, lat, t, seg


def trip_projector(clean_track: pd.DataFrame):
    """Projector centred on the mean lon/lat of the trip's CLEANED GPS points (same centre as Person B)."""
    _, lon, lat, _, _ = _track_arrays(clean_track)
    return make_projector(float(lon.mean()), float(lat.mean()))


def check_projection_matches_clean_track(clean_track: pd.DataFrame, projector, tol_m: float = 0.05) -> None:
    """If the cleaned track carries B's x, y (metres), our projector must reproduce them.

    This proves arc_length_m / speed_mps use exactly the same projection as dist_m.
    Skipped (silently) when the track has no x, y columns.
    """
    if not {"x", "y"} <= set(clean_track.columns):
        return
    trk = _sorted_track(clean_track)
    x, y = projector(trk["lon"].to_numpy(float), trk["lat"].to_numpy(float))
    err = max(float(np.max(np.abs(np.asarray(x) - trk["x"].to_numpy(float)))),
              float(np.max(np.abs(np.asarray(y) - trk["y"].to_numpy(float)))))
    if err > tol_m:
        raise ValueError(f"our projection differs from clean_tracks x,y by up to {err:.3f} m -- "
                         "different projection centre or formula; arc_length_m would not match dist_m")


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #
def neighbour_indices(segment_id) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the 'previous' and 'next' point used for centred differences.

    Interior points: (i-1, i+1). First/last point of a segment: one-sided, i.e.
    prev == i (first) or next == i (last). Never crosses a segment_id boundary.
    A single-point segment gets (i, i): no motion information.
    """
    seg = np.asarray(segment_id)
    n = len(seg)
    idx = np.arange(n)
    prev, nxt = idx.copy(), idx.copy()
    if n > 1:
        same = seg[1:] == seg[:-1]          # same[k]: points k and k+1 share a segment
        prev[1:] = np.where(same, idx[1:] - 1, idx[1:])
        nxt[:-1] = np.where(same, idx[:-1] + 1, idx[:-1])
    return prev, nxt


def point_kinematics(clean_track: pd.DataFrame, projector) -> pd.DataFrame:
    """Per-GPS-point gps_bearing, speed_mps, heading_valid, indexed by obs_idx (metres via `projector`)."""
    obs, lon, lat, t, seg = _track_arrays(clean_track)
    x, y = projector(lon, lat)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    prev, nxt = neighbour_indices(seg)
    dx, dy = x[nxt] - x[prev], y[nxt] - y[prev]
    dist = np.hypot(dx, dy)

    # guard: projected step lengths must agree with great-circle lengths (catches unit mix-ups)
    gc = haversine_m(lon[prev], lat[prev], lon[nxt], lat[nxt])
    big = gc > 1.0
    if big.any():
        ratio = np.median(dist[big] / gc[big])
        if abs(ratio - 1.0) > PROJECTION_TOLERANCE:
            raise ValueError(f"projector looks wrong: projected/great-circle step ratio = {ratio:.3f} (expected ~1)")

    dt = t[nxt] - t[prev]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, dist / dt, 0.0)       # no time span -> treat as not moving
    bearing = bearing_from_xy(dx, dy)
    bearing = np.where(np.isfinite(bearing), bearing, 0.0)
    valid = (speed >= SPEED_MIN_MPS).astype(np.int8)
    return pd.DataFrame(
        {"gps_bearing": bearing, "speed_mps": speed, "heading_valid": valid},
        index=pd.Index(obs, name="obs_idx"),
    )


def arc_geometry(nodes, arcs, projector) -> pd.DataFrame:
    """Per-arc arc_length_m and arc_bearing (from_node -> to_node), indexed by arc_id (metres via `projector`)."""
    nd, ac = _nodes_array(nodes), _arcs_array(arcs)
    if ac.size and (ac.min() < 0 or ac.max() >= len(nd)):
        raise ValueError("arcs reference a node id outside the nodes table")
    nx, ny = projector(nd[:, 0], nd[:, 1])
    nx, ny = np.asarray(nx, dtype=float), np.asarray(ny, dtype=float)
    dx, dy = nx[ac[:, 1]] - nx[ac[:, 0]], ny[ac[:, 1]] - ny[ac[:, 0]]
    return pd.DataFrame(
        {"arc_length_m": np.hypot(dx, dy), "arc_bearing": bearing_from_xy(dx, dy)},
        index=pd.RangeIndex(len(ac), name="arc_id"),
    )


def add_obs_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """dist_rank, dist_minus_min, n_candidates -- computed within each (trip_id, obs_idx)."""
    g = df.groupby(OBS_KEY, sort=False)["dist_m"]
    return pd.DataFrame(
        {
            "dist_rank": g.rank(method="min").to_numpy().astype(np.int32),
            "dist_minus_min": (df["dist_m"] - g.transform("min")).to_numpy(),
            "n_candidates": g.transform("size").to_numpy().astype(np.int32),
        },
        index=df.index,
    )


def heading_difference(gps_bearing, arc_bearing, heading_valid) -> np.ndarray:
    """Smallest angle between motion and arc direction; neutral (90) when not moving."""
    diff = angle_diff_deg(gps_bearing, arc_bearing)
    return np.where(np.asarray(heading_valid) == 1, diff, NEUTRAL_HEADING_DEG)


def heading_rank(df: pd.DataFrame) -> np.ndarray:
    """Rank of heading_diff_deg among the candidates of the same GPS point (1 = best aligned).

    Like dist_rank, this is relative: "is this the arc that points most nearly the way we move,
    compared with the other arcs here?" Ties share the lowest rank (method "min"); when the
    vehicle is stationary every candidate has the neutral 90 deg, so all get rank 1.
    """
    return df.groupby(OBS_KEY, sort=False)["heading_diff_deg"].rank(method="min").to_numpy().astype(np.int32)


def assert_feature_list_leak_free(features) -> None:
    bad = [f for f in features if f in FORBIDDEN_COLS]
    assert not bad, f"label/route columns in feature list: {bad}"


def assert_features_clean(df: pd.DataFrame, features=ALL_FEATURES) -> None:
    """NaN / range / dtype sanity for the final feature columns."""
    missing = [f for f in features if f not in df.columns]
    assert not missing, f"missing feature columns: {missing}"
    vals = df[list(features)].to_numpy(dtype=float)
    assert np.isfinite(vals).all(), "NaN/inf in features: " + str(
        df[list(features)].columns[~np.isfinite(vals).all(axis=0)].tolist()
    )
    if len(df):
        assert df["dist_rank"].min() >= 1 and df["n_candidates"].min() >= 1
        assert (df["dist_minus_min"] >= -1e-9).all()
        assert df["heading_diff_deg"].between(0, 180).all()
        assert df["heading_rank"].min() >= 1 and (df["heading_rank"] <= df["n_candidates"]).all()
        assert set(df["heading_valid"].unique()) <= {0, 1}
        assert (df.loc[df["heading_valid"] == 0, "heading_diff_deg"] == NEUTRAL_HEADING_DEG).all()
        assert df["arc_bearing"].between(0, 360).all() and df["gps_bearing"].between(0, 360).all()
        assert (df["arc_length_m"] >= 0).all() and (df["speed_mps"] >= 0).all()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def add_features(cand_df: pd.DataFrame, trip, clean_track: pd.DataFrame, projector=None) -> pd.DataFrame:
    """Return a copy of cand_df (one trip) with all ALL_FEATURES columns present.

    projector: optional override; default = trip-centred projector from clean_track.
    """
    missing = [c for c in INPUT_COLS if c not in cand_df.columns]
    if missing:
        raise KeyError(f"candidate frame is missing {missing}")

    # Only these columns are ever read. Label columns cannot influence any feature.
    base = cand_df[INPUT_COLS]
    if base[["dist_m", "proj_t", "arc_id", "obs_idx"]].isna().any().any():
        raise ValueError("NaN in dist_m/proj_t/arc_id/obs_idx of the candidate table -- fix candidate generation")
    if base["trip_id"].nunique() > 1:
        raise ValueError("add_features works on ONE trip at a time (obs_idx is mapped to that trip's track)")

    proj = projector if projector is not None else trip_projector(clean_track)
    check_projection_matches_clean_track(clean_track, proj)
    out = cand_df.copy()

    # 1) within-observation features
    out[["dist_rank", "dist_minus_min", "n_candidates"]] = add_obs_rank_features(base)

    # 2) arc geometry (metres), looked up by arc_id
    geo = arc_geometry(_get(trip, "nodes"), _get(trip, "arcs"), proj)
    arc_id = base["arc_id"].to_numpy().astype(np.int64)
    if len(arc_id) and (arc_id.min() < 0 or arc_id.max() >= len(geo)):
        raise ValueError("candidate arc_id outside the arcs table")
    out["arc_length_m"] = geo["arc_length_m"].to_numpy()[arc_id]
    out["arc_bearing"] = geo["arc_bearing"].to_numpy()[arc_id]

    # 3) per-GPS-point motion from the FULL cleaned track, looked up by obs_idx
    kin = point_kinematics(clean_track, proj)
    pos = kin.index.get_indexer(base["obs_idx"].to_numpy())
    if (pos < 0).any():
        raise ValueError("candidate obs_idx not present in clean_track")
    out["gps_bearing"] = kin["gps_bearing"].to_numpy()[pos]
    out["speed_mps"] = kin["speed_mps"].to_numpy()[pos]
    out["heading_valid"] = kin["heading_valid"].to_numpy()[pos]

    # 4) heading agreement (neutral 90 when the vehicle is (nearly) stationary)
    out["heading_diff_deg"] = heading_difference(out["gps_bearing"], out["arc_bearing"], out["heading_valid"])
    out["heading_rank"] = heading_rank(out)                     # 1 = best aligned arc of this GPS point

    assert_features_clean(out)
    return out
