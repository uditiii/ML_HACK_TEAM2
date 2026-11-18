"""One check per function in src/features/build_features.py.

Run from the repo root:  python tests/test_build_features.py      (or: pytest -q tests)
Uses only synthetic data (tests/standin_candidates.py) -- no real dataset needed.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.features import build_features as bf  # noqa: E402
from standin_candidates import make_trip  # noqa: E402

PROJ = bf._local_make_projector(5.0, 52.0)       # known formula, independent of Person B's module


# ---- _local_make_projector / make_projector ----------------------------------
def test_make_projector():
    x, y = PROJ(5.0, 52.0)
    assert abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6                     # centre -> (0, 0)
    _, y = PROJ(5.0, 52.001)
    assert abs(float(y) - 111.195) < 0.01                                    # 0.001 deg lat = R*pi/180/1000 m
    x, _ = PROJ(5.001, 52.0)
    assert abs(float(x) - 111.195 * np.cos(np.radians(52))) < 0.01           # lon shrinks by cos(lat0)
    to_xy = bf.make_projector(5.0, 52.0)
    assert callable(to_xy)                                                   # a function, NOT B's (to_xy, to_lonlat) tuple


# ---- haversine_m (guard only) ------------------------------------------------
def test_haversine_m():
    assert abs(bf.haversine_m(0, 0, 0, 0.001)[()] - 111.19) < 0.2
    assert abs(bf.haversine_m(0, 0, 1, 0)[()] - 111_195) < 50
    assert bf.haversine_m(5, 52, 5, 52)[()] == 0
    # projected distance near the centre agrees with great-circle distance within 0.1 %
    x, y = PROJ(np.array([5.0, 5.002]), np.array([52.0, 52.001]))
    assert abs(np.hypot(np.diff(x), np.diff(y))[0] / bf.haversine_m(5.0, 52.0, 5.002, 52.001)[()] - 1) < 1e-3


# ---- bearing_from_xy ---------------------------------------------------------
def test_bearing_from_xy():
    b = lambda dx, dy: float(bf.bearing_from_xy(dx, dy))
    assert b(0, 10) == 0 and b(10, 0) == 90 and b(0, -10) == 180 and b(-10, 0) == 270
    assert abs(b(10, 10) - 45) < 1e-9 and 0 <= b(-1, 1e-9) < 360               # always in [0, 360)
    assert b(0, 0) == 0                                                      # zero move -> 0 (gated by heading_valid)


# ---- angle_diff_deg ----------------------------------------------------------
def test_angle_diff_deg():
    got = bf.angle_diff_deg([350, 0, 90, 10, 359, 270], [10, 180, 270, 10, 1, 0])
    assert np.allclose(got, [20, 180, 180, 0, 2, 90])
    assert (got >= 0).all() and (got <= 180).all()


# ---- neighbour_indices -------------------------------------------------------
def test_neighbour_indices():
    prev, nxt = bf.neighbour_indices([0, 0, 0, 1, 1, 2])
    assert prev.tolist() == [0, 0, 1, 3, 3, 5]      # first-of-segment uses itself
    assert nxt.tolist() == [1, 2, 2, 4, 4, 5]       # last-of-segment uses itself; singleton -> (i, i)
    p1, n1 = bf.neighbour_indices([7])
    assert p1.tolist() == [0] and n1.tolist() == [0]


# ---- point_kinematics --------------------------------------------------------
def test_point_kinematics():
    # 6 points heading due east at 10 m/s, then a 100 s gap and a segment heading north, then a lone point
    dlon10 = 10 / (111_195 * np.cos(np.radians(52)))
    dlat10 = 10 / 111_195
    lon = [5 + i * dlon10 for i in range(6)] + [5.0005] * 4 + [5.001]
    lat = [52.0] * 6 + [52 + i * dlat10 for i in range(4)] + [52.0005]
    t = list(range(6)) + [105, 106, 107, 108] + [300]
    seg = [0] * 6 + [1] * 4 + [2]
    kin = bf.point_kinematics(pd.DataFrame({"lon": lon, "lat": lat, "t": t, "segment_id": seg}), PROJ)
    assert len(kin) == 11 and not kin.isna().any().any()
    assert np.allclose(kin["speed_mps"][:6], 10, atol=0.2) and np.allclose(kin["gps_bearing"][:6], 90, atol=0.5)
    # second segment: northbound, speed unaffected by the 99 s gap before it
    assert np.allclose(kin["speed_mps"][6:10], 10, atol=0.2)
    assert np.all((kin["gps_bearing"][6:10] < 0.5) | (kin["gps_bearing"][6:10] > 359.5))
    # lone point: no motion information -> invalid heading, speed 0
    assert kin["heading_valid"].iloc[10] == 0 and kin["speed_mps"].iloc[10] == 0
    assert (kin["heading_valid"].iloc[:10] == 1).all()
    # stationary: identical points -> invalid
    st = bf.point_kinematics(pd.DataFrame({"lon": [5.0] * 4, "lat": [52.0] * 4, "t": [0, 1, 2, 3], "segment_id": 0}), PROJ)
    assert (st["heading_valid"] == 0).all()
    # contract: with no obs_idx column the row position is the obs_idx
    assert kin.index.tolist() == list(range(11))


def test_projection_guard():
    """A projector that returns degrees/radians instead of metres must be rejected, not silently used."""
    trk = pd.DataFrame({"lon": 5 + np.arange(5) * 1e-4, "lat": 52.0, "t": np.arange(5.0), "segment_id": 0})
    try:
        bf.point_kinematics(trk, lambda lon, lat: (np.asarray(lon) - 5, np.asarray(lat) - 52))
        raise AssertionError("bad projector not caught")
    except ValueError as e:
        assert "projector looks wrong" in str(e)


# ---- arc_geometry ------------------------------------------------------------
def test_arc_geometry():
    nodes = pd.DataFrame({"lon": [5.0, 5.001, 5.001], "lat": [52.0, 52.0, 52.001]})
    arcs = pd.DataFrame({"from_node": [0, 1, 1], "to_node": [1, 0, 2]})
    g = bf.arc_geometry(nodes, arcs, PROJ)
    assert abs(g["arc_bearing"].iloc[0] - 90) < 0.1 and abs(g["arc_bearing"].iloc[1] - 270) < 0.1
    assert abs(g["arc_length_m"].iloc[0] - g["arc_length_m"].iloc[1]) < 1e-6      # reverse arc, same length
    assert abs(bf.angle_diff_deg(g["arc_bearing"].iloc[0], g["arc_bearing"].iloc[1])[()] - 180) < 0.2
    assert abs(g["arc_bearing"].iloc[2] - 0) < 0.1 or abs(g["arc_bearing"].iloc[2] - 360) < 0.1
    assert 66 < g["arc_length_m"].iloc[0] < 70 and 110 < g["arc_length_m"].iloc[2] < 112
    try:
        bf.arc_geometry(nodes, pd.DataFrame({"from_node": [0], "to_node": [9]}), PROJ)
        raise AssertionError("expected ValueError for out-of-range node id")
    except ValueError:
        pass


# ---- add_obs_rank_features ---------------------------------------------------
def test_add_obs_rank_features():
    df = pd.DataFrame(
        {"trip_id": "t", "obs_idx": [0, 0, 0, 1, 1], "dist_m": [5.0, 5.0, 9.0, 3.0, 8.0]}
    )
    r = bf.add_obs_rank_features(df)
    assert r["dist_rank"].tolist() == [1, 1, 3, 1, 2]            # ties share the lowest rank (method="min")
    assert r["dist_minus_min"].tolist() == [0.0, 0.0, 4.0, 0.0, 5.0]
    assert r["n_candidates"].tolist() == [3, 3, 3, 2, 2]


# ---- heading_difference ------------------------------------------------------
def test_heading_difference():
    h = bf.heading_difference([350, 90, 10], [10, 270, 20], [1, 1, 0])
    assert np.allclose(h, [20, 180, 90])                        # invalid heading -> neutral 90


# ---- assert_features_clean / assert_feature_list_leak_free ---------------------
def test_assertions_catch_problems():
    trip, trk, cand = make_trip(seed=1)
    ok = bf.add_features(cand, trip, trk)
    bad = ok.copy()
    bad.loc[bad.index[0], "speed_mps"] = np.nan
    try:
        bf.assert_features_clean(bad)
        raise AssertionError("NaN not caught")
    except AssertionError as e:
        assert "NaN" in str(e)
    try:
        bf.assert_feature_list_leak_free(["dist_m", "label"])
        raise AssertionError("label not caught")
    except AssertionError as e:
        assert "label" in str(e)
    bf.assert_feature_list_leak_free(bf.ALL_FEATURES)


# ---- trip_projector ----------------------------------------------------------
def test_trip_projector():
    trk = pd.DataFrame({"lon": [5.0, 5.002], "lat": [52.0, 52.002], "t": [0.0, 1.0], "segment_id": 0})
    x, y = bf.trip_projector(trk)(trk["lon"], trk["lat"])
    assert abs(x[0] + x[1]) < 1e-6 and abs(y[0] + y[1]) < 1e-6                # mean of the GPS points maps to (0, 0)


# ---- check_projection_matches_clean_track ------------------------------------
def test_check_projection_matches_clean_track():
    trk = pd.DataFrame({"obs_idx": [0, 1, 2], "lon": [5.0, 5.001, 5.002], "lat": [52.0, 52.0005, 52.001],
                        "t": [0.0, 1.0, 2.0], "segment_id": 0})
    proj = bf.trip_projector(trk)
    x, y = proj(trk["lon"], trk["lat"])
    good = trk.assign(x=x, y=y)
    bf.check_projection_matches_clean_track(good, proj)                       # agrees with B's x, y -> silent
    bf.check_projection_matches_clean_track(trk, proj)                        # no x, y columns -> skipped
    for bad in (good.assign(x=good["x"] + 1.0),                               # shifted centre / different formula
                good.assign(y=good["y"] * 1.01)):
        try:
            bf.check_projection_matches_clean_track(bad, proj)
            raise AssertionError("projection mismatch not caught")
        except ValueError as e:
            assert "differs from clean_tracks" in str(e)
    shuffled = good.iloc[[2, 0, 1]]                                           # row order must not matter (sorted by obs_idx)
    bf.check_projection_matches_clean_track(shuffled, proj)


# ---- add_features (end to end) ----------------------------------------------
def test_add_features_end_to_end():
    trip, trk, cand = make_trip(seed=2)
    out = bf.add_features(cand, trip, trk)
    assert len(out) == len(cand) and set(bf.ALL_FEATURES) <= set(out.columns)
    assert not out[bf.ALL_FEATURES].isna().any().any()
    for c in ["dist_m", "proj_t", "label", "true_arc", "has_true_candidate"]:      # inputs untouched, labels passed through
        assert out[c].equals(cand[c])
    assert (out.loc[out["dist_rank"] == 1].groupby("obs_idx")["dist_m"].min() ==
            out.groupby("obs_idx")["dist_m"].min()).all()
    # heading separates an arc from its reverse: for moving points the true arc agrees with the motion
    mv = out[(out["heading_valid"] == 1)]
    assert mv.loc[mv["label"] == 1, "heading_diff_deg"].median() < 30
    assert (mv.loc[mv["label"] == 0, "heading_diff_deg"] > 150).mean() > 0.05
    st = out[out["heading_valid"] == 0]
    assert len(st) > 0 and (st["heading_diff_deg"] == 90).all()
    # arc lengths are in metres (synthetic grid spacing is ~68 m east-west, ~78 m north-south)
    assert set(np.round(out["arc_length_m"].unique(), 0)) <= {68.0, 69.0, 78.0}
    # explicit failure modes
    for col in ["dist_m", "proj_t"]:
        broken = cand.copy()
        broken.loc[broken.index[0], col] = np.nan
        try:
            bf.add_features(broken, trip, trk)
            raise AssertionError(f"NaN in {col} not rejected")
        except ValueError:
            pass


def test_add_features_with_b_style_track():
    """Track shaped like clean_tracks_*.parquet (obs_idx, x, y, dt, shuffled row order) gives identical features."""
    trip, trk, cand = make_trip(seed=5)
    plain = bf.add_features(cand, trip, trk)
    x, y = bf.trip_projector(trk)(trk["lon"], trk["lat"])
    b_style = trk.assign(trip_id="trip_000", x=x, y=y, dt=trk["t"].diff().fillna(0.0)).sample(frac=1.0, random_state=0)
    out = bf.add_features(cand, trip, b_style)
    pd.testing.assert_frame_equal(plain[bf.ALL_FEATURES], out[bf.ALL_FEATURES])
    wrong = b_style.assign(x=b_style["x"] + 3.0)
    try:
        bf.add_features(cand, trip, wrong)
        raise AssertionError("mismatching x,y not caught")
    except ValueError:
        pass


def test_subsampled_rows_use_full_track():
    """Train rows keep every 5th point, but speed/bearing must still come from the full 1 Hz track."""
    trip, trk, cand = make_trip(seed=4)
    full = bf.add_features(cand, trip, trk)
    sub_c = cand[cand["obs_idx"] % 5 == 0]
    sub = bf.add_features(sub_c, trip, trk)
    assert len(sub) < len(full)
    pd.testing.assert_frame_equal(sub[bf.ALL_FEATURES], full.loc[sub.index, bf.ALL_FEATURES])


# ---- leakage -----------------------------------------------------------------
def test_no_label_leakage():
    trip, trk, cand = make_trip(seed=3)
    a = bf.add_features(cand, trip, trk)
    no_labels = cand.drop(columns=["label", "true_arc", "has_true_candidate"])
    b = bf.add_features(no_labels, trip, trk)
    shuffled = cand.copy()
    rng = np.random.default_rng(0)
    for c in ["label", "true_arc", "has_true_candidate"]:
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
    c_ = bf.add_features(shuffled, trip, trk)
    trip_no_route = {"nodes": trip.nodes, "arcs": trip.arcs}                    # exactly what build_tables passes
    d = bf.add_features(cand, trip_no_route, trk)
    for other in (b, c_, d):
        pd.testing.assert_frame_equal(a[bf.ALL_FEATURES], other[bf.ALL_FEATURES])


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n{len(tests)} checks passed  (projector: {bf.PROJECTOR_SOURCE})")
