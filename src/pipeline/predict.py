"""End-to-end prediction for ONE trip (Person C): the function the API calls.

predict_trip(trip_dir_root, trip_id, model, meta, use_hmm=True) runs exactly the training-time steps:
  1. read <trip_dir_root>/<trip_id>/<trip_id>.{track,nodes,arcs}            (same parsing as the loader)
  2. clean_track (Person A)                                                 -> obs_idx = cleaned row number
  3. project points and arcs with ONE projector centred on the trip         (same as candidate generation)
  4. candidates within 50 m for EVERY cleaned point (Person B)
  5. features, built EXACTLY as the model was trained (the route is never passed):
     the 5 quick features (D's final model) -> add_quick_features on the candidate table;
     any other feature list                 -> Person C's add_features (track + road network)
  6. model.predict_proba on meta["features"]                                -> P(this arc is the true one)
  7. classifier-only answer: per point, the highest-probability candidate
  8. HMM + Viterbi (Person B)                                               -> one coherent route
  9. ONLY IF a .route file exists: reference labels (label_points, the same "v2" labels used for training)
     are computed here, server-side, to score the prediction. They are never returned and never used
     by steps 2-8.
The returned dict holds predictions and scores only: no reference arcs, no route.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import RADIUS_M
from src.candidates.generate import generate_candidates
from src.candidates.geometry import arc_geometry, trip_projector
from src.candidates.labels import label_points
from src.data.preprocess import clean_track
from src.features.build_features import add_features
from src.features.quick_features import FEATURES as QUICK_FEATURES
from src.features.quick_features import add_quick_features
from src.hmm.viterbi import DEFAULT_PARAMS, classifier_only, decode_trip, route_continuity, top1_counts

ROOT = Path(__file__).resolve().parents[2]
LABEL_METHOD = "v2"          # must match the labels the model was trained on (scripts/build_candidates.py)
HMM_PARAMS_FILE = ROOT / "results" / "hmm_chosen_params.json"
CONTRACT_KEYS = ("trip_id", "n_points", "gps", "predicted_arcs", "predicted_route_coords", "metrics")


def _read(path, names, dtype):
    """Headerless whitespace-separated file -> DataFrame; row number = id (same as src/data/loader.py)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)
    df.index = pd.RangeIndex(len(df))
    return df


def load_trip_dir(trip_dir_root, trip_id):
    """Load one trip from ANY folder that contains <trip_id>/ (student_test now, the evaluator folder at the demo).

    .route is optional: without it the trip can still be predicted, only accuracy cannot be computed.
    """
    d = Path(trip_dir_root) / trip_id
    f = lambda ext: d / f"{trip_id}.{ext}"  # noqa: E731
    for ext in ("track", "nodes", "arcs"):
        if not f(ext).exists():
            raise FileNotFoundError(f"missing {f(ext)}")
    track = _read(f("track"), ["lon", "lat", "t"], {"lon": np.float64, "lat": np.float64, "t": np.float64})
    nodes = _read(f("nodes"), ["lon", "lat"], {"lon": np.float64, "lat": np.float64})
    arcs = _read(f("arcs"), ["from_node", "to_node"], {"from_node": np.int64, "to_node": np.int64})
    nodes.index.name, arcs.index.name = "node_id", "arc_id"
    route = None
    if f("route").exists() and f("route").stat().st_size > 0:
        route = _read(f("route"), ["arc_id"], {"arc_id": np.int64})
    return {"trip_id": trip_id, "track": track, "nodes": nodes, "arcs": arcs, "route": route}


def load_hmm_params(path=HMM_PARAMS_FILE):
    """HMM parameters chosen on TRAIN trips (Person B's search); DEFAULT_PARAMS if the file is missing."""
    p = Path(path)
    if p.exists():
        return {**DEFAULT_PARAMS, **json.loads(p.read_text())["params"]}
    return dict(DEFAULT_PARAMS)


def _num(x, ndigits=4):
    """JSON-safe number: NaN/None -> None, otherwise rounded float."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), ndigits)


def _route_polylines(pred_arcs, nodes, arcs):
    """One [[lat, lon], [lat, lon]] polyline per distinct consecutive predicted arc (-1 dropped, repeats collapsed)."""
    a = np.asarray(pred_arcs, dtype=np.int64)
    a = a[a >= 0]
    if len(a) == 0:
        return []
    a = a[np.r_[True, a[1:] != a[:-1]]]
    fr, to = arcs["from_node"].to_numpy()[a], arcs["to_node"].to_numpy()[a]
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    return [[[float(lat[i]), float(lon[i])], [float(lat[j]), float(lon[j])]] for i, j in zip(fr, to)]


def build_model_features(cand, trip, clean, features):
    """Feature table for the model: the same builder that produced its training table."""
    if set(features) <= set(QUICK_FEATURES):
        return add_quick_features(cand.copy())
    return add_features(cand, {"nodes": trip["nodes"], "arcs": trip["arcs"]}, clean)


def predict_trip(trip_dir_root, trip_id, model, meta, use_hmm=True, hmm_params=None):
    """CONTRACT: run the full pipeline on one trip and return the API response dict (see module docstring)."""
    t0 = time.perf_counter()
    features = list(meta["features"])
    trip = load_trip_dir(trip_dir_root, trip_id)

    # 2-3. clean + one projection for points and arcs
    clean, report = clean_track(trip["track"])
    if len(clean) == 0:
        raise ValueError(f"trip {trip_id} has no valid GPS points after cleaning")
    proj = trip_projector(clean, report)
    clean = clean.copy()
    clean["x"], clean["y"] = proj(clean["lon"].to_numpy(), clean["lat"].to_numpy())
    geom = arc_geometry(trip["nodes"], trip["arcs"], proj)

    # 4-6. candidates for every point, features, probabilities
    obs = np.arange(len(clean))
    cand = generate_candidates(trip_id, clean["x"].to_numpy(), clean["y"].to_numpy(), obs, geom, RADIUS_M)
    if len(cand):
        feats = build_model_features(cand, trip, clean, features)
        missing = [f for f in features if f not in feats.columns]
        if missing:
            raise KeyError(f"model expects features that add_features does not produce: {missing}")
        proba = model.predict_proba(feats[features])[:, 1]
    else:
        proba = np.empty(0)

    # 7-8. classifier-only and HMM answers
    params = hmm_params if hmm_params is not None else load_hmm_params()
    clf = classifier_only(cand, proba, clean)
    hmm = decode_trip(cand, proba, trip["arcs"], clean, params) if use_hmm else None
    final = hmm if use_hmm else clf

    # 9. server-side accuracy, only if reference data exists
    metrics = {"top1_classifier": None, "top1_hmm": None,
               "route_continuity_classifier": _num(route_continuity(clf["pred_arc"].to_numpy(), trip["arcs"])),
               "route_continuity_hmm": (_num(route_continuity(hmm["pred_arc"].to_numpy(), trip["arcs"]))
                                        if use_hmm else None),
               "n_labelled_points": None, "runtime_s": None}
    route = trip["route"]
    if route is not None and len(route):
        truth = pd.DataFrame({"obs_idx": obs, "true_arc": label_points(clean, geom, route, method=LABEL_METHOD)})
        ok, n = top1_counts(clf, truth)
        metrics["n_labelled_points"] = n
        metrics["top1_classifier"] = _num(ok / n if n else float("nan"))
        if use_hmm:
            ok, n = top1_counts(hmm, truth)
            metrics["top1_hmm"] = _num(ok / n if n else float("nan"))
        del truth                                   # reference arcs never leave this function

    out = {
        "trip_id": trip_id,
        "n_points": int(len(clean)),
        "gps": [[float(la), float(lo)] for la, lo in zip(clean["lat"].to_numpy(), clean["lon"].to_numpy())],
        "predicted_arcs": [{"obs_idx": int(o), "arc_id": int(a)}
                           for o, a in zip(final["obs_idx"].to_numpy(), final["pred_arc"].to_numpy())],
        "predicted_route_coords": _route_polylines(final["pred_arc"].to_numpy(), trip["nodes"], trip["arcs"]),
        "metrics": metrics,
    }
    metrics["runtime_s"] = round(time.perf_counter() - t0, 3)
    assert tuple(out) == CONTRACT_KEYS
    return out
