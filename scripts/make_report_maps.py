"""Build the offline REPORT maps: GPS + reference route + predicted route for 3 student_test trips.

    python scripts/make_report_maps.py
    python scripts/make_report_maps.py --trips 00000052 00000056 00000065
    python scripts/make_report_maps.py --model artifacts/models/final_model.joblib --meta artifacts/models/final_model_meta.json

Needs Person C's src/pipeline/predict.py:
    predict_trip(trip_dir_root, trip_id, model, meta, use_hmm=True) -> dict with "predicted_arcs"

Output: results/maps/report_<trip_id>.html
The reference (green) route is drawn ONLY here, for the written report. The UI never gets it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_ROOT  # noqa: E402
from src.data.loader import list_trips, load_trip  # noqa: E402
from src.data.preprocess import clean_track  # noqa: E402
from src.visualisation.maps import (MAPS_DIR, arcs_to_polylines,  # noqa: E402
                                    network_polylines_in_bbox, route_map)

PARTITION = "student_test"
TRIP_ROOT = Path(DATA_ROOT) / PARTITION
DEFAULT_MODEL = ROOT / "artifacts" / "models" / "final_model.joblib"
DEFAULT_META = ROOT / "artifacts" / "models" / "final_model_meta.json"


def load_model_and_meta(model_path: Path, meta_path: Path):
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\nAsk Person C/D where the final model is saved "
                         f"and pass it with --model")
    model = joblib.load(model_path)
    if isinstance(model, dict) and "model" in model:          # {"model": ..., "meta": ...} bundle
        return model["model"], model.get("meta", {})
    if meta_path.exists():
        meta = (json.loads(meta_path.read_text()) if meta_path.suffix == ".json"
                else joblib.load(meta_path))
    else:
        from src.features.feature_list import FEATURES
        print(f"note: {meta_path.name} not found, using meta = {{'features': FEATURES}}")
        meta = {"features": list(FEATURES)}
    return model, meta


def pick_trips():
    """short = fewest clean points, long = most, big gap = largest max_gap_s (all different)."""
    stats = []
    for tid in list_trips(PARTITION):
        _, rep = clean_track(load_trip(PARTITION, tid)["track"])
        stats.append((tid, rep["clean_points"], rep["max_gap_s"]))
    short = min(stats, key=lambda s: s[1])[0]
    long_ = max(stats, key=lambda s: s[1])[0]
    gap = max((s for s in stats if s[0] not in (short, long_)), key=lambda s: s[2])[0]
    return {short: "short", long_: "long", gap: "big gap"}


def build_report_map(trip_id, predict_trip, model, meta, label=""):
    trip = load_trip(PARTITION, trip_id)
    nodes, arcs = trip["nodes"], trip["arcs"]
    clean, rep = clean_track(trip["track"])

    result = predict_trip(TRIP_ROOT, trip_id, model, meta, use_hmm=True)
    pred_ids = np.asarray([a["arc_id"] if isinstance(a, dict) else a for a in result["predicted_arcs"]], dtype=np.int64)
    pred_ids = pred_ids[(pred_ids >= 0) & (pred_ids < len(arcs))]   # -1 = point had no candidate
    pred = arcs_to_polylines(nodes, arcs, pred_ids)
    ref = arcs_to_polylines(nodes, arcs, trip["route"]["arc_id"]) if trip["route"] is not None else None
    net = network_polylines_in_bbox(nodes, arcs, clean["lat"], clean["lon"])

    title = (f"{PARTITION}/{trip_id}" + (f" ({label})" if label else "")
             + f" | {rep['clean_points']} pts, max gap {rep['max_gap_s']:.0f} s")
    m = route_map(clean[["lat", "lon"]].to_numpy(), pred, ref, net, title=title)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    out = MAPS_DIR / f"report_{trip_id}.html"
    m.save(str(out))
    return out, result.get("metrics", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*", help="student_test trip ids (default: auto short / long / big gap)")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    args = ap.parse_args()

    try:
        from src.pipeline.predict import predict_trip
    except ImportError as e:
        raise SystemExit(f"Cannot import src.pipeline.predict.predict_trip ({e}).\n"
                         f"Person C's pipeline must be merged first.")
    model, meta = load_model_and_meta(args.model, args.meta)
    chosen = {t: "" for t in args.trips} if args.trips else pick_trips()
    for tid, label in chosen.items():
        out, metrics = build_report_map(tid, predict_trip, model, meta, label)
        print(f"{tid} {label:8s} -> {out.relative_to(ROOT)} | metrics {metrics}")


if __name__ == "__main__":
    main()
