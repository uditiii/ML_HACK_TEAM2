"""Call a RUNNING PathSnap API for every trip and record the runtimes (Task 4 test).

Usage (API already running, from anywhere):
  python scripts/test_api.py                          # http://localhost:8000
  python scripts/test_api.py --url http://192.168.1.20:8000
Prints one line per trip and saves results/api_test_runtimes.csv.
Also checks that every response has exactly the contract keys and no reference data.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_KEYS = ["trip_id", "n_points", "gps", "predicted_arcs", "predicted_route_coords", "metrics"]
METRIC_KEYS = ["top1_classifier", "top1_hmm", "route_continuity_classifier", "route_continuity_hmm",
               "n_labelled_points", "runtime_s"]

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
args = ap.parse_args()

trips = requests.get(f"{args.url}/trips", timeout=30).json()["trips"]
print(f"GET /trips -> {len(trips)} trips: {trips}")
rows, bad = [], 0
for tid in trips:
    t0 = time.perf_counter()
    r = requests.post(f"{args.url}/predict", json={"trip_id": tid}, timeout=600)
    wall = time.perf_counter() - t0
    if r.status_code != 200:
        bad += 1
        print(f"FAIL {tid}: HTTP {r.status_code} {r.text[:200]}")
        continue
    out = r.json()
    ok = list(out) == CONTRACT_KEYS and list(out["metrics"]) == METRIC_KEYS and "true_arc" not in r.text
    bad += not ok
    m = out["metrics"]
    rows.append({"trip_id": tid, "n_points": out["n_points"], "server_runtime_s": m["runtime_s"],
                 "client_wall_s": round(wall, 3), "response_kb": round(len(r.content) / 1024, 1), **{
                     k: m[k] for k in METRIC_KEYS if k != "runtime_s"}, "contract_ok": ok})
    print(f"{'OK  ' if ok else 'BAD '} {tid}: points={out['n_points']} server={m['runtime_s']}s wall={wall:.2f}s "
          f"top1_clf={m['top1_classifier']} top1_hmm={m['top1_hmm']} cont_hmm={m['route_continuity_hmm']}")
unknown = requests.post(f"{args.url}/predict", json={"trip_id": "does_not_exist"}, timeout=30).status_code
print(f"unknown trip_id -> HTTP {unknown} (expected 404)")
bad += unknown != 404

df = pd.DataFrame(rows)
(ROOT / "results").mkdir(exist_ok=True)
df.to_csv(ROOT / "results" / "api_test_runtimes.csv", index=False)
if len(df):
    print(f"saved results/api_test_runtimes.csv | mean server runtime {df['server_runtime_s'].mean():.3f}s, "
          f"max {df['server_runtime_s'].max():.3f}s")
print("ALL API CHECKS PASSED" if bad == 0 else f"{bad} PROBLEM(S)")
sys.exit(1 if bad else 0)
