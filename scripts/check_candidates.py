"""Task 2 check: the fast KD-tree candidate search must return EXACTLY the same (obs, arc) pairs
as a brute-force search that measures the distance to every arc of the network.

Usage (from the repo root): python scripts/check_candidates.py [--n 50]
Tests n random cleaned points in every student_test trip and the first 5 train trips.
Exits with code 1 if any trip differs.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import RANDOM_STATE
from scripts.build_candidates import load_projected
from src.candidates.generate import brute_force_candidates, generate_candidates
from src.data.loader import list_trips

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
args = ap.parse_args()

rng = np.random.default_rng(RANDOM_STATE)
trips = [("student_test", t) for t in list_trips("student_test")] + \
        [("train", t) for t in list_trips("train")[:5]]
bad = 0
for partition, trip_id in trips:
    _, clean, geom, _ = load_projected(partition, trip_id)
    obs = np.sort(rng.choice(len(clean), size=min(args.n, len(clean)), replace=False))
    x, y = clean["x"].to_numpy()[obs], clean["y"].to_numpy()[obs]
    fast = generate_candidates(trip_id, x, y, obs, geom)
    slow = brute_force_candidates(trip_id, x, y, obs, geom)
    same = set(zip(fast.obs_idx, fast.arc_id)) == set(zip(slow.obs_idx, slow.arc_id))
    bad += not same
    print(f"{'OK  ' if same else 'FAIL'} {partition} {trip_id}: {len(obs)} points, {len(geom)} arcs, "
          f"fast pairs={len(fast)}, brute-force pairs={len(slow)}")
print("FAST SEARCH == BRUTE FORCE ON ALL TESTED TRIPS" if bad == 0 else f"{bad} TRIP(S) DIFFER")
sys.exit(1 if bad else 0)
