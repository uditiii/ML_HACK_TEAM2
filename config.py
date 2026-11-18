"""Project-wide constants. Every module imports from here so paths/thresholds live in ONE place."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Raw data: <DATA_ROOT>/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
# Override with env var PATHSNAP_DATA if your copy lives elsewhere.
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", PROJECT_ROOT / "data" / "raw"))

RESULTS_DIR = PROJECT_ROOT / "results"
MAPS_DIR = RESULTS_DIR / "maps"

PARTITIONS = ("train", "student_test")   # evaluator_test is never touched by us

# A step longer than this (seconds) starts a new segment.
# Data is nominally 1 Hz (~97% of steps are exactly 1 s), so 10 s = 10 missed fixes;
# at 50 km/h that is ~140 m of unobserved driving -> enough to take a different road.
GAP_THRESHOLD_S = 10.0

# Map: pad GPS bounding box by this many metres when selecting road arcs to draw.
MAP_BBOX_PAD_M = 200.0

RANDOM_SEED = 42
