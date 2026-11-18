"""Project-wide constants. Every module imports from here so paths/thresholds live in ONE place.

Names marked CONTRACT are the team's shared repo contract: do not rename them.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Raw data: <DATA_ROOT>/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
# Override with env var PATHSNAP_DATA if your copy lives elsewhere.
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", PROJECT_ROOT / "data" / "raw" / "Student folder"))  # CONTRACT

RESULTS_DIR = PROJECT_ROOT / "results"
MAPS_DIR = RESULTS_DIR / "maps"

PARTITIONS = ["train", "student_test"]   # CONTRACT; evaluator_test is never touched by us

RADIUS_M = 50          # CONTRACT: candidate radius in metres
TRAIN_STEP = 5         # CONTRACT: train rows use obs_idx % TRAIN_STEP == 0

# CONTRACT (locked team decision): a step longer than this (seconds) starts a new segment.
# NOTE: Person A's first version used 10.0. Changing it only changes segment_id
# (features / HMM), not candidates or labels. Confirm with the team before changing.
GAP_THRESHOLD_S = 30

# Map: pad GPS bounding box by this many metres when selecting road arcs to draw.
MAP_BBOX_PAD_M = 200.0

RANDOM_STATE = 42      # CONTRACT
RANDOM_SEED = RANDOM_STATE   # alias used by Person A's code
