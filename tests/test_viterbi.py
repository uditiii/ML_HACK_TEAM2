"""Tests for src/hmm/viterbi.py on a tiny hand-made road network.

Run: python -m pytest tests/test_viterbi.py -q     (or: python tests/test_viterbi.py)

Network: a straight two-way street 0 - 1 - 2 - 3, plus a far-away arc 10 -> 11.
  arc 0: 0->1   arc 1: 1->2   arc 2: 2->3        (forward)
  arc 3: 1->0   arc 4: 2->1   arc 5: 3->2        (reverse twins)
  arc 6: 10->11                                  (disconnected)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hmm.viterbi import (DISCONNECTED, SAME, UTURN, RoadGraph, classifier_only,  # noqa: E402
                             decode_trip, route_continuity)

ARCS = pd.DataFrame({"from_node": [0, 1, 2, 1, 2, 3, 10], "to_node": [1, 2, 3, 0, 1, 2, 11]})


def track(n, segment_ids=None):
    return pd.DataFrame({"obs_idx": np.arange(n), "t": np.arange(n, dtype=float),
                         "segment_id": segment_ids if segment_ids is not None else np.zeros(n, int)})


def cands(rows):
    """rows: list of (obs_idx, arc_id, proba) -> (cand_df, proba)."""
    df = pd.DataFrame(rows, columns=["obs_idx", "arc_id", "proba"])
    return df[["obs_idx", "arc_id"]], df["proba"].to_numpy()


def test_graph_relations():
    g = RoadGraph(ARCS)
    assert g.reverse.tolist() == [3, 4, 5, 0, 1, 2, -1]
    assert g.reachable(0, 3) == {1: 1, 4: 2, 2: 2, 3: 1, 0: 2, 5: 3}   # BFS through the two-way street
    rel = g.relation(np.array([0]), np.array([0, 3, 1, 2, 6]), max_hops=3)
    assert rel.tolist() == [[SAME, UTURN, 1, 2, DISCONNECTED]]


def test_hmm_fixes_a_wrong_direction_pick():
    # true route 0,0,1,1,2,2; at obs 2 the classifier prefers the reverse twin (arc 4) by mistake
    c, p = cands([(0, 0, .9), (0, 3, .1), (1, 0, .8), (1, 3, .2), (2, 1, .4), (2, 4, .6),
                  (3, 1, .8), (3, 4, .2), (4, 2, .9), (4, 5, .1), (5, 2, .9), (5, 5, .1)])
    clf = classifier_only(c, p, track(6))
    hmm = decode_trip(c, p, ARCS, track(6))
    assert clf["pred_arc"].tolist() == [0, 0, 4, 1, 2, 2]
    assert hmm["pred_arc"].tolist() == [0, 0, 1, 1, 2, 2]


def test_no_transition_across_a_segment_boundary():
    # obs 3 prefers the far-away arc 6. Same segment: the HMM refuses the impossible jump.
    # New segment (GPS gap): no transition is scored, so arc 6 is kept.
    rows = [(0, 0, .9), (1, 0, .9), (2, 1, .9), (3, 6, .9), (3, 2, .1)]
    c, p = cands(rows)
    same_seg = decode_trip(c, p, ARCS, track(4, [0, 0, 0, 0]))
    new_seg = decode_trip(c, p, ARCS, track(4, [0, 0, 0, 1]))
    assert same_seg["pred_arc"].tolist() == [0, 0, 1, 2]
    assert new_seg["pred_arc"].tolist() == [0, 0, 1, 6]


def test_point_without_candidates_is_minus_one_and_skipped():
    c, p = cands([(0, 0, .9), (0, 3, .1), (2, 1, .4), (2, 4, .6), (3, 1, .9)])
    hmm = decode_trip(c, p, ARCS, track(4))
    assert hmm["obs_idx"].tolist() == [0, 1, 2, 3]
    assert hmm["pred_arc"].tolist() == [0, -1, 1, 1]


def test_route_continuity():
    assert route_continuity([0, 0, 1, -1, 1, 2], ARCS) == 1.0      # 0->1->2: both changes connected
    assert route_continuity([0, 2], ARCS) == 0.0                   # arc 0 ends at 1, arc 2 starts at 2
    assert route_continuity([0, 1, 6], ARCS) == 0.5
    assert np.isnan(route_continuity([0, 0, -1], ARCS))            # no change at all


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK  ", name)
    print("ALL VITERBI TESTS PASSED")
