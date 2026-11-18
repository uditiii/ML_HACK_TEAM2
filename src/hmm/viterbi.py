"""HMM + Viterbi route reconstruction (Person B).

Hidden state at GPS point t = one of that point's candidate arcs (within 50 m).
Observation = the GPS point; how well an arc explains it = the classifier probability.

  emission     log P(point t | arc)  = EMISSION_WEIGHT * log(clip(p, 1e-6, 1)), p = classifier probability
               (optionally renormalised so a point's candidate probabilities sum to 1).
               EMISSION_WEIGHT < 1 trusts an over-confident classifier less relative to the road graph
               (1.0 = the plain HMM).
  transition   log P(arc b at t+1 | arc a at t):
                 b == a                          -> log(p_stay)                 (still on the same arc)
                 b is the reverse arc of a       -> log(p_move) - UTURN_PENALTY (U-turn: allowed, discouraged)
                 b reachable from a in h hops    -> log(p_move) - HOP_PENALTY * (h - 1)
                   (h = 1: b starts where a ends; h = 2: one arc in between; ...)
                 otherwise                       -> log(EPS)                    (disconnected: almost impossible)
               The hop limit grows with the time between the two points:
                 max_hops = MAX_HOPS + int(dt / SECONDS_PER_EXTRA_HOP)
  Viterbi      delta_t(j) = emission_t(j) + max_i [ delta_{t-1}(i) + transition(i -> j) ]
               psi_t(j)   = the i that achieved the max (backpointer)
               The best final state is followed back through psi to get one arc per point.

Gaps and missing candidates:
  * The chain restarts at every segment_id change (a GPS gap > GAP_THRESHOLD_S): no transition is
    ever scored across a big gap, because we cannot know what happened in between.
  * A point with no candidate gets pred_arc = -1 and is skipped inside the chain: the transition goes
    from the last point that had candidates to the next one, and dt is the time between those two.
"""
from collections import deque

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {"p_stay": 0.6, "p_move": 0.4, "MAX_HOPS": 3, "HOP_PENALTY": 1.0,
                  "UTURN_PENALTY": 3.0, "EPS": 1e-6, "renormalise": True, "EMISSION_WEIGHT": 1.0}
SECONDS_PER_EXTRA_HOP = 5.0   # max_hops_t = MAX_HOPS + int(dt / 5)
PROBA_FLOOR = 1e-6            # emission: log(clip(p, 1e-6, 1))

# relation codes between a previous arc a and a current arc b (positive values = number of hops)
SAME, UTURN, DISCONNECTED = 0, -1, -2


# --------------------------------------------------------------------------------------------- #
# Road graph
# --------------------------------------------------------------------------------------------- #
class RoadGraph:
    """Directed road network of ONE trip, built once and reused for every step of the chain.

    outgoing arcs of a node are stored CSR-style (arcs sorted by from_node + start offsets), so
    "which arcs leave node v" is a slice, not a search. reverse[a] is the arc with the same two
    nodes swapped (-1 if the road is one-way). reachable() runs a breadth-first search over arcs
    and caches its answer per (arc, max_hops).
    """

    def __init__(self, arcs):
        a = arcs[["from_node", "to_node"]].to_numpy() if hasattr(arcs, "columns") else np.asarray(arcs)
        self.from_node = a[:, 0].astype(np.int64)
        self.to_node = a[:, 1].astype(np.int64)
        n_nodes = int(max(self.from_node.max(initial=-1), self.to_node.max(initial=-1))) + 1
        self._order = np.argsort(self.from_node, kind="stable")
        self._start = np.searchsorted(self.from_node[self._order], np.arange(n_nodes + 1))
        # reverse arc: look up (to, from) among all (from, to) pairs; first match wins
        pair_to_arc = pd.Series(np.arange(len(a)), index=pd.MultiIndex.from_arrays([self.from_node, self.to_node]))
        pair_to_arc = pair_to_arc[~pair_to_arc.index.duplicated(keep="first")]
        rev = pair_to_arc.reindex(pd.MultiIndex.from_arrays([self.to_node, self.from_node]))
        self.reverse = rev.fillna(-1).to_numpy().astype(np.int64)
        self._cache = {}

    def outgoing(self, node):
        """Arc ids that start at `node`."""
        return self._order[self._start[node]:self._start[node + 1]]

    def reachable(self, arc, max_hops):
        """{arc_id: hops} for every arc reachable from `arc` in 1..max_hops hops (BFS = fewest hops).

        1 hop = the next arc starts where `arc` ends (to_node == from_node).
        """
        key = (int(arc), int(max_hops))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        hops = {}
        frontier = deque((int(b), 1) for b in self.outgoing(self.to_node[arc]))
        while frontier:
            b, h = frontier.popleft()
            if b in hops:
                continue
            hops[b] = h
            if h < max_hops:
                frontier.extend((int(c), h + 1) for c in self.outgoing(self.to_node[b]) if int(c) not in hops)
        self._cache[key] = hops
        return hops

    def relation(self, prev_arcs, curr_arcs, max_hops):
        """(n_prev x n_curr) int matrix of relation codes: SAME, UTURN, hops (>0) or DISCONNECTED."""
        rel = np.full((len(prev_arcs), len(curr_arcs)), DISCONNECTED, dtype=np.int64)
        for i, a in enumerate(prev_arcs):
            r = self.reachable(a, max_hops)
            rel[i] = [r.get(int(b), DISCONNECTED) for b in curr_arcs]
            rel[i, curr_arcs == self.reverse[a]] = UTURN     # a U-turn is penalised, not a cheap 1-hop move
            rel[i, curr_arcs == a] = SAME
        return rel


# --------------------------------------------------------------------------------------------- #
# Emission and transition scores
# --------------------------------------------------------------------------------------------- #
def emission_logprob(proba, renormalise=True):
    """log(clip(p, 1e-6, 1)) for one point's candidates; optionally p / sum(p) first."""
    p = np.asarray(proba, dtype=np.float64)
    if renormalise and p.sum() > 0:
        p = p / p.sum()
    return np.log(np.clip(p, PROBA_FLOOR, 1.0))


def transition_logprob(rel, params):
    """Turn a relation-code matrix into log transition scores (see module docstring)."""
    out = np.full(rel.shape, np.log(params["EPS"]))
    out[rel == SAME] = np.log(params["p_stay"])
    out[rel == UTURN] = np.log(params["p_move"]) - params["UTURN_PENALTY"]
    move = rel > 0
    out[move] = np.log(params["p_move"]) - params["HOP_PENALTY"] * (rel[move] - 1)
    return out


# --------------------------------------------------------------------------------------------- #
# Chain preparation (graph work, done once) and Viterbi decoding (cheap, can be repeated)
# --------------------------------------------------------------------------------------------- #
def _track_columns(clean_track):
    trk = clean_track.sort_values("obs_idx", kind="stable") if "obs_idx" in clean_track.columns else clean_track
    obs = trk["obs_idx"].to_numpy() if "obs_idx" in trk.columns else np.arange(len(trk))
    seg = trk["segment_id"].to_numpy() if "segment_id" in trk.columns else np.zeros(len(trk), np.int64)
    return obs.astype(np.int64), trk["t"].to_numpy(dtype=np.float64), seg


def _candidates_by_obs(cand_df, proba):
    """{obs_idx: (arc_ids array, proba array)} in the candidate table's row order."""
    obs = cand_df["obs_idx"].to_numpy()
    arcs = cand_df["arc_id"].to_numpy().astype(np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    if len(proba) != len(cand_df):
        raise ValueError("proba must have one value per candidate row")
    order = np.argsort(obs, kind="stable")
    obs, arcs, proba = obs[order], arcs[order], proba[order]
    cut = np.flatnonzero(np.diff(obs)) + 1
    return {int(o[0]): (a, p) for o, a, p in zip(np.split(obs, cut), np.split(arcs, cut), np.split(proba, cut))
            if len(o)}


def prepare_chains(cand_df, proba, arcs, clean_track, params=None, graph=None):
    """Split the trip into Viterbi chains and pre-compute every step's relation matrix.

    A chain = consecutive points WITH candidates inside one segment_id. Returns a dict that
    decode_prepared() can decode many times with different penalty values (same MAX_HOPS).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    graph = graph if graph is not None else RoadGraph(arcs)
    obs_all, t_all, seg_all = _track_columns(clean_track)
    cands = _candidates_by_obs(cand_df, proba)

    chains, chain = [], None
    last_t = last_seg = None
    for o, t, s in zip(obs_all, t_all, seg_all):
        c = cands.get(int(o))
        if c is None:
            continue                                  # no candidate: -1, skipped inside the chain
        arc_ids, p = c
        if chain is None or s != last_seg:            # first point, or a GPS gap: start a new chain
            chain = {"obs": [], "arcs": [], "proba": [], "rel": []}
            chains.append(chain)
            rel = None
        else:
            max_hops = params["MAX_HOPS"] + int((t - last_t) / SECONDS_PER_EXTRA_HOP)
            rel = graph.relation(chain["arcs"][-1], arc_ids, max_hops)
        chain["obs"].append(int(o))
        chain["arcs"].append(arc_ids)
        chain["proba"].append(p)
        chain["rel"].append(rel)
        last_t, last_seg = t, s
    return {"obs_all": obs_all, "chains": chains, "MAX_HOPS": params["MAX_HOPS"]}


def viterbi_chain(chain, params):
    """Standard Viterbi in log space on one chain. Returns the best arc for each of its points."""
    n = len(chain["obs"])
    w = params.get("EMISSION_WEIGHT", 1.0)
    delta = w * emission_logprob(chain["proba"][0], params["renormalise"])
    psi = [None] * n
    for k in range(1, n):
        score = delta[:, None] + transition_logprob(chain["rel"][k], params)   # (n_prev x n_curr)
        best_prev = score.argmax(axis=0)                                       # backpointer per current arc
        delta = score[best_prev, np.arange(score.shape[1])] + w * emission_logprob(chain["proba"][k],
                                                                                    params["renormalise"])
        psi[k] = best_prev
    path = np.empty(n, dtype=np.int64)
    j = int(delta.argmax())
    for k in range(n - 1, -1, -1):                                             # backtracking
        path[k] = chain["arcs"][k][j]
        if k > 0:
            j = int(psi[k][j])
    return path


def decode_prepared(prepared, params=None):
    """Decode every chain; DataFrame(obs_idx, pred_arc) for every cleaned point (-1 = no candidate)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    if params["MAX_HOPS"] != prepared["MAX_HOPS"]:
        raise ValueError("MAX_HOPS differs from the one used in prepare_chains(); prepare again")
    pred = pd.Series(-1, index=pd.Index(prepared["obs_all"], name="obs_idx"), dtype=np.int64)
    for chain in prepared["chains"]:
        pred.loc[chain["obs"]] = viterbi_chain(chain, params)
    return pred.rename("pred_arc").reset_index()


def decode_trip(cand_df, proba, arcs, clean_track, params=None):
    """CONTRACT: HMM + Viterbi route for one trip -> DataFrame(obs_idx, pred_arc).

    cand_df: that trip's candidates (obs_idx, arc_id, ...); proba: classifier P(label=1) per row;
    arcs: the trip's arcs (from_node, to_node; row = arc_id); clean_track: cleaned track with
    obs_idx (or row order), t, segment_id. pred_arc = -1 where a point has no candidate.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    return decode_prepared(prepare_chains(cand_df, proba, arcs, clean_track, params), params)


# --------------------------------------------------------------------------------------------- #
# Baseline and metrics
# --------------------------------------------------------------------------------------------- #
def classifier_only(cand_df, proba, clean_track):
    """Baseline: each point independently takes its highest-probability candidate (-1 if none)."""
    obs_all, _, _ = _track_columns(clean_track)
    pred = pd.Series(-1, index=pd.Index(obs_all, name="obs_idx"), dtype=np.int64)
    for o, (arc_ids, p) in _candidates_by_obs(cand_df, proba).items():
        pred.loc[o] = arc_ids[int(np.argmax(p))]
    return pred.rename("pred_arc").reset_index()


def continuity_counts(pred_arcs_in_order, arcs):
    """(connected changes, all changes) after dropping -1 and collapsing repeated arcs."""
    a = np.asarray(pred_arcs_in_order, dtype=np.int64)
    a = a[a >= 0]
    if len(a) < 2:
        return 0, 0
    a = a[np.r_[True, a[1:] != a[:-1]]]            # collapse repeats: 5,5,5,7,7 -> 5,7
    if len(a) < 2:
        return 0, 0
    fr = arcs["from_node"].to_numpy() if hasattr(arcs, "columns") else np.asarray(arcs)[:, 0]
    to = arcs["to_node"].to_numpy() if hasattr(arcs, "columns") else np.asarray(arcs)[:, 1]
    connected = to[a[:-1]] == fr[a[1:]]
    return int(connected.sum()), int(len(connected))


def route_continuity(pred_arcs_in_order, arcs):
    """CONTRACT: share of arc changes where the next arc starts where the previous one ends.

    -1 (no candidate) is ignored and repeats are collapsed. NaN if the route never changes arc.
    Note: changes across a GPS gap are counted too, so a perfect route can score below 1.
    """
    ok, n = continuity_counts(pred_arcs_in_order, arcs)
    return ok / n if n else float("nan")


def top1_counts(pred_df, true_arc_by_obs):
    """(correct, labelled) over labelled points; a point predicted -1 counts as a miss."""
    m = pred_df.merge(true_arc_by_obs[["obs_idx", "true_arc"]], on="obs_idx", how="inner")
    m = m[m["true_arc"] >= 0]
    return int((m["pred_arc"] == m["true_arc"]).sum()), int(len(m))
