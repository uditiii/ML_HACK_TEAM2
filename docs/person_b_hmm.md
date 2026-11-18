# HMM + Viterbi route reconstruction (Person B)

Code: `src/hmm/viterbi.py` (model + decoder), `src/hmm/experiment.py` (evaluation),
`scripts/run_hmm_experiment.py` / `notebooks/05_hmm.ipynb` (the experiment), `tests/test_viterbi.py` (checks).
All numbers come from `results/hmm_comparison.csv`, `results/hmm_param_search.csv` and the printed run.
This file only explains the method.

## Why an HMM at all

The classifier looks at each GPS point **alone**. Two problems follow:
1. **Direction.** Most roads are stored twice: A→B and its reverse twin B→A. Both are exactly the same
   distance from the GPS point, so distance-only features cannot tell them apart; the classifier is
   close to a coin flip there. That is why every Phase 1 model had about the same Top-1.
2. **Jumps.** Consecutive points can be matched to roads that are not connected (a parallel street,
   a side road), which no real car could drive.

The HMM adds the missing information: **the order of the points and how the roads connect.**

## The model in plain words

* **Hidden state** (what we want to know): for each GPS point, *which of its candidate arcs* the
  vehicle is on. Only the candidates within 50 m are allowed, so the state space is small (about 10
  per point), and every state is a real `arc_id`, so the candidate-to-arc correspondence is preserved.
* **Observation** (what we see): the GPS point.
* **Emission** (how well an arc explains the point): `log(clip(p, 1e-6, 1))`, where `p` is the
  classifier probability for that candidate. With `renormalise = True`, a point's probabilities are first
  divided by their sum, so every point gives the decoder the same total "vote". The clip stops
  `log(0) = -infinity` from forbidding an arc completely.
  **Emission weight:** the log emission is multiplied by `EMISSION_WEIGHT` (1 = plain HMM). A value below 1
  tells the decoder to trust the classifier less relative to the road graph. This matters for the tuned final
  forest: it is deeper and more confident, so at weight 1 its probabilities can overrule the connectivity
  evidence that fixes direction errors. The weight is chosen by the same train-only search as the penalties.
  Mathematically, scaling the emissions by w gives the same best path as scaling all transition scores by 1/w.
* **Transition** (how plausible it is to go from arc *a* at one point to arc *b* at the next):

| Case | Score (log) | Meaning |
|---|---|---|
| b = a | `log(p_stay)` | still on the same road segment (most common at 1 Hz) |
| b reachable from a in h hops | `log(p_move) − HOP_PENALTY·(h−1)` | normal driving; 1 hop = b starts where a ends; each extra arc in between costs more |
| b = reverse twin of a | `log(p_move) − UTURN_PENALTY` | a U-turn: possible, but discouraged |
| otherwise | `log(EPS)` = log(1e-6) | disconnected: almost impossible, but not forbidden, so one bad point cannot break the decode |

  "Reachable in h hops" is found by a breadth-first search along directed arcs (`to_node` of one arc =
  `from_node` of the next), cached per arc. The hop limit grows with the time between the two points,
  `max_hops = MAX_HOPS + int(dt / 5)`, because a car can cover more arcs when more time has passed.
  Directed arcs mean one-way streets are respected automatically.

* **How the HMM fixes direction:** if the car drives A→B→C, the forward arcs follow each other in 1 hop
  each (cheap). The reverse twins (B→A, then C→B) do not connect in that time order; reaching them costs
  extra hops or a U-turn. So over a sequence of points, the correct direction wins clearly, even though
  each single point cannot tell.

## Viterbi (the decoder)

We want the single sequence of arcs with the highest total score (emissions + transitions). There are
far too many sequences to try (≈10ⁿ for n points), but Viterbi finds the best one exactly in
O(n · k²) time (k = candidates per point), using two tables:

* `delta_t(j)` = score of the **best path that ends in candidate j at point t**.
* `psi_t(j)` = which candidate at point t−1 that best path came from (the backpointer).

Recurrence:

```
delta_0(j) = emission_0(j)
delta_t(j) = emission_t(j) + max_i [ delta_{t-1}(i) + transition(i -> j) ]
psi_t(j)   = argmax_i [ delta_{t-1}(i) + transition(i -> j) ]
```

At the end we take the best final candidate and **backtrack** with `psi` to read off one arc per point.
Each step is computed as one `(k_prev × k_curr)` matrix in NumPy. Everything is in **log space**, so
products of many small probabilities become sums and never underflow to 0.

Why it is exact: the best path to (t, j) must extend the best path to some (t−1, i). Any other path
to that i scores lower and can be thrown away. This is the dynamic-programming principle.

## Gaps, missing candidates and disconnected roads (report questions 4 and 5)

* **Big GPS gap** (time step > `GAP_THRESHOLD_S` = 30 s → new `segment_id`): the chain is **cut** and
  Viterbi restarts. We do not know what happened during the gap, so no transition is scored across it.
* **Point with no candidate** (no arc within 50 m): it gets `pred_arc = -1` and is **skipped** inside
  the chain. The transition goes from the previous point that had candidates to the next one, and the
  hop limit uses the real time between them.
* **Consecutive points suggesting disconnected roads:** that transition only gets `log(1e-6)`, so the
  decoder prefers a connected alternative whenever the emissions allow it. If the GPS strongly insists
  (or the network itself is broken), the disconnected move is still possible, so the decode never fails.

## Parameter choice (no evaluator data)

Grid: `p_stay ∈ {0.5, 0.6, 0.8}` (with `p_move = 1 − p_stay`), `HOP_PENALTY ∈ {0.5, 1, 2}`,
`UTURN_PENALTY ∈ {1, 3, 6}`, `EMISSION_WEIGHT ∈ {1, 0.5, 0.25}`: 81 settings, on 15 randomly chosen TRAIN trips (seed 42) at full rate.
Selection = highest pooled observation-level Top-1. **The train trips are scored out-of-fold**: each trip
gets probabilities from a forest refitted without that trip's CV fold (`results/folds.csv`), because
in-sample probabilities are over-confident and would make the HMM trust emissions too much. The chosen
setting is then evaluated **once** on student_test. If the best value sits at the edge of the grid,
say so: a wider grid is future work.

## Metrics

* **Top-1:** per labelled point, predicted arc = reference arc (route labels v2); a point with no
  candidate counts as a miss. Pooled over all points for the OVERALL row.
* **Route continuity:** drop −1, collapse repeats, then the share of arc changes where the next arc
  starts at the node where the previous one ends. Changes across GPS gaps are counted too, so even a
  perfect route can score below 1.
* **reverse_err:** labelled points predicted as the reverse twin of the reference arc (the direction error).

## Answering the report's HMM questions

1. **Hidden states:** the candidate arcs (within 50 m) of each GPS point; see "The model in plain words".
2. **Emissions:** log of the classifier probability, clipped at 1e-6, renormalised per point.
3. **Transitions:** stay / reachable in h hops / U-turn / disconnected, using directed connectivity and
   a time-dependent hop limit; values from the train grid search (`results/hmm_chosen_params.json`).
4. **Missing candidates / gaps:** −1 and skipped; chain restarts at every segment boundary.
5. **Disconnected consecutive segments:** penalised with log(1e-6), never forbidden.
6. **Did it help?** Quote the OVERALL row of `results/hmm_comparison.csv` (`top1_clf` vs `top1_hmm`,
   `cont_clf` vs `cont_hmm`, `reverse_err_clf` vs `reverse_err_hmm`). The drop in reverse-twin errors
   shows *why* it helps: the order of points reveals the driving direction. For "where it did not help",
   use the printed longest wrong runs (step [4]) and the maps in `results/figures/hmm_student_test_*.png`.
   A long run on **one** arc, with GPS points inside a small box and almost no start-to-end distance,
   means the vehicle was (nearly) standing or circling. No arc change gives no transition evidence,
   and distance-only emissions are equal for both twins, so the direction is a guess for that stretch.
   Direction-aware features (Person C's `heading_diff_deg`) attack the same problem from the classifier side.

## Viva questions you may get

* *Why log space?* Multiplying hundreds of probabilities underflows to 0; sums of logs do not.
* *Complexity?* O(n·k²) time, O(n·k) memory for the backpointers; k ≈ 10, so it is fast.
* *Why not just smooth the classifier output?* Smoothing ignores the road graph; the HMM only allows
  moves the network permits and prices them by hops.
* *Why EPS instead of −infinity for disconnected moves?* A single noisy point or a missing arc in the
  map would otherwise make every path impossible.
* *Why restart at gaps?* Over a 30+ s gap the vehicle may have taken any road; forcing a connection
  would invent a route.
* *Why an emission weight?* A better classifier by macro-F1 is not automatically better inside the HMM: a more
  confident model leaves the decoder less room to use the road graph. Down-weighting emissions (chosen on train
  only) restores that balance. Compare the student_test rows with and without it if asked.
* *Why out-of-fold scores for tuning?* The forest has seen the train trips; its in-sample probabilities
  are too confident, which would bias the HMM towards ignoring transitions.
* *Limitations:* fixed hand-set transition form; hop count ignores arc lengths and speed; continuity
  counts gap transitions; stationary/circling stretches stay ambiguous; labels are derived from the
  route, not observed.

## Using it in the API (`src/pipeline/predict.py`)

```python
from src.hmm.viterbi import DEFAULT_PARAMS, decode_trip, classifier_only, route_continuity, top1_counts
hmm = decode_trip(cand_df, proba, trip["arcs"], clean_track, params)   # DataFrame(obs_idx, pred_arc)
clf = classifier_only(cand_df, proba, clean_track)
cont = route_continuity(hmm["pred_arc"].to_numpy(), trip["arcs"])
```
Use the parameters in `results/hmm_chosen_params.json`. `cand_df` must have candidates for every
cleaned point (`generate_candidates` with `obs_idx = 0..n-1`).
