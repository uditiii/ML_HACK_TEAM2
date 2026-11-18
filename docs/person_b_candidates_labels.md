# Candidates and labels (Person B): how they are built and their limitations

All numbers live in `results/` and come from running the scripts. This file explains the method.

## What this part produces

For every GPS point we want (1) the short list of road arcs the car could plausibly be on
(**candidates**) and (2) which of them is the correct one (**label**). The classifier later
scores each (point, candidate) pair; the HMM picks one candidate per point.

| File | One row per | Columns |
|---|---|---|
| `data/processed/candidates_{partition}.parquet` | (GPS point, candidate arc) | trip_id, obs_idx, arc_id, dist_m, proj_t, label |
| `data/processed/obs_{partition}.parquet` | GPS point | trip_id, obs_idx, true_arc, n_candidates, has_true_candidate |
| `data/processed/clean_tracks_{partition}.parquet` | cleaned GPS point | trip_id, obs_idx, lon, lat, t, x, y, dt, segment_id |

Train uses every 5th cleaned point (`obs_idx % 5 == 0`), so consecutive train rows are ~5 s apart
and nearly identical rows are not repeated; student_test uses every cleaned point.

## Step 1: put points and roads in metres (`geometry.py`)

* `trip_projector`: one equirectangular projection per trip, centred on the mean of the trip's
  cleaned GPS points: `x = R·(lon−lon0)·cos(lat0)`, `y = R·(lat−lat0)`. It is a flat map in metres.
  The error is tiny over a trip-sized area, and much smaller than GPS noise. **GPS points and road
  nodes use the same centre**, otherwise distances would be wrong.
* `arc_geometry`: project every node once, then each arc becomes a straight segment from
  A = from_node to B = to_node: columns `ax, ay, bx, by, length_m`. Vectorised: no loop over arcs.

## Step 2: find candidates fast (`generate.py`)

**Exact distance** (`point_segment_distance`): project P onto the line AB, giving
`t = (AP·AB)/|AB|²`, then clamp t to [0, 1]. The closest point is `A + t·AB`. t = 0 means the
closest point is A, t = 1 means B, in between means the foot of the perpendicular. This t is
saved as `proj_t`.

**Why not just a KD-tree on arc midpoints?** A long arc can pass 5 m from a point while its midpoint
is hundreds of metres away, so it would be missed.

**What we do instead:**
1. `prefilter_arcs`: drop arcs whose bounding box is nowhere near the track (big networks shrink a lot).
2. `densify_arcs`: put sample points along every arc at most 10 m apart, endpoints included
   (`n = ceil(L/10)` pieces → `n+1` samples), and remember which arc each sample came from.
3. Build a `cKDTree` on the samples and, for each GPS point, ask for all samples within
   **60 m** (= 50 + 10).
4. The arcs of those samples are the "maybe" list; compute the **exact** point-to-segment distance
   and keep arcs with distance ≤ 50 m.

**Why 60 m can never miss an arc (the key argument):** if an arc is within 50 m of P, let Q be its
closest point to P. Samples are ≤ 10 m apart, so some sample S is ≤ 5 m from Q. Then
|PS| ≤ |PQ| + |QS| ≤ 50 + 5 < 60. So the KD-tree always returns at least one sample of every true
candidate. The exact check in step 4 then removes the extras, so the result is exactly
"all arcs within 50 m". `scripts/check_candidates.py` confirms the fast search returns the same pairs
as brute force (distance to every arc).

**Speed:** the KD-tree query is O(log n) per point instead of measuring all arcs (up to ~307k).

## Step 3: labels (`labels.py`): label creation only

The `.route` file is the ordered list of arcs driven, but it has far fewer rows than the track and no
timestamps, so we must decide which route arc each GPS point belongs to. **The route is only used
here, to make the `label` column. It is never used for candidates or features**, and nothing like it
exists at inference time.

* **v1, nearest route arc:** for each point, the closest arc among the route's arcs; if it is more
  than 50 m away, the point is unlabelled (-1).
  *Weakness:* it ignores order. When the car drives a road in both directions, the arc and its reverse
  are equally close, and v1 cannot tell which direction is right. It also flips between neighbouring
  route arcs at junctions. This shows up as **backward steps** (the label moves backwards along the
  route).
* **v2, monotone dynamic programme (used by default):** each point chooses a *position* in the route
  (up to the 8 closest positions within 50 m). Cost = distance in metres + 30 for every backward step
  + 15 for a forward jump of more than 10 positions. We minimise the total cost over the whole track
  with dynamic programming (keep the cheapest cost of reaching each state, store the best predecessor),
  then backtrack from the cheapest final state. It is the same idea as Viterbi, applied to labelling.
  Result: almost no backward steps, while labels stay almost as close to the GPS points as v1's.
  `scripts/compare_labels.py` prints both versions side by side.
* **Ties:** distances are rounded to 1 µm before choosing, so exact ties (an arc and its reverse)
  are broken the same way on every computer (lowest arc_id / lowest route position). Without this,
  different NumPy versions gave slightly different labels.

Then `label = 1` for the candidate whose arc_id equals the point's label, else 0. There is at most
one positive per point.

## Reading the summary (`results/candidates_summary.csv`)

* **Candidate Miss Rate = 0 is expected, not a success claim.** A label is always an arc within 50 m,
  and candidates are *all* arcs within 50 m, so the labelled arc is always among the candidates.
  The real misses are the **unlabelled** points, where no route arc is within 50 m.
* **Unlabelled points:** most train unlabelled points come from a few trips (see
  `results/label_quality_*_train.csv`, column `pct_unlabelled`). Those tracks run far (> 50 m) from
  every route arc for long stretches, so the route and the track disagree there. These points are kept
  with true_arc = -1 and count as misses in Top-1.
* **Positive rate around 10%:** about 1 correct arc among about 9–11 candidates per point. The classes
  are imbalanced, which is why macro-F1 is used.
* **Both directions:** most roads appear as two arcs (A→B and B→A) with identical distance. Distance
  alone cannot separate them; heading/bearing features or the HMM have to.

## Limitations (say these before you are asked)

1. **Fixed 50 m radius:** a point with GPS error > 50 m has no correct candidate; dense city centres
   give many candidates.
2. **Arcs are straight segments:** curved roads are approximated by their node-to-node chords.
3. **The projection is local and flat:** fine for one trip; it would be wrong across the ±180°
   longitude line (no trip crosses it).
4. **Labels are derived, not observed:** they depend on the route file being correct, on the 50 m
   rule, and (v2) on hand-picked penalties (30, 15, 10, 8). They are reference labels, not verified
   ground truth.
5. **v2 considers only the 8 closest route positions per point:** on routes that pass the same place
   many times, the correct position could be outside those 8.

## How to verify (run from the repo root)

```
python scripts/build_candidates.py        # builds everything, prints per-trip progress and the summary
python scripts/check_outputs.py           # columns, ranges, joins, one positive per obs -> ALL CHECKS PASSED
python scripts/check_candidates.py        # fast search == brute force on sampled points
python scripts/compare_labels.py          # v1 vs v2 label quality
python scripts/plot_candidates.py --trip 00000000 --obs 100   # picture of one point's candidates
```
