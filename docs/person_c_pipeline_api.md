# Features v2, feature selection, prediction pipeline and API (Person C)

Numbers live in `results/` and the console output of your own runs; this file explains the method.

## 1. Features (`src/features/build_features.py`)

Every feature uses **only the cleaned GPS track and the road network** (`.nodes`, `.arcs`).
`add_features` physically selects only `trip_id, obs_idx, arc_id, dist_m, proj_t` before computing
anything, so `label`, `true_arc`, `has_true_candidate` and the route can never influence a feature.
`tests/test_build_features.py::test_no_label_leakage` shuffles/drops the label columns and checks that every
feature stays bit-for-bit identical.

Distances use the locked projection (equirectangular, centred on the mean of the trip's cleaned GPS
points, R = 6,371,008.8 m). It is checked against the `x, y` stored in `clean_tracks` (must agree within 5 cm),
plus a guard that projected step lengths agree with great-circle lengths.

| Feature | Plain-words meaning | Why it helps |
|---|---|---|
| `dist_m` | distance from the GPS point to the arc segment | the true road is usually close |
| `proj_t` | where on the arc the point projects (0 = start, 1 = end) | near 0/1 = near a junction, more ambiguity |
| `dist_rank` | 1 = closest arc for this point (ties share the best rank) | relative, works for noisy points too |
| `dist_minus_min` | how much further than the closest arc | 0 = best match; large = clearly worse |
| `n_candidates` | how many arcs compete for this point | dense area = harder |
| `arc_length_m` | length of the arc | short connector arcs are rarely the right answer |
| `gps_bearing` | direction of motion: from the previous to the next cleaned point (same segment only; one-sided at segment ends) | input for the heading comparison |
| `arc_bearing` | direction of the arc, from_node → to_node | input for the heading comparison |
| `heading_diff_deg` | smallest angle between motion and arc, 0–180 (90 when stationary) | **separates an arc from its reverse twin** (≈0 vs ≈180), which distance cannot |
| `heading_rank` | rank of `heading_diff_deg` among this point's candidates (1 = best aligned) | relative version: "is this the arc pointing most nearly our way?" |
| `speed_mps` | centred distance / centred time | tells whether the direction is trustworthy |
| `heading_valid` | 1 if speed ≥ 1 m/s | when stopped the bearing is noise; heading_diff is then set to neutral 90 |

`speed_mps`, `heading_valid` and `n_candidates` are the same for every candidate of one point, so they
cannot rank candidates on their own; they change how much the model trusts the other features.

**Why 90° when stationary?** 90 is "neither aligned nor reversed", so a stopped vehicle gives no direction
evidence either way, and `heading_valid = 0` tells the model to ignore it.

Rebuild the tables: `python scripts/build_tables.py` (needs `scripts/build_candidates.py` outputs).

## 2. Feature selection (`notebooks/03_feature_selection.ipynb`, TRAIN only)

Five views of each feature: summary by label; Spearman correlation (flag |ρ| > 0.9); ANOVA F (`f_classif`);
mutual information (stratified sample ≤ 200k rows, seed 42); Random-Forest impurity importance; permutation
importance (drop in macro-F1) on a 20% hold-out of **whole trips** (GroupShuffleSplit), so the forest is
scored on trips it never saw. **Rule:** drop a feature only if it is redundant (|ρ| > 0.9 with a stronger
feature) **and** below the median on F, MI and RF importance. The notebook writes `results/feature_selection.csv`
and overwrites `src/features/feature_list.py`. Fill `results/feature_selection.md` with your printed numbers.

Viva points: univariate scores (F, MI) ignore interactions (`heading_diff_deg` only matters with
`heading_valid`); impurity importance favours continuous features; permutation importance on held-out trips is the
least biased; student_test and evaluator data are never loaded in this notebook.

## 3. Prediction pipeline (`src/pipeline/predict.py`)

`predict_trip(trip_dir_root, trip_id, model, meta, use_hmm=True)` runs the training-time steps on one trip:

1. read `<trip_dir_root>/<trip_id>/` (works for `student_test` now and the evaluator folder at the demo);
2. `clean_track` (Person A); `obs_idx` = cleaned row number;
3. one projector for points and arcs;
4. candidates within 50 m for **every** point (Person B);
5. features built **exactly as the model was trained** (`build_model_features`): D's final model uses the
   5 quick features, so `add_quick_features` (`src/features/quick_features.py`) is applied to the candidate table;
   a model with any other feature list gets Person C's `add_features`. The route is never passed in;
6. `model.predict_proba(feats[meta["features"]])` = probability that each candidate is the true arc;
7. classifier-only answer = highest-probability candidate per point;
8. HMM + Viterbi (`decode_trip`, Person B) with the parameters in `results/hmm_chosen_params.json`;
9. **only if a `.route` file exists:** reference labels (`label_points`, the same "v2" labels the model was
   trained on) are computed **server-side** to score both answers (Top-1, plus route continuity, which needs no
   labels). They are deleted before returning, so they never reach the output.

Output = exactly the contract dict: `trip_id, n_points, gps [[lat, lon]], predicted_arcs [{obs_idx, arc_id}]`
(−1 = no candidate), `predicted_route_coords` (one `[[lat, lon], [lat, lon]]` polyline per distinct consecutive
predicted arc) and `metrics` (`top1_classifier, top1_hmm, route_continuity_classifier, route_continuity_hmm,
n_labelled_points, runtime_s`; `None` where not computable). NaN is turned into `None` so the JSON is valid.

## 4. API (`api/main.py`)

* **Startup:** model + meta + HMM parameters loaded **once** (FastAPI lifespan), not per request.
* `GET /trips` → `{"trips": [...]}` = sub-folders of `EVAL_DIR`, sorted.
* `POST /predict` `{"trip_id": "..."}` → 404 if the id is not one of those folders. This whitelist also blocks
  path tricks like `../train/x`. Otherwise it returns `predict_trip`. Errors are logged with the full traceback
  on Laptop 1 and returned as a clean 500.
* `GET /health` → model name, number of features, folder, number of trips.
* **Logging:** Python `logging` to stdout, INFO. One line per request: time, client IP, trip_id, points,
  runtime and every metric. Uvicorn's access log adds the HTTP status line. This is what the evaluator sees.
* **CORS** open to all origins, because the UI runs on another laptop (browser pages may only call another
  machine's API if that API allows it).
* `def predict` (not `async def`): the heavy NumPy work runs in FastAPI's worker threads, so the server keeps
  answering other requests.

Environment variables: `EVAL_DIR` (default `data/raw/Student folder/student_test`), `MODEL_PATH` (default
`artifacts/models/final_model.joblib`), `META_PATH` (default `<model>_meta.json`). Relative paths are taken from
the repo root.

### Run it (from the repo root)

PowerShell (Windows):
```
$env:EVAL_DIR = "data/raw/Student folder/student_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
bash: `EVAL_DIR="data/raw/Student folder/student_test" uvicorn api.main:app --host 0.0.0.0 --port 8000`

The API loads D's `artifacts/models/final_model.joblib` by default. Only if it is missing, train an interim
model for testing and point the API at it:
```
python scripts/train_interim_model.py
$env:MODEL_PATH = "artifacts/models/interim_model.joblib"
```

### Test it (second terminal, API running)
```
curl http://localhost:8000/trips
python scripts/test_api.py            # all trips -> results/api_test_runtimes.csv, ALL API CHECKS PASSED
```
PowerShell's `curl` is an alias, so use `curl.exe` there. For a POST:
`curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\":\"00000052\"}"`
From the UI laptop, use `http://<laptop-1-IP>:8000`, and allow port 8000 through the Windows firewall on Laptop 1.

## Viva questions you may get

* *Why compute features the same way in training and in the API?* Training/serving skew: if the API built
  features differently, the model would see different inputs than it learned from. Both paths call the same
  `clean_track` and `generate_candidates`, and the same feature builder the model was trained with
  (chosen from `meta["features"]`).
* *How do you guarantee no ground truth reaches the UI?* `predict_trip` builds the output from predictions only;
  reference labels are computed in a local variable, used for scores, and deleted. `scripts/test_api.py` checks
  that the response has exactly the contract keys.
* *What if the evaluator trip has no `.route`?* Prediction still works; accuracy fields are `None`.
* *Why load the model at startup?* Loading a forest takes seconds; doing it per request would multiply latency.
* *Why a whitelist for trip ids?* It is the only safe way to turn user input into a file path.
* *Why 0.0.0.0?* So the API listens on the network interface, not only on localhost; the UI is on another laptop.
