# Feature selection (Person C)

**Data:** train table only (80 training trips). Student-test and evaluator data were never used for any selection decision.
**Leakage rule:** features use only the cleaned GPS track and the road network (nodes, arcs). They never use the `.route` file, `label`, `true_arc` or `has_true_candidate`; using the answer as an input would be label leakage.

**Method:** Spearman correlation (flag |rho| > 0.9); ANOVA F (`f_classif`); mutual information (`mutual_info_classif`, seed 42, stratified sample <= 200,000 rows); Random-Forest impurity importance (200 trees, depth 12, balanced, seed 42); permutation importance (macro-F1, 20 % trip-grouped hold-out, 5 repeats).

**Decision rule:** drop a feature only if it is redundant (|rho| > 0.9 with a stronger feature) AND below the median on all of f_score, mutual_info and rf_importance. Otherwise keep it.

Placeholders `<...>` are to be filled from `results/feature_selection.csv` and the notebook output.

| Feature | Why it should help pick the right road (plain words) | F | MI | RF imp. | Perm. imp. | Redundant with (rho) | Decision |
|---|---|---|---|---|---|---|---|
| `dist_m` | The true road is usually among the closest to the GPS dot, because GPS error is typically a few metres to tens of metres. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `dist_rank` | Raw distance depends on how noisy the point is; rank says "this is the 1st/2nd/3rd closest road *for this point*". It stays meaningful when the whole point is noisy. Ties share the lowest rank. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `dist_minus_min` | How much further this road is than the best road for the same point. 0 = closest; a large value means a clearly worse match even if the absolute distance looks small. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `n_candidates` | How many roads compete for this point. At a junction or in a dense grid there are many look-alikes (harder); on a lone road there is one (easy). It tells the model how much to trust distance. Same value for all candidates of one point, so it only modulates other features. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `arc_length_m` | Short arcs (junction connectors, slip roads) are visited briefly, so a point landing on one is less likely to be the true road than a long through-road. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `gps_bearing` | The direction the vehicle is actually moving, from the neighbouring GPS points. Raw input for the heading comparison; on its own it is a circular angle (0 and 360 are the same), so it is a weak standalone feature. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `arc_bearing` | The direction the road points (from-node to to-node). Raw input for the heading comparison; on its own it only says which compass direction a road runs. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `heading_diff_deg` | A car drives along a road, not across it, and one-way direction matters. Small angle = road points the way we are going; ~180 = it is the reverse arc of the same street. This is what separates a road from its opposite-direction twin, which have identical distance. 90 (neutral) when stationary. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `heading_rank` | Rank of `heading_diff_deg` among the candidates of the same GPS point (1 = the arc that points most nearly the way the vehicle moves). Like `dist_rank`, it is relative to the competitors at that point, so it stays meaningful whatever the absolute angles are, and it directly separates an arc (rank 1) from its reverse twin. All candidates get rank 1 when the vehicle is stationary (all angles neutral 90). | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `speed_mps` | Speed from the previous to the next point. Tells us if the direction is trustworthy (a moving car) or not (stopped, GPS wobbling), and fast movement favours long through-roads. Same value for all candidates of one point, so it modulates other features. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `heading_valid` | 1 if speed >= 1 m/s. When stopped, the computed direction is just noise; this flag lets the model know that `heading_diff_deg` (set to neutral 90) should be ignored. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |
| `proj_t` | Where along the arc the GPS dot projects (0 = start, 1 = end). Near 0 or 1 the dot is by a junction where neighbouring roads also fit; mid-arc is more decisive. Also useful for ordering consecutive points along a route. | `<f>` | `<mi>` | `<rf>` | `<perm>` | `<partner, rho>` | `<keep/drop>` |

## Flagged pairs (|rho| > 0.9)
`<paste results/feature_corr_flagged_pairs.csv>`

## Final list
`<paste FEATURES from src/features/feature_list.py>`   Dropped: `<DROPPED>`

## Caveats to state honestly
- Univariate scores (F, MI) ignore feature interactions; `heading_diff_deg` is only meaningful together with `heading_valid`.
- `speed_mps`, `heading_valid`, `n_candidates` are constant within one GPS point, so they cannot rank candidates by themselves.
- Impurity importance (RF) favours continuous features; permutation importance on the trip-grouped hold-out is the less biased check.
- Selection used the training trips only; the final judgement on generalisation is the student-test macro-F1 from the model experiments.
