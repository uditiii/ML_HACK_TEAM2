# Final model features. Auto-written by notebooks/03_feature_selection.ipynb from the TRAIN table only.
# Rule: drop a feature only if |Spearman rho| > 0.9 with a stronger feature AND it is below the median on
# f_score, mutual_info and rf_importance. Sample seed 42, <= 200k stratified rows.

FEATURES = [
    "dist_m",
    "dist_rank",
    "dist_minus_min",
    "n_candidates",
    "arc_length_m",
    "gps_bearing",
    "arc_bearing",
    "heading_diff_deg",
    "heading_rank",
    "speed_mps",
    "heading_valid",
    "proj_t"
]

DROPPED = []
