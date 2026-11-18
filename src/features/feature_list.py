# PROVISIONAL feature list (all engineered features) so downstream code can import it right away.
# notebooks/03_feature_selection.ipynb OVERWRITES this file from the TRAIN table once it has been run
# on the real train_table.parquet. Do not edit by hand after that.

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
    "proj_t",
]

DROPPED = []
