"""The 5 features used by the final model (see artifacts/models/final_model_meta.json).
Input: candidate table with trip_id, obs_idx, dist_m, proj_t. Used for training, HMM experiments and the API."""

FEATURES = ["dist_m", "proj_t", "dist_rank", "dist_minus_min", "n_candidates"]


def add_quick_features(df):
    g = df.groupby(["trip_id", "obs_idx"])["dist_m"]
    df["dist_rank"] = g.rank(method="min")              # 1 = closest candidate of this GPS point
    df["dist_minus_min"] = df["dist_m"] - g.transform("min")
    df["n_candidates"] = g.transform("size")
    return df