import json, time
from datetime import datetime
from pathlib import Path
import joblib, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from src.models.metrics import macro_f1, safe_log_loss, top1_accuracy


def make_folds(train_df, n_splits=5, path="results/folds.csv"):
    """Assign each train trip to one of 5 folds (GroupKFold by trip), save once, reuse everywhere."""
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (_, val_idx) in enumerate(gkf.split(train_df, train_df["label"], groups=train_df["trip_id"])):
        for trip in train_df["trip_id"].iloc[val_idx].unique():
            rows.append((trip, fold))
    folds = pd.DataFrame(rows, columns=["trip_id", "fold"])
    folds.to_csv(path, index=False)
    return folds


def load_folds(path="results/folds.csv"):
    return pd.read_csv(path, dtype={"trip_id": str})   # str! otherwise "00000001" becomes 1


def run_experiment(model, name, family, config, train_df, test_df, obs_train, obs_test,
                   features, folds, results_path="results/model_results.csv", max_train_rows=None):
    obs_train = obs_train[obs_train["obs_idx"] % 5 == 0]          # train rows are every 5th point
    fold_of = train_df["trip_id"].map(dict(zip(folds["trip_id"], folds["fold"])))

    cv = {"f1": [], "ll": [], "top1": []}
    for k in sorted(folds["fold"].unique()):                     # ---- 5-fold group CV ----
        tr, va = train_df[fold_of != k],