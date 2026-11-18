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
        tr, va = train_df[fold_of != k], train_df[fold_of == k]
        _, p_va, _, _ = _fit_predict(model, tr, va, features, max_train_rows)
        obs_va = obs_train[obs_train["trip_id"].isin(folds.loc[folds["fold"] == k, "trip_id"])]
        cv["f1"].append(macro_f1(va["label"], p_va))
        cv["ll"].append(safe_log_loss(va["label"], p_va))
        cv["top1"].append(top1_accuracy(va, p_va, obs_va))

    # ---- refit on all 80 train trips, evaluate once on student_test ----
    final, p_te, fit_s, pred_s = _fit_predict(model, train_df, test_df, features, max_train_rows)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": name, "family": family, "config": json.dumps(config),
        "n_features": len(features), "features": json.dumps(list(features)),
        "cv_macro_f1_mean": float(pd.Series(cv["f1"]).mean()),
        "cv_macro_f1_std": float(pd.Series(cv["f1"]).std()),
        "cv_log_loss_mean": float(pd.Series(cv["ll"]).mean()),
        "cv_top1_mean": float(pd.Series(cv["top1"]).mean()),
        "test_macro_f1": float(macro_f1(test_df["label"], p_te)),
        "test_log_loss": float(safe_log_loss(test_df["label"], p_te)),
        "test_top1": float(top1_accuracy(test_df, p_te, obs_test)),
        "train_time_s": round(fit_s, 2), "inference_time_s": round(pred_s, 2),
        "max_train_rows": max_train_rows,
    }
    out = Path(results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    model_dir = Path("data/models")                      # data/ is gitignored: models are not committed
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_dir / f"{name}.joblib")
    print(f"{name:14s} | CV macro-F1 {row['cv_macro_f1_mean']:.4f} +- {row['cv_macro_f1_std']:.4f} "
          f"| CV top1 {row['cv_top1_mean']:.4f} | TEST macro-F1 {row['test_macro_f1']:.4f} "
          f"| TEST log loss {row['test_log_loss']:.4f} | TEST top1 {row['test_top1']:.4f} "
          f"| train {fit_s:.1f}s | infer {pred_s:.2f}s", flush=True)
    return row


def _fit_predict(model, train_df, eval_df, features, max_train_rows=None, seed=42):
    """Fit a fresh copy of the model on train_df (optionally subsampled) and return
    (fitted model, P(label = 1) for eval_df, fit seconds, predict seconds)."""
    if max_train_rows and len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=seed)
    m = clone(model)
    t0 = time.time()
    m.fit(train_df[features], train_df["label"])
    fit_s = time.time() - t0
    t0 = time.time()
    proba = m.predict_proba(eval_df[features])[:, 1]
    return m, proba, fit_s, time.time() - t0
