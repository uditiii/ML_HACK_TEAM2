"""Bayesian hyperparameter optimisation (Optuna, TPE sampler) for Random Forest.
Objective: mean macro-F1 over our 5 fixed GroupKFold folds (results/folds.csv), train trips only."""
import json, sys, time
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, optuna, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.experiment import load_folds, run_experiment

FEATURES = ["dist_m", "proj_t", "dist_rank", "dist_minus_min", "n_candidates"]
SEED = 42
N_TRIALS = 25
TIMEOUT_S = 35 * 60          # stop after 35 min even if trials remain
MAX_FIT_ROWS = 100_000       # per-fold training subsample during the search (speed)


def quick_feats(df):
    g = df.groupby(["trip_id", "obs_idx"])["dist_m"]
    df["dist_rank"] = g.rank(method="min")
    df["dist_minus_min"] = df["dist_m"] - g.transform("min")
    df["n_candidates"] = g.transform("size")
    return df


def load_data():
    P = Path("data/processed")
    train = quick_feats(pd.read_parquet(P / "candidates_train.parquet"))
    test = quick_feats(pd.read_parquet(P / "candidates_student_test.parquet"))
    obs_tr = pd.read_parquet(P / "obs_train.parquet")
    obs_te = pd.read_parquet(P / "obs_student_test.parquet")
    for d in (train, test, obs_tr, obs_te):
        d["trip_id"] = d["trip_id"].astype(str)
    return train, test, obs_tr, obs_te


def make_objective(train, folds):
    fold_of = train["trip_id"].map(dict(zip(folds["trip_id"], folds["fold"]))).to_numpy()
    X, y = train[FEATURES].to_numpy(), train["label"].to_numpy()
    fold_ids = sorted(folds["fold"].unique())
    rng = np.random.RandomState(SEED)
    fit_idx, val_idx = {}, {}
    for k in fold_ids:                                  # fixed rows -> every trial sees the same data
        idx = np.where(fold_of != k)[0]
        fit_idx[k] = rng.choice(idx, MAX_FIT_ROWS, replace=False) if len(idx) > MAX_FIT_ROWS else idx
        val_idx[k] = np.where(fold_of == k)[0]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_int("max_features", 1, len(FEATURES)),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        }
        scores = []
        for step, k in enumerate(fold_ids):
            m = RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)
            m.fit(X[fit_idx[k]], y[fit_idx[k]])
            p = m.predict_proba(X[val_idx[k]])[:, 1]
            scores.append(f1_score(y[val_idx[k]], (p >= 0.5).astype(int), average="macro"))
            trial.report(float(np.mean(scores)), step)   # tell Optuna the running score
            if trial.should_prune():                     # clearly worse than the median trial -> stop early
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", [round(s, 4) for s in scores])
        trial.set_user_attr("cv_std", float(np.std(scores)))
        return float(np.mean(scores))
    return objective


def main():
    train, test, obs_tr, obs_te = load_data()
    folds = load_folds("results/folds.csv")
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize", study_name="rf_macro_f1",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    t0 = time.perf_counter()
    study.optimize(make_objective(train, folds), n_trials=N_TRIALS, timeout=TIMEOUT_S)
    search_time = time.perf_counter() - t0

    study.trials_dataframe().to_csv("results/optuna_trials.csv", index=False)
    best = study.best_params
    print(f"\nBest (search) CV macro-F1: {study.best_value:.4f}  params: {best}  search time: {search_time:.0f}s")

    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    plot_optimization_history(study); plt.savefig("results/figures/optuna_history.png", dpi=150, bbox_inches="tight"); plt.close("all")
    try:
        plot_param_importances(study); plt.savefig("results/figures/optuna_param_importance.png", dpi=150, bbox_inches="tight"); plt.close("all")
    except Exception as e:
        print("param-importance plot skipped:", e)

    # Fair comparison: same protocol as the untuned random_forest row (full train, same folds, same test)
    tuned = RandomForestClassifier(**best, n_jobs=-1, random_state=SEED)
    run_experiment(tuned, "random_forest_tuned", "Ensemble", {**best, "tuning": "optuna_tpe"},
                   train, test, obs_tr, obs_te, FEATURES, folds, results_path="results/model_results.csv")

    final = RandomForestClassifier(**best, n_jobs=-1, random_state=SEED).fit(train[FEATURES], train["label"])
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(final, "artifacts/models/final_model.joblib", compress=3)
    meta = {"model": "RandomForestClassifier", "features": FEATURES, "params": best,
            "search_cv_macro_f1": study.best_value, "n_trials_run": len(study.trials),
            "sampler": "TPE (seed 42)", "pruner": "MedianPruner", "search_fit_rows_per_fold": MAX_FIT_ROWS,
            "search_time_s": round(search_time, 1), "trained_on": "train (all 80 trips, every 5th point)",
            "label_method": "v2", "date": datetime.now().isoformat()}
    json.dump(meta, open("artifacts/models/final_model_meta.json", "w"), indent=2)
    print("Saved artifacts/models/final_model.joblib + final_model_meta.json")


if __name__ == "__main__":
    main()