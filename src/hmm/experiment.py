"""HMM experiment helpers (Person B): classifier-only vs classifier + HMM/Viterbi.

Pipeline per trip (the same steps the API runs on an evaluator trip):
  load + clean track -> candidates for EVERY cleaned point -> features -> classifier probability
  -> classifier-only argmax and HMM/Viterbi decoding.
Labels (label_points v2, from the .route file) are computed ONLY to score the predictions;
they never reach features, the classifier or the decoder.

Model: artifacts/models/final_model.joblib + final_model_meta.json if they exist; otherwise the
Phase 1 random forest (data/models/random_forest.joblib, written by scripts/run_quick_models.py)
with the 5 Phase 1 features. Which one was used is printed and saved with every result.
Features are built the way the model was trained: the 5 quick features with add_quick_features
(src/features/quick_features.py), any other feature list with Person C's add_features.
"""
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from config import RADIUS_M
from scripts.build_candidates import load_projected
from src.candidates.generate import generate_candidates
from src.candidates.labels import label_points
from src.hmm.viterbi import (DEFAULT_PARAMS, RoadGraph, classifier_only, continuity_counts,
                             decode_prepared, prepare_chains, top1_counts)
from src.features.quick_features import FEATURES as QUICK_FEATURES
from src.features.quick_features import add_quick_features
from src.models.experiment import load_folds

ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL = ROOT / "artifacts" / "models" / "final_model.joblib"
FINAL_META = ROOT / "artifacts" / "models" / "final_model_meta.json"
PHASE1_MODEL = ROOT / "data" / "models" / "random_forest.joblib"
PHASE1_FEATURES = list(QUICK_FEATURES)

GRID = {"p_stay": [0.5, 0.6, 0.8], "HOP_PENALTY": [0.5, 1.0, 2.0], "UTURN_PENALTY": [1.0, 3.0, 6.0],
        "EMISSION_WEIGHT": [1.0, 0.5, 0.25]}


# --------------------------------------------------------------------------------------------- #
# Model and features
# --------------------------------------------------------------------------------------------- #
def load_scorer():
    """The classifier used to score candidates, its feature list, and where it came from."""
    if FINAL_MODEL.exists() and FINAL_META.exists():
        meta = json.loads(FINAL_META.read_text())
        print(f"Using FINAL model {FINAL_MODEL.relative_to(ROOT)} with features {meta['features']}")
        return {"kind": "final", "model": joblib.load(FINAL_MODEL), "features": list(meta["features"]),
                "source": str(FINAL_MODEL.relative_to(ROOT))}
    if PHASE1_MODEL.exists():
        print("WARNING: final model not found -> using the PHASE 1 random forest "
              f"({PHASE1_MODEL.relative_to(ROOT)}) with the 5 Phase 1 distance-only features.")
        return {"kind": "phase1", "model": joblib.load(PHASE1_MODEL), "features": PHASE1_FEATURES,
                "source": str(PHASE1_MODEL.relative_to(ROOT))}
    raise FileNotFoundError("No model found. Either put the final model in artifacts/models/, or train the "
                            "Phase 1 forest: python scripts/run_quick_models.py --models random_forest")


def uses_quick_features(scorer):
    """True if every model feature is one of the 5 quick features (D's final model, Phase 1 forest)."""
    return set(scorer["features"]) <= set(QUICK_FEATURES)


def add_model_features(cand, trip, clean, scorer):
    """Feature columns for one trip's candidates, built the SAME way the model was trained.

    5 quick features -> src/features/quick_features.add_quick_features (candidate table only);
    anything else    -> Person C's add_features (track + road network). Never the route or labels.
    """
    if uses_quick_features(scorer):
        return add_quick_features(cand.copy())
    from src.features.build_features import add_features          # Person C
    return add_features(cand, {"nodes": trip["nodes"], "arcs": trip["arcs"]}, clean)


def training_table(scorer):
    """The table the scorer was trained on (train trips, every 5th point), with its features."""
    proc = ROOT / "data" / "processed"
    if uses_quick_features(scorer):
        df = add_quick_features(pd.read_parquet(proc / "candidates_train.parquet"))
    else:
        df = pd.read_parquet(proc / "train_table.parquet")
    df["trip_id"] = df["trip_id"].astype(str)
    return df


# --------------------------------------------------------------------------------------------- #
# Trips
# --------------------------------------------------------------------------------------------- #
def load_trip_for_hmm(partition, trip_id, scorer):
    """Candidates + features for EVERY cleaned point of one trip, plus reference labels for scoring."""
    trip, clean, geom, _ = load_projected(partition, trip_id)
    obs = np.arange(len(clean))
    cand = generate_candidates(trip_id, clean["x"].to_numpy(), clean["y"].to_numpy(), obs, geom, RADIUS_M)
    feats = add_model_features(cand, trip, clean, scorer)
    true_arc = label_points(clean, geom, trip["route"], method="v2")       # evaluation only
    return {"trip_id": trip_id, "partition": partition, "arcs": trip["arcs"], "clean": clean,
            "cand": cand, "X": feats[scorer["features"]],
            "truth": pd.DataFrame({"obs_idx": obs, "true_arc": true_arc})}


def score_trips(trips, scorer, out_of_fold=False):
    """Add classifier probabilities ("proba") to each trip.

    out_of_fold=True (train trips): each trip is scored by a copy of the model refitted WITHOUT that
    trip's CV fold (results/folds.csv), so HMM parameters are tuned on honest, not in-sample, scores.
    """
    if not out_of_fold:
        for d in trips:
            d["proba"] = scorer["model"].predict_proba(d["X"])[:, 1]
        return trips
    folds = load_folds(ROOT / "results" / "folds.csv")
    fold_of = dict(zip(folds["trip_id"], folds["fold"]))
    table = training_table(scorer)
    table_fold = table["trip_id"].map(fold_of)
    for k in sorted({fold_of[d["trip_id"]] for d in trips}):
        t0 = time.time()
        fit = table[table_fold != k]
        m = clone(scorer["model"]).fit(fit[scorer["features"]], fit["label"])
        for d in trips:
            if fold_of[d["trip_id"]] == k:
                d["proba"] = m.predict_proba(d["X"])[:, 1]
        print(f"  out-of-fold model for fold {k} fitted in {time.time() - t0:.1f}s", flush=True)
    return trips


# --------------------------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------------------------- #
def evaluate_trip(d, params, prepared=None, graph=None):
    """Classifier-only vs HMM on one scored trip -> one row of counts and rates."""
    graph = graph if graph is not None else RoadGraph(d["arcs"])
    t0 = time.time()
    prepared = prepared if prepared is not None else prepare_chains(d["cand"], d["proba"], d["arcs"], d["clean"],
                                                                     params, graph)
    hmm = decode_prepared(prepared, params)
    hmm_s = time.time() - t0
    clf = classifier_only(d["cand"], d["proba"], d["clean"])
    row = {"trip_id": d["trip_id"], "n_points": len(d["clean"])}
    for name, pred in (("clf", clf), ("hmm", hmm)):
        ok, n = top1_counts(pred, d["truth"])
        c_ok, c_n = continuity_counts(pred["pred_arc"].to_numpy(), d["arcs"])
        m = pred.merge(d["truth"], on="obs_idx")
        m = m[m["true_arc"] >= 0]
        rev = (m["pred_arc"].to_numpy() == graph.reverse[m["true_arc"].to_numpy()]).sum()
        row.update({f"top1_{name}": ok / n if n else np.nan, f"cont_{name}": c_ok / c_n if c_n else np.nan,
                    f"correct_{name}": ok, f"cont_ok_{name}": c_ok, f"cont_n_{name}": c_n,
                    f"reverse_err_{name}": int(rev)})
    row["n_labelled"] = n
    row["n_segments"] = int(d["clean"]["segment_id"].nunique())
    row["mean_candidates"] = float(d["cand"].groupby("obs_idx").size().mean()) if len(d["cand"]) else 0.0
    row["pct_points_changed_by_hmm"] = float(100 * (hmm["pred_arc"].to_numpy() != clf["pred_arc"].to_numpy()).mean())
    row["hmm_decode_s"] = round(hmm_s, 3)
    return row


def overall_row(df):
    """Pooled (observation-level) totals over all trips."""
    tot = {"trip_id": "OVERALL", "n_points": int(df["n_points"].sum()), "n_labelled": int(df["n_labelled"].sum())}
    for name in ("clf", "hmm"):
        tot[f"correct_{name}"] = int(df[f"correct_{name}"].sum())
        tot[f"top1_{name}"] = tot[f"correct_{name}"] / tot["n_labelled"]
        tot[f"cont_ok_{name}"] = int(df[f"cont_ok_{name}"].sum())
        tot[f"cont_n_{name}"] = int(df[f"cont_n_{name}"].sum())
        tot[f"cont_{name}"] = tot[f"cont_ok_{name}"] / tot[f"cont_n_{name}"] if tot[f"cont_n_{name}"] else np.nan
        tot[f"reverse_err_{name}"] = int(df[f"reverse_err_{name}"].sum())
    tot["hmm_decode_s"] = round(float(df["hmm_decode_s"].sum()), 3)
    return tot


COMPARISON_COLS = ["trip_id", "n_points", "top1_clf", "top1_hmm", "cont_clf", "cont_hmm", "n_labelled",
                   "correct_clf", "correct_hmm", "reverse_err_clf", "reverse_err_hmm", "cont_ok_clf", "cont_n_clf",
                   "cont_ok_hmm", "cont_n_hmm", "n_segments", "mean_candidates", "pct_points_changed_by_hmm",
                   "hmm_decode_s"]


def compare(trips, params):
    """Per-trip classifier-only vs HMM table plus an OVERALL (pooled) row."""
    rows = [evaluate_trip(d, params) for d in trips]
    df = pd.DataFrame(rows)
    df = pd.concat([df, pd.DataFrame([overall_row(df)])], ignore_index=True)
    return df[[c for c in COMPARISON_COLS if c in df.columns]]


def param_search(trips, base=DEFAULT_PARAMS, grid=GRID):
    """Grid search of the HMM penalties on scored TRAIN trips; graph work is done once per trip.

    p_move is set to 1 - p_stay so the two transition probabilities always sum to 1.
    Selection metric: pooled observation-level Top-1 (correct / labelled over all points).
    """
    graphs = [RoadGraph(d["arcs"]) for d in trips]
    prepared = [prepare_chains(d["cand"], d["proba"], d["arcs"], d["clean"], base, g) for d, g in zip(trips, graphs)]
    rows = []
    base_rows = [evaluate_trip(d, base, p, g) for d, p, g in zip(trips, prepared, graphs)]
    clf = overall_row(pd.DataFrame(base_rows))
    rows.append({"config": "classifier_only", "top1": clf["top1_clf"], "cont": clf["cont_clf"],
                 "top1_trip_mean": float(pd.DataFrame(base_rows)["top1_clf"].mean())})
    names = list(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        params = {**base, **dict(zip(names, values))}
        params["p_move"] = round(1.0 - params["p_stay"], 10)
        per_trip = pd.DataFrame([evaluate_trip(d, params, p, g) for d, p, g in zip(trips, prepared, graphs)])
        o = overall_row(per_trip)
        rows.append({"config": "hmm", **{n: params[n] for n in names}, "p_move": params["p_move"],
                     "top1": o["top1_hmm"], "cont": o["cont_hmm"],
                     "top1_trip_mean": float(per_trip["top1_hmm"].mean())})
    res = pd.DataFrame(rows)
    hmm_rows = res[res["config"] == "hmm"]
    best = hmm_rows.loc[hmm_rows["top1"].idxmax()]                  # first best in grid order on ties
    chosen = {**base, **{n: float(best[n]) for n in names}, "p_move": float(best["p_move"])}
    return res, chosen


# --------------------------------------------------------------------------------------------- #
# Diagnostics for the report: where does the HMM still go wrong, and what does it look like?
# --------------------------------------------------------------------------------------------- #
def error_runs(d, pred, top=3):
    """Longest runs of consecutive wrong HMM points, with how long they last and how far the GPS spreads.

    A long run on ONE arc whose GPS points stay inside a small box means the vehicle was (nearly)
    standing still or circling: no arc change -> no transition evidence for the direction.
    """
    graph = RoadGraph(d["arcs"])
    m = pred.merge(d["truth"], on="obs_idx").sort_values("obs_idx")
    clean = d["clean"].set_index("obs_idx")
    lab = m["true_arc"].to_numpy() >= 0
    wrong = lab & (m["pred_arc"].to_numpy() != m["true_arc"].to_numpy())
    rev = lab & (m["pred_arc"].to_numpy() == graph.reverse[np.maximum(m["true_arc"].to_numpy(), 0)])
    idx = np.flatnonzero(wrong)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []
    rows = []
    for r in sorted(runs, key=len, reverse=True)[:top]:
        obs = m["obs_idx"].to_numpy()[r]
        c = clean.loc[obs]
        rows.append({"trip_id": d["trip_id"], "first_obs": int(obs[0]), "last_obs": int(obs[-1]), "n_points": len(r),
                     "duration_s": float(c["t"].iloc[-1] - c["t"].iloc[0]),
                     "gps_box_m": f"{c['x'].max() - c['x'].min():.0f} x {c['y'].max() - c['y'].min():.0f}",
                     "start_to_end_m": float(np.hypot(c["x"].iloc[-1] - c["x"].iloc[0], c["y"].iloc[-1] - c["y"].iloc[0])),
                     "distinct_pred_arcs": int(m["pred_arc"].iloc[r].nunique()),
                     "share_reverse_twin": float(rev[r].mean())})
    return pd.DataFrame(rows), int(wrong.sum()), int(rev.sum())


def plot_trip_comparison(d, params, path):
    """Side-by-side map (metres) of one trip: GPS points coloured by whether the classifier-only
    (left) and the HMM (right) prediction is correct. Grey lines = road arcs near the track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    _, _, geom, _ = load_projected(d["partition"], d["trip_id"])
    clean = d["clean"]
    pad = 100.0
    box = ((geom[["ax", "bx"]].max(axis=1) >= clean["x"].min() - pad) & (geom[["ax", "bx"]].min(axis=1) <= clean["x"].max() + pad)
           & (geom[["ay", "by"]].max(axis=1) >= clean["y"].min() - pad) & (geom[["ay", "by"]].min(axis=1) <= clean["y"].max() + pad))
    segs = np.stack([geom.loc[box, ["ax", "ay"]].to_numpy(), geom.loc[box, ["bx", "by"]].to_numpy()], axis=1)
    graph = RoadGraph(d["arcs"])
    preds = {"classifier only": classifier_only(d["cand"], d["proba"], clean),
             "classifier + HMM": decode_prepared(prepare_chains(d["cand"], d["proba"], d["arcs"], clean, params, graph),
                                                  params)}
    ratio = (clean["x"].max() - clean["x"].min() + 1) / (clean["y"].max() - clean["y"].min() + 1)
    fig, axes = plt.subplots(1, 2, figsize=(min(18.0, max(9.0, 12 * ratio + 2)), 7), sharex=True, sharey=True)
    for ax, (title, pred) in zip(axes, preds.items()):
        m = pred.merge(d["truth"], on="obs_idx").merge(clean[["obs_idx", "x", "y"]], on="obs_idx")
        lab = m["true_arc"] >= 0
        ok = lab & (m["pred_arc"] == m["true_arc"])
        rev = lab & ~ok & (m["pred_arc"].to_numpy() == graph.reverse[np.maximum(m["true_arc"].to_numpy(), 0)])
        other = lab & ~ok & ~rev
        ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.6, zorder=1))
        for mask, colour, name in ((ok, "tab:green", "correct"), (rev, "tab:orange", "reverse twin of true arc"),
                                   (other, "tab:red", "other wrong arc"), (~lab, "0.4", "unlabelled")):
            ax.scatter(m.loc[mask, "x"], m.loc[mask, "y"], s=4, c=colour, label=f"{name} ({int(mask.sum())})", zorder=2)
        ax.set_title(f"{title}: Top-1 {ok.sum() / max(lab.sum(), 1):.3f}")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, east)")
        ax.legend(loc="best", fontsize=8, markerscale=3)
    axes[0].set_ylabel("y (m, north)")
    fig.suptitle(f"{d['partition']} trip {d['trip_id']}: classifier-only vs HMM (labels: route v2)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
