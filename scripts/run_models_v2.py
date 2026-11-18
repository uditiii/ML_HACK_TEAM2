import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.models.experiment import load_folds, run_experiment
from src.models.models import get_models
from src.features.feature_list import FEATURES

P = Path("data/processed")
train, test = pd.read_parquet(P / "train_table.parquet"), pd.read_parquet(P / "test_table.parquet")
obs_tr, obs_te = pd.read_parquet(P / "obs_train.parquet"), pd.read_parquet(P / "obs_student_test.parquet")
for d in (train, test, obs_tr, obs_te): d["trip_id"] = d["trip_id"].astype(str)
folds = load_folds("results/folds.csv")
for name, fam, model, cfg, mx in get_models():
    if name in ["logreg", "gaussian_nb", "knn", "decision_tree", "random_forest"]:
        run_experiment(model, name, fam, cfg, train, test, obs_tr, obs_te, FEATURES, folds,
                       results_path="results/model_results_v2.csv", max_train_rows=mx)