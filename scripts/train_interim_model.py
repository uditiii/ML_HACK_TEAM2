"""Train an INTERIM model so the pipeline and API can be tested before the tuned final model exists.

NOT the final model: same Random Forest configuration as Phase 1 (src/models/models.py), fitted on
data/processed/train_table.parquet (80 train trips) with the features in src/features/feature_list.py.
Writes artifacts/models/interim_model.joblib + interim_model_meta.json (never final_model.*).

Usage (from anywhere):  python scripts/train_interim_model.py
Use it with the API:    set MODEL_PATH=artifacts/models/interim_model.joblib (see docs/person_c_pipeline_api.md)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.features.feature_list import FEATURES
from src.models.models import get_models

out_dir = ROOT / "artifacts" / "models"
out_dir.mkdir(parents=True, exist_ok=True)
train = pd.read_parquet(ROOT / "data" / "processed" / "train_table.parquet")
name, family, model, config, _ = next(m for m in get_models() if m[0] == "random_forest")

t0 = time.time()
model.fit(train[FEATURES], train["label"])
fit_s = time.time() - t0
joblib.dump(model, out_dir / "interim_model.joblib")
meta = {"features": list(FEATURES), "params": config, "model": name, "cv_macro_f1": None,
        "trained_on": "train", "note": "INTERIM model for pipeline/API testing, not the tuned final model"}
(out_dir / "interim_model_meta.json").write_text(json.dumps(meta, indent=2))
print(f"trained {name} on {len(train)} rows x {len(FEATURES)} features in {fit_s:.1f}s")
print(f"saved {out_dir / 'interim_model.joblib'} and interim_model_meta.json")
