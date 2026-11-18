"""PathSnap FastAPI backend (Person C). Runs on Laptop 1.

Endpoints
  GET  /health                -> model + data folder status
  GET  /trips                 -> {"trips": [...]} = trip folders in EVAL_DIR
  POST /predict {"trip_id"}   -> predict_trip() output: predictions + scores, NO reference arcs/route

Environment variables
  EVAL_DIR    folder containing <trip_id>/ sub-folders  (default: data/raw/Student folder/student_test)
  MODEL_PATH  fitted model                             (default: artifacts/models/final_model.joblib)
  META_PATH   its meta JSON with "features"            (default: <MODEL_PATH without .joblib>_meta.json)
Relative paths are resolved against the repository root.

Run (from the repo root):
  bash/zsh:    EVAL_DIR="data/raw/Student folder/student_test" uvicorn api.main:app --host 0.0.0.0 --port 8000
  PowerShell:  $env:EVAL_DIR = "data/raw/Student folder/student_test"; uvicorn api.main:app --host 0.0.0.0 --port 8000
Every request is logged to the console (stdout) so the evaluator can see it on Laptop 1.
"""
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.pipeline.predict import load_hmm_params, predict_trip  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")


def _path(env, default):
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


EVAL_DIR = _path("EVAL_DIR", "data/raw/Student folder/student_test")
MODEL_PATH = _path("MODEL_PATH", "artifacts/models/final_model.joblib")
META_PATH = _path("META_PATH", str(MODEL_PATH).replace(".joblib", "_meta.json"))
STATE = {}


@asynccontextmanager
async def lifespan(app):
    """Load the model, its feature list and the HMM parameters ONCE at startup."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(f"model or meta not found: {MODEL_PATH} / {META_PATH} (set MODEL_PATH / META_PATH)")
    if not EVAL_DIR.is_dir():
        raise RuntimeError(f"EVAL_DIR is not a folder: {EVAL_DIR}")
    STATE["model"] = joblib.load(MODEL_PATH)
    STATE["meta"] = json.loads(META_PATH.read_text())
    STATE["hmm_params"] = load_hmm_params()
    log.info("model loaded: %s | features (%d): %s", MODEL_PATH.name, len(STATE["meta"]["features"]),
             STATE["meta"]["features"])
    log.info("HMM params: %s", STATE["hmm_params"])
    log.info("EVAL_DIR: %s | %d trips", EVAL_DIR, len(list_eval_trips()))
    yield
    STATE.clear()


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def list_eval_trips():
    """Trip ids = sub-folder names of EVAL_DIR, sorted."""
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir())


def _client(request: Request):
    return request.client.host if request.client else "unknown"


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "n_features": len(STATE["meta"]["features"]),
            "eval_dir": str(EVAL_DIR), "n_trips": len(list_eval_trips())}


@app.get("/trips")
def trips(request: Request):
    t = list_eval_trips()
    log.info("GET /trips | client=%s | %d trips", _client(request), len(t))
    return {"trips": t}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    """Run the whole pipeline for one trip. Plain `def`: FastAPI runs it in a worker thread."""
    client = _client(request)
    if req.trip_id not in list_eval_trips():          # whitelist: also blocks paths like "../x"
        log.warning("POST /predict | client=%s | trip_id=%s | 404 unknown trip", client, req.trip_id)
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id!r}")
    try:
        out = predict_trip(EVAL_DIR, req.trip_id, STATE["model"], STATE["meta"], use_hmm=True,
                           hmm_params=STATE["hmm_params"])
    except Exception as e:  # noqa: BLE001 -- log the full error on Laptop 1, return a clean 500
        log.exception("POST /predict | client=%s | trip_id=%s | FAILED", client, req.trip_id)
        raise HTTPException(status_code=500, detail=f"prediction failed: {type(e).__name__}: {e}") from e
    m = out["metrics"]
    log.info("POST /predict | client=%s | trip_id=%s | points=%d | runtime=%.3fs | top1_clf=%s | top1_hmm=%s | "
             "cont_clf=%s | cont_hmm=%s | labelled=%s", client, req.trip_id, out["n_points"], m["runtime_s"],
             m["top1_classifier"], m["top1_hmm"], m["route_continuity_classifier"], m["route_continuity_hmm"],
             m["n_labelled_points"])
    return out
