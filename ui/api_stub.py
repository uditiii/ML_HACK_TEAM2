"""
##########################################################################
#  STUB API - FOR TESTING THE UI ONLY. NEVER USE IN THE DEMO.           #
#  Returns FAKE data with the same JSON shape as the real API.          #
##########################################################################

Run:   uvicorn ui.api_stub:app --port 8000
Then:  streamlit run ui/app.py

Response contract the real API must follow (the UI reads exactly these keys):
GET  /trips    -> {"trips": ["00000000", ...]}
POST /predict  body {"trip_id": "..."} ->
  {
    "trip_id": str,
    "gps": [[lat, lon], ...],                        # observations sent to the model
    "predicted_arcs": [int, ...],                    # predicted arc ids, in order
    "predicted_route_coords": [[lat, lon], ...]      # one polyline, OR a list of polylines
    "metrics": {
        "top1_classifier": float, "top1_hmm": float,                 # 0-1
        "continuity_classifier": float, "continuity_hmm": float,     # 0-1
        "n_points": int, "runtime_s": float
    }
  }
No ground-truth arcs or reference route are ever returned.
"""
import math
import random
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PathSnap STUB API (fake data)")
FAKE_TRIPS = ["STUB_0001", "STUB_0002", "STUB_0003"]


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/trips")
def trips():
    return {"trips": FAKE_TRIPS, "stub": True}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in FAKE_TRIPS:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    rng = random.Random(req.trip_id)
    lat0, lon0, n = 49.87, 8.65, 300
    road = [[lat0 + 0.00004 * i, lon0 + 0.003 * math.sin(i / 60)] for i in range(n)]
    gps = [[la + rng.gauss(0, 0.00005), lo + rng.gauss(0, 0.00007)] for la, lo in road]
    pred = road[::10]
    return {
        "trip_id": req.trip_id,
        "stub": True,
        "gps": gps,
        "predicted_arcs": list(range(1000, 1000 + len(pred) - 1)),
        "predicted_route_coords": pred,
        "metrics": {"top1_classifier": 0.71, "top1_hmm": 0.84,
                    "continuity_classifier": 0.62, "continuity_hmm": 0.97,
                    "n_points": n, "runtime_s": round(time.time() - t0, 3)},
    }
