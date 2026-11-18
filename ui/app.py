"""PathSnap Streamlit UI.

Run (Laptop 2, API on Laptop 1):
    API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
Windows PowerShell:
    $env:API_URL="http://<laptop1-ip>:8000"; streamlit run ui/app.py

The UI only ever sends a trip id. It never requests, receives or draws ground-truth arcs
or the reference route - those stay on the server.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualisation.maps import route_map  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TRIPS_TIMEOUT_S = 10
PREDICT_TIMEOUT_S = 300

# metric key in the API response (team contract) -> (label, is_fraction)
METRICS = {
    "top1_classifier": ("Top-1 accuracy, classifier", True),
    "top1_hmm": ("Top-1 accuracy, HMM", True),
    "route_continuity_classifier": ("Route continuity, classifier", True),
    "route_continuity_hmm": ("Route continuity, HMM", True),
    "n_points": ("GPS points", False),
    "runtime_s": ("Runtime (s)", False),
}


def metric_value(res, key):
    """Contract: metrics.route_continuity_* and top-level n_points. Older names (the stub's
    continuity_* and metrics.n_points) are still accepted."""
    m = res.get("metrics") or {}
    for k in (key, key.replace("route_", "")):
        if k in m:
            return m[k]
    return res.get(key)


def arc_ids(predicted_arcs):
    """Contract: [{"obs_idx": i, "arc_id": a}, ...]; plain ints (stub) also accepted. -1 = no candidate."""
    out = []
    for a in predicted_arcs or []:
        a = a.get("arc_id") if isinstance(a, dict) else a
        if a is not None and int(a) >= 0:
            out.append(int(a))
    return out

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: route reconstruction from GPS")
st.caption(f"API: {API_URL}")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_trips(url: str):
    r = requests.get(f"{url}/trips", timeout=TRIPS_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    trips = data["trips"] if isinstance(data, dict) else data
    return [str(t) for t in trips], bool(isinstance(data, dict) and data.get("stub"))


def fmt(value, is_fraction):
    if value is None:
        return "N/A"
    if is_fraction:
        return f"{100 * float(value):.1f}%"
    return f"{value:,}" if isinstance(value, int) else f"{float(value):.2f}"


# ------------------------------------------------------------------ trip list
try:
    trips, is_stub = fetch_trips(API_URL)
except requests.exceptions.RequestException as e:
    st.error(f"Cannot reach the API at **{API_URL}/trips**.\n\n"
             f"Check the API is running on Laptop 1, both laptops are on the same network, "
             f"and API_URL is correct.\n\nDetails: `{type(e).__name__}: {e}`")
    if st.button("Retry"):
        fetch_trips.clear()
        st.rerun()
    st.stop()

if is_stub:
    st.warning("Connected to the STUB API: all results are FAKE. Do not use for the demo.")
if not trips:
    st.error("The API returned an empty trip list.")
    st.stop()

col_sel, col_btn = st.columns([3, 1])
trip_id = col_sel.selectbox("Trip", trips)
run = col_btn.button("Run prediction", type="primary")

# ------------------------------------------------------------------ prediction
if run:
    with st.spinner(f"Predicting route for trip {trip_id} (can take a few minutes)..."):
        try:
            r = requests.post(f"{API_URL}/predict", json={"trip_id": trip_id}, timeout=PREDICT_TIMEOUT_S)
            r.raise_for_status()
            st.session_state["result"] = r.json()
        except requests.exceptions.Timeout:
            st.error(f"The API did not answer within {PREDICT_TIMEOUT_S} s ({API_URL}/predict).")
        except requests.exceptions.HTTPError as e:
            st.error(f"The API returned an error: {e}\n\n{r.text[:500]}")
        except requests.exceptions.RequestException as e:
            st.error(f"Cannot reach {API_URL}/predict: `{type(e).__name__}: {e}`")

res = st.session_state.get("result")
if res and res.get("trip_id") == trip_id:
    st.subheader(f"Results for trip {trip_id}")
    cols = st.columns(len(METRICS))
    for col, (key, (label, frac)) in zip(cols, METRICS.items()):
        col.metric(label, fmt(metric_value(res, key), frac))

    gps = res.get("gps") or []
    pred_coords = res.get("predicted_route_coords") or []
    if gps:
        m = route_map(gps, pred_coords, title=f"Trip {trip_id}: GPS vs predicted route", height=580)
        components.html(m._repr_html_(), height=600)
    else:
        st.info("The API response has no GPS points to draw.")

    raw = arc_ids(res.get("predicted_arcs"))
    arcs = [a for i, a in enumerate(raw) if i == 0 or a != raw[i - 1]]   # collapse per-point repeats
    st.markdown(f"**First 20 predicted arcs** (of {len(arcs):,} distinct consecutive arcs)")
    st.dataframe(pd.DataFrame({"step": range(1, min(20, len(arcs)) + 1), "arc_id": arcs[:20]}),
                 hide_index=True)
elif res:
    st.info("Trip changed: press Run prediction.")
