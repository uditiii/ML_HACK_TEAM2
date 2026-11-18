"""Folium maps for PathSnap.

route_map(...)   generic map builder used by the UI and the report scripts.
                 Takes plain [lat, lon] coordinate lists, so it never needs to load data itself.
trip_map(...)    convenience wrapper: loads one trip and draws GPS + network + reference route
                 (EDA / report use only).

Helper functions turn arc ids into [lat, lon] polylines:
    arcs_to_polylines(nodes, arcs, arc_ids)   chain consecutive arcs into continuous lines
    network_polylines_in_bbox(nodes, arcs, lats, lons, pad_m)   road arcs near the trip

Colour code (same everywhere so the legend is always valid):
    blue  #2a78d6  raw GPS points
    grey  #888888  road network
    green #1a9850  reference route   (offline report maps ONLY - never in the UI)
    red   #d7301f  predicted route   (dashed)
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import folium
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as _cfg  # noqa: E402

EARTH_RADIUS_M = 6_371_008.8
MAP_BBOX_PAD_M = float(getattr(_cfg, "MAP_BBOX_PAD_M", 200.0))
MAPS_DIR = Path(getattr(_cfg, "MAPS_DIR", Path(_cfg.__file__).resolve().parent / "results" / "maps"))

GPS_COLOUR = "#2a78d6"
NETWORK_COLOUR = "#888888"
REFERENCE_COLOUR = "#1a9850"
PREDICTED_COLOUR = "#d7301f"


# --------------------------------------------------------------------------- helpers
def _as_polylines(coords):
    """Accept either one polyline [[lat,lon],...] or many [[[lat,lon],...], ...].
    Always return a list of polylines; drop empty ones."""
    if coords is None:
        return []
    coords = list(coords)
    if not coords:
        return []
    first = coords[0]
    # one polyline: first element is a [lat, lon] pair of numbers
    if len(first) == 2 and all(isinstance(v, (int, float, np.floating, np.integer)) for v in first):
        lines = [coords]
    else:
        lines = coords
    out = []
    for ln in lines:
        ln = [[float(p[0]), float(p[1])] for p in ln]
        if len(ln) >= 2:
            out.append(ln)
    return out


def arcs_to_polylines(nodes, arcs, arc_ids):
    """Arc ids -> list of [lat, lon] polylines. Consecutive repeats are collapsed
    (per-observation predictions repeat the same arc many times). A new polyline starts
    wherever two consecutive arcs do not share a node, so breaks stay visible."""
    ids = np.asarray(list(arc_ids), dtype=np.int64)
    if ids.size == 0:
        return []
    ids = ids[np.r_[True, ids[1:] != ids[:-1]]]
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    fr = arcs["from_node"].to_numpy()[ids]
    to = arcs["to_node"].to_numpy()[ids]
    lines, cur = [], []
    for a, b in zip(fr, to):
        if cur and cur[-1] != a:
            lines.append(cur)
            cur = []
        if not cur:
            cur = [a]
        cur.append(b)
    if cur:
        lines.append(cur)
    return [[[float(lat[i]), float(lon[i])] for i in ln] for ln in lines]


def network_polylines_in_bbox(nodes, arcs, lats, lons, pad_m=MAP_BBOX_PAD_M):
    """Road arcs with at least one end inside the GPS bounding box + pad_m metres.
    Filtering happens BEFORE building any Folium object, so huge networks stay fast."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    dlat = np.degrees(pad_m / EARTH_RADIUS_M)
    dlon = dlat / max(np.cos(np.radians(lats.mean())), 1e-6)
    nlon, nlat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    inside = ((nlon >= lons.min() - dlon) & (nlon <= lons.max() + dlon)
              & (nlat >= lats.min() - dlat) & (nlat <= lats.max() + dlat))
    fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    keep = inside[fr] | inside[to]
    fr, to = fr[keep], to[keep]
    return [[[float(nlat[a]), float(nlon[a])], [float(nlat[b]), float(nlon[b])]] for a, b in zip(fr, to)]


# --------------------------------------------------------------------------- main builder
def route_map(gps_latlon, predicted_route_coords=None, reference_route_coords=None,
              network_coords=None, title="", max_points=5000, height=None) -> folium.Map:
    """Build a layered Folium map.

    gps_latlon              [[lat, lon], ...]  raw GPS observations (required)
    predicted_route_coords  one polyline or a list of polylines, [lat, lon] points
    reference_route_coords  same format; ONLY for offline report maps - the UI never passes it
    network_coords          list of 2-point polylines (road arcs near the trip)
    title                   shown at the top of the map
    height                  pixel height for embedding (e.g. Streamlit); None = default
    """
    gps = np.asarray(gps_latlon, dtype=np.float64).reshape(-1, 2)
    gps = gps[np.isfinite(gps).all(axis=1)]
    if len(gps) == 0:
        raise ValueError("route_map: gps_latlon is empty")

    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if height:
        folium.Figure(height=f"{int(height)}px").add_child(m)
    legend_rows = []

    net = _as_polylines(network_coords) if network_coords is not None else []
    if net:
        g = folium.FeatureGroup(name=f"Road network ({len(net):,} arcs)")
        folium.PolyLine(net, color=NETWORK_COLOUR, weight=1, opacity=0.6).add_to(g)
        g.add_to(m)
        legend_rows.append(f'<span style="display:inline-block;width:18px;border-top:2px solid {NETWORK_COLOUR}"></span> Road network')

    ref = _as_polylines(reference_route_coords)
    if ref:
        g = folium.FeatureGroup(name="Reference route")
        for ln in ref:
            folium.PolyLine(ln, color=REFERENCE_COLOUR, weight=7, opacity=0.75).add_to(g)
        g.add_to(m)
        legend_rows.append(f'<span style="display:inline-block;width:18px;border-top:6px solid {REFERENCE_COLOUR}"></span> Reference route')

    pred = _as_polylines(predicted_route_coords)
    if pred:
        g = folium.FeatureGroup(name="Predicted route")
        for ln in pred:
            folium.PolyLine(ln, color=PREDICTED_COLOUR, weight=4, opacity=0.9, dash_array="8 6").add_to(g)
        g.add_to(m)
        legend_rows.append(f'<span style="display:inline-block;width:18px;border-top:3px dashed {PREDICTED_COLOUR}"></span> Predicted route')

    step = max(1, len(gps) // max_points)   # thin very long tracks so the HTML stays light
    g = folium.FeatureGroup(name=f"Raw GPS ({len(gps):,} pts)")
    for la, lo in gps[::step]:
        folium.CircleMarker([la, lo], radius=2, color=GPS_COLOUR, fill=True,
                            fill_opacity=0.8, weight=0).add_to(g)
    g.add_to(m)
    legend_rows.insert(0, f'<span style="color:{GPS_COLOUR}">&#9679;</span> Raw GPS points'
                          + (f" (every {step}th)" if step > 1 else ""))

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[gps[:, 0].min(), gps[:, 1].min()], [gps[:, 0].max(), gps[:, 1].max()]])

    safe_title = html.escape(title)
    if safe_title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;'
            f'background:white;padding:6px 14px;border:1px solid #999;border-radius:6px;'
            f'font:600 15px sans-serif">{safe_title}</div>'))
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:28px;right:10px;z-index:9999;background:white;padding:8px 12px;'
        'border:1px solid #999;border-radius:6px;font:13px sans-serif;line-height:1.7">'
        '<b>Legend</b><br>' + "<br>".join(legend_rows) + "</div>"))
    return m


# --------------------------------------------------------------------------- EDA wrapper
def trip_map(partition: str, trip_id: str, save: bool = True, max_points: int = 5000) -> folium.Map:
    """GPS + nearby network + reference route for one trip (EDA / report only)."""
    from src.data.loader import load_trip
    trip = load_trip(partition, trip_id)
    tr = trip["track"].dropna()
    gps = tr[["lat", "lon"]].to_numpy()
    net = network_polylines_in_bbox(trip["nodes"], trip["arcs"], tr["lat"], tr["lon"])
    ref = None
    if trip["route"] is not None:
        ref = arcs_to_polylines(trip["nodes"], trip["arcs"], trip["route"]["arc_id"])
    m = route_map(gps, None, ref, net, title=f"Trip {partition}/{trip_id}", max_points=max_points)
    if save:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        m.save(str(MAPS_DIR / f"trip_{trip_id}.html"))
    return m


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "train"
    tid = sys.argv[2] if len(sys.argv) > 2 else "00000001"
    trip_map(part, tid)
    print("saved", MAPS_DIR / f"trip_{tid}.html")
