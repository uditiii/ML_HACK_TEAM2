"""Folium map of one trip: raw GPS points, nearby road network, reference route."""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import MAP_BBOX_PAD_M, MAPS_DIR  # noqa: E402
from src.data.geo import EARTH_RADIUS_M  # noqa: E402
from src.data.loader import load_trip  # noqa: E402


def _segments_to_lines(nodes, arcs_df):
    """Return list of [[lat,lon],[lat,lon]] for each arc (Folium wants lat,lon order)."""
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    f, t = arcs_df["from_node"].to_numpy(), arcs_df["to_node"].to_numpy()
    return [[[lat[a], lon[a]], [lat[b], lon[b]]] for a, b in zip(f, t)]


def _route_polyline(nodes, arcs, route):
    """Chain route arcs into polylines; break where consecutive arcs don't share a node."""
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    sel = arcs.loc[route["arc_id"].to_numpy()]
    lines, cur = [], []
    for a, b in zip(sel["from_node"].to_numpy(), sel["to_node"].to_numpy()):
        if cur and cur[-1] != a:
            lines.append(cur)
            cur = []
        if not cur:
            cur = [a]
        cur.append(b)
    if cur:
        lines.append(cur)
    return [[[lat[i], lon[i]] for i in ln] for ln in lines]


def trip_map(partition: str, trip_id: str, save: bool = True, max_points: int = 5000) -> folium.Map:
    trip = load_trip(partition, trip_id)
    track = trip["track"].dropna()
    nodes, arcs, route = trip["nodes"], trip["arcs"], trip["route"]

    # GPS bounding box, padded by MAP_BBOX_PAD_M metres
    lat_min, lat_max = track["lat"].min(), track["lat"].max()
    lon_min, lon_max = track["lon"].min(), track["lon"].max()
    dlat = np.degrees(MAP_BBOX_PAD_M / EARTH_RADIUS_M)
    dlon = dlat / np.cos(np.radians((lat_min + lat_max) / 2))
    blat = (lat_min - dlat, lat_max + dlat)
    blon = (lon_min - dlon, lon_max + dlon)

    # Filter arcs BEFORE drawing: keep an arc if either endpoint is inside the padded box
    inside = (nodes["lon"].between(*blon) & nodes["lat"].between(*blat)).to_numpy()
    keep = inside[arcs["from_node"].to_numpy()] | inside[arcs["to_node"].to_numpy()]
    arcs_box = arcs.loc[keep]

    m = folium.Map(tiles="OpenStreetMap", control_scale=True)

    g_road = folium.FeatureGroup(name=f"Road network ({len(arcs_box):,} arcs in bbox+{MAP_BBOX_PAD_M:.0f} m)")
    folium.PolyLine(_segments_to_lines(nodes, arcs_box), color="#888888", weight=1, opacity=0.6).add_to(g_road)
    g_road.add_to(m)

    if route is not None:
        g_route = folium.FeatureGroup(name=f"Reference route ({len(route)} arcs)")
        for line in _route_polyline(nodes, arcs, route):
            folium.PolyLine(line, color="#1a9850", weight=6, opacity=0.8).add_to(g_route)
        g_route.add_to(m)

    g_gps = folium.FeatureGroup(name=f"Raw GPS ({len(track):,} pts)")
    step = max(1, len(track) // max_points)   # thin very long tracks so the HTML stays light
    for la, lo, t in track[["lat", "lon", "t"]].to_numpy()[::step]:
        folium.CircleMarker([la, lo], radius=2, color="#2166ac", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={t:.0f}").add_to(g_gps)
    g_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

    legend = f"""
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;
                border:1px solid #999;border-radius:6px;font:13px sans-serif;line-height:1.6">
      <b>Trip {partition}/{trip_id}</b><br>
      <span style="color:#2166ac">&#9679;</span> Raw GPS points{' (every %d)' % step if step > 1 else ''}<br>
      <span style="display:inline-block;width:18px;border-top:2px solid #888"></span> Road arcs (bbox + {MAP_BBOX_PAD_M:.0f} m)<br>
      <span style="display:inline-block;width:18px;border-top:5px solid #1a9850"></span> Reference route
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    if save:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        m.save(str(MAPS_DIR / f"trip_{trip_id}.html"))
    return m


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "train"
    tid = sys.argv[2] if len(sys.argv) > 2 else "00000001"
    trip_map(part, tid)
    print("saved", MAPS_DIR / f"trip_{tid}.html")
