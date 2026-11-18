"""Distance utilities.

Lon/lat are ANGLES, not metres: 1 deg of longitude at lat 49.9 N is ~71.7 km but 1 deg of
latitude is ~111.2 km, so Euclidean distance in degrees is distorted ~1.55x east-west.
We provide:
  * haversine_m      - great-circle distance in metres (exact enough for any scale)
  * make_projector   - local equirectangular projection to x/y metres around a reference
                       point; error < 0.1% within a few km, i.e. far below GPS noise (~5-10 m),
                       and it lets us use fast planar maths (point-to-segment distance etc.).
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # IUGG mean Earth radius


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres. Accepts scalars or numpy arrays (broadcasts)."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def make_projector(lon0: float, lat0: float):
    """Return (to_xy, to_lonlat) functions for a local equirectangular projection.

    x = R * dlon * cos(lat0),  y = R * dlat   (metres, origin at lon0/lat0)
    """
    lon0 = float(lon0)
    lat0 = float(lat0)
    k = np.cos(np.radians(lat0))

    def to_xy(lon, lat):
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        x = EARTH_RADIUS_M * np.radians(lon - lon0) * k
        y = EARTH_RADIUS_M * np.radians(lat - lat0)
        return x, y

    def to_lonlat(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * k))
        lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
        return lon, lat

    return to_xy, to_lonlat


def point_segment_distance(px, py, ax, ay, bx, by):
    """Planar distance (same units as inputs) from points P to segments AB, vectorised.
    Also returns t in [0,1], the projection position along AB."""
    px, py, ax, ay, bx, by = (np.asarray(v, dtype=np.float64) for v in (px, py, ax, ay, bx, by))
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    t = np.where(denom > 0, ((px - ax) * abx + (py - ay) * aby) / np.where(denom > 0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy), t


def path_length_m(lon, lat) -> float:
    """Sum of haversine step lengths along a polyline (metres)."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.size < 2:
        return 0.0
    return float(haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:]).sum())


if __name__ == "__main__":
    # Self-test: 1 deg of latitude ~ 111.19 km; projector vs haversine agree on a 1 km hop.
    d = haversine_m(0, 0, 0, 1)
    assert abs(d - 111_195) < 50, d
    to_xy, to_ll = make_projector(8.6, 49.86)
    x, y = to_xy(8.6 + 0.01, 49.86 + 0.005)
    h = haversine_m(8.6, 49.86, 8.61, 49.865)
    assert abs(np.hypot(x, y) - h) / h < 1e-3, (np.hypot(x, y), h)
    lo, la = to_ll(x, y)
    assert abs(lo - 8.61) < 1e-9 and abs(la - 49.865) < 1e-9
    dist, t = point_segment_distance(0, 1, -1, 0, 1, 0)
    assert abs(dist - 1) < 1e-12 and abs(t - 0.5) < 1e-12
    print(f"geo.py OK  | 1 deg lat = {d:.0f} m | projector vs haversine on ~900 m hop: "
          f"{np.hypot(x, y):.2f} vs {h:.2f} m")


def bearing_deg(dx, dy):
    """Compass bearing of a displacement (dx east, dy north) in degrees, in [0, 360).
    0 = north, 90 = east. (Team contract function; added by Person B.)"""
    b = np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)
    return np.where(b >= 360.0, b - 360.0, b)
