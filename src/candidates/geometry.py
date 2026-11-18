"""Task 1: arc geometry in metres, per trip."""
import numpy as np
import pandas as pd

from src.data.geo import make_projector


def trip_projector(clean_track, report=None):
    """Projector for one trip, centred on the mean of its (cleaned) GPS points.

    If clean_track's report gives lon0/lat0 we reuse them, so GPS points and nodes are
    guaranteed to be projected with the SAME centre.
    """
    if report is not None and "lon0" in report and "lat0" in report:
        lon0, lat0 = report["lon0"], report["lat0"]
    else:
        lon0, lat0 = float(clean_track["lon"].mean()), float(clean_track["lat"].mean())
    proj = make_projector(lon0, lat0)
    # Person A's make_projector returns (to_xy, to_lonlat); we only need to_xy
    return proj[0] if isinstance(proj, tuple) else proj


def arc_geometry(nodes, arcs, projector):
    """Each arc is a straight segment A (from_node) -> B (to_node).

    Project every node once, then look up the two end nodes of every arc (vectorised).
    Returns a DataFrame indexed by arc_id: ax, ay, bx, by, length_m.
    """
    nx, ny = projector(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    fpos = nodes.index.get_indexer(arcs["from_node"].to_numpy())
    tpos = nodes.index.get_indexer(arcs["to_node"].to_numpy())
    if (fpos < 0).any() or (tpos < 0).any():
        raise ValueError("arcs reference node_ids that are not in .nodes")
    ax, ay, bx, by = nx[fpos], ny[fpos], nx[tpos], ny[tpos]
    return pd.DataFrame({"ax": ax, "ay": ay, "bx": bx, "by": by,
                         "length_m": np.hypot(bx - ax, by - ay)}, index=arcs.index)
