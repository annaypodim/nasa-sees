"""
Pittsburgh sensor elevations on real topography
=============================================================================
`data/pittsburgh_data/pittsburgh_loc_elev.txt` is just coordinate points --
[sensor_id, lat, lon, elevation_ft] -- with NO PM2.5 time series, so we can't
draw the correlation overlay that `topo_correlation.py` does (that needs
sensor readings). What we CAN do is reuse that script's DEM machinery to show
where each sensor sits on the region's real terrain.

  panel 0: every sensor plotted on a hillshaded Pittsburgh DEM, colored by its
           reported elevation.
  panel 1: sanity check -- reported elevation (ft -> m) vs the DEM's elevation
           at that exact lat/lon. Points off the 1:1 line = coordinate or
           altitude-units problems.

Elevation basemap comes from AWS Terrain Tiles via topo_correlation.fetch_dem
(cached to data/dem_pittsburgh.npz after the first run).

run:  .venv/bin/python pittsburgh_elevation.py
      .venv/bin/python pittsburgh_elevation.py --zoom 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import build_graph2 as bg
import topo_correlation as tc

FEET_TO_M = 0.3048
LOC_FILE = bg.DATA_DIR / "pittsburgh_data" / "pittsburgh_loc_elev.txt"
DEM_CACHE = bg.DATA_DIR / "dem_pittsburgh.npz"
OUT = bg.OUT_DIR / "viz" / "pittsburgh_elevation.png"


def load_points(path: Path):
    """Parse the `"data": [[id, lat, lon, elev_ft], ...]` block."""
    obj = json.loads("{" + path.read_text() + "}")
    rows = obj["data"]
    ids = [str(int(r[0])) for r in rows]
    lat = np.array([r[1] for r in rows], dtype=float)
    lon = np.array([r[2] for r in rows], dtype=float)
    elev_ft = np.array([r[3] for r in rows], dtype=float)
    return ids, lat, lon, elev_ft


def sample_dem(elev, dlon, dlat, lon, lat):
    """Nearest-pixel DEM elevation (m) at each sensor lat/lon."""
    ix = np.clip(np.searchsorted(dlon, lon), 0, len(dlon) - 1)
    # dlat is descending (north-up), so search on a reversed copy
    iy = np.clip(len(dlat) - 1 - np.searchsorted(dlat[::-1], lat),
                 0, len(dlat) - 1)
    return elev[iy, ix]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=tc.DEFAULT_ZOOM)
    args = ap.parse_args()

    ids, lat, lon, elev_ft = load_points(LOC_FILE)
    elev_m = elev_ft * FEET_TO_M
    print(f"loaded {len(ids)} Pittsburgh sensors, "
          f"elev {elev_ft.min():.0f}-{elev_ft.max():.0f} ft")

    bounds = (lon.min() - tc.MARGIN_DEG, lon.max() + tc.MARGIN_DEG,
              lat.min() - tc.MARGIN_DEG, lat.max() + tc.MARGIN_DEG)
    # point fetch_dem's cache at a Pittsburgh-specific file
    tc.DEM_CACHE = DEM_CACHE
    elev, dlon, dlat = tc.fetch_dem(bounds, args.zoom)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 8))

    # --- panel 0: sensors on the DEM, colored by reported elevation ---------
    tc.draw_topography(ax0, elev, dlon, dlat)
    sc = ax0.scatter(lon, lat, c=elev_ft, cmap="inferno", s=90,
                     edgecolors="w", linewidths=0.7, zorder=3)
    for i, sid in enumerate(ids):
        ax0.annotate(sid, (lon[i], lat[i]), fontsize=4.5,
                     xytext=(2, 2), textcoords="offset points")
    fig.colorbar(sc, ax=ax0, label="reported sensor elevation (ft)",
                 fraction=0.046, pad=0.02)
    ax0.set(title="Pittsburgh sensors on real topography\n"
            "(points colored by reported elevation)",
            xlabel="longitude", ylabel="latitude")
    ax0.set_xlim(dlon.min(), dlon.max())
    ax0.set_ylim(dlat.min(), dlat.max())

    # --- panel 1: reported vs DEM elevation (units / coordinate sanity) -----
    dem_at = sample_dem(elev, dlon, dlat, lon, lat)
    lo = min(elev_m.min(), dem_at.min())
    hi = max(elev_m.max(), dem_at.max())
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax1.scatter(dem_at, elev_m, s=60, c="tab:blue", edgecolors="k", zorder=3)
    resid = elev_m - dem_at
    ax1.set(title=f"Reported vs DEM elevation  "
            f"(mean diff {resid.mean():+.0f} m, RMSE {np.sqrt((resid**2).mean()):.0f} m)",
            xlabel="DEM elevation at sensor (m)",
            ylabel="reported sensor elevation (m)")
    ax1.legend(loc="best")
    ax1.set_aspect("equal", adjustable="datalim")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
