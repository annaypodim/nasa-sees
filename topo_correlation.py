"""
PM2.5 correlation overlaid on REAL topography
=============================================================================
The existing `5_spatial_corr.png` answers "are physical neighbours actually
correlated?" on a blank lon/lat plane. This script draws the SAME two
correlation views, but on top of the region's real terrain -- so you can read
correlation structure against ridges, valleys and the Boulder foothills
gradient instead of against empty space.

  panel 0: kNN sensor edges colored by PM2.5 correlation, over a hillshaded
           + contoured elevation basemap.
  panel 1: correlation field from one reference sensor, same basemap.

Elevation source
----------------
The repo only stores elevation AT the sensor points, which can't draw a
regional surface. So we pull a real DEM once from AWS Terrain Tiles
("terrarium" PNGs, public, no API key), decode it to metres, and cache the
mosaic in data/dem_boulder.npz. Later runs read the cache and never hit the
network. Correlation + coordinates come from the same loaders the pipeline
uses (build_graph2), so the overlay matches the model's actual inputs.

    elevation_m = (R * 256 + G + B / 256) - 32768        # terrarium encoding

run:  .venv/bin/python topo_correlation.py
      .venv/bin/python topo_correlation.py --raw     # skip preprocessing drop
      .venv/bin/python topo_correlation.py --zoom 13 # finer DEM (more tiles)
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource, Normalize
from PIL import Image

import build_graph2 as bg
import data_visualizations as dv

# public global DEM, terrarium RGB encoding, no key required
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
DEM_CACHE = bg.DATA_DIR / "dem_boulder.npz"
MARGIN_DEG = 0.02          # pad the sensor bbox so nodes aren't on the edge
DEFAULT_ZOOM = 12          # ~30 m/px near this latitude; plenty for a city


# ===========================================================================
# 1. DEM: fetch (once) a real elevation mosaic for the sensor bounding box
# ===========================================================================
def _deg2tile(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Web-mercator (slippy-map) fractional tile coords for a lat/lon."""
    lat_r = math.radians(lat)
    n = 2.0 ** z
    xt = (lon + 180.0) / 360.0 * n
    yt = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return xt, yt


def _tile2lon(xt: float, z: int) -> float:
    return xt / 2.0 ** z * 360.0 - 180.0


def _tile2lat(yt: float, z: int) -> float:
    n = math.pi - 2.0 * math.pi * yt / 2.0 ** z
    return math.degrees(math.atan(math.sinh(n)))


def _decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    return (rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0) - 32768.0


def fetch_dem(bounds: tuple[float, float, float, float], zoom: int):
    """Return (elev[H,W] metres, lon[W], lat[H]) covering `bounds`.

    bounds = (lon_min, lon_max, lat_min, lat_max). Mosaics every terrarium
    tile the box touches, then crops to the box. Result is cached so we only
    ever download once per (bounds, zoom).
    """
    lon_min, lon_max, lat_min, lat_max = bounds
    key = f"{zoom}_{lon_min:.4f}_{lon_max:.4f}_{lat_min:.4f}_{lat_max:.4f}"

    if DEM_CACHE.exists():
        cached = np.load(DEM_CACHE, allow_pickle=True)
        if str(cached.get("key")) == key:
            print(f"DEM cache hit ({DEM_CACHE.name})")
            return cached["elev"], cached["lon"], cached["lat"]

    # tile index range covering the box (y grows southward)
    x0, y1 = _deg2tile(lat_min, lon_min, zoom)
    x1, y0 = _deg2tile(lat_max, lon_max, zoom)
    xa, xb = int(math.floor(x0)), int(math.floor(x1))
    ya, yb = int(math.floor(y0)), int(math.floor(y1))

    n_tiles = (xb - xa + 1) * (yb - ya + 1)
    print(f"fetching {n_tiles} DEM tile(s) at zoom {zoom} ...")
    rows = []
    for ty in range(ya, yb + 1):
        cols = []
        for tx in range(xa, xb + 1):
            url = TILE_URL.format(z=zoom, x=tx, y=ty)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            img = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
            cols.append(_decode_terrarium(img))
        rows.append(np.hstack(cols))
    mosaic = np.vstack(rows)                     # [H, W] north-up

    # lon/lat of every pixel centre in the mosaic
    h, w = mosaic.shape
    px = (xa + (np.arange(w) + 0.5) / 256.0)
    py = (ya + (np.arange(h) + 0.5) / 256.0)
    lon = np.array([_tile2lon(v, zoom) for v in px])
    lat = np.array([_tile2lat(v, zoom) for v in py])

    # crop to the requested box
    cx = (lon >= lon_min) & (lon <= lon_max)
    cy = (lat >= lat_min) & (lat <= lat_max)
    elev = mosaic[np.ix_(cy, cx)]
    lon, lat = lon[cx], lat[cy]

    np.savez_compressed(DEM_CACHE, elev=elev, lon=lon, lat=lat, key=key)
    print(f"DEM cached -> {DEM_CACHE}  ({elev.shape[0]}x{elev.shape[1]} px, "
          f"{elev.min():.0f}-{elev.max():.0f} m)")
    return elev, lon, lat


# ===========================================================================
# 2. basemap: hillshade + filled contours on a matplotlib axis
# ===========================================================================
def draw_topography(ax, elev, lon, lat):
    """Paint terrain into `ax`: colour by elevation, shade by slope, then thin
    contour lines. Returns the elevation image (for a shared colorbar)."""
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    # exaggerate vertical relief so the foothills read at city scale
    dx = (lon[-1] - lon[0]) / max(len(lon) - 1, 1) * 111_000 * math.cos(
        math.radians(np.mean(lat)))
    dy = (lat[0] - lat[-1]) / max(len(lat) - 1, 1) * 111_000
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(elev, vert_exag=5.0, dx=abs(dx), dy=abs(dy))

    im = ax.imshow(elev, extent=extent, origin="upper", cmap="terrain",
                   alpha=0.85, aspect="auto")
    ax.imshow(shade, extent=extent, origin="upper", cmap="gray",
              alpha=0.35, aspect="auto")
    cs = ax.contour(lon, lat, elev, levels=12, colors="k",
                    linewidths=0.4, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=5, fmt="%.0f")
    return im


# ===========================================================================
# 3. the figure: two correlation views on the terrain basemap
# ===========================================================================
def fig_topo_correlation(pm_wide, coords, ids, zoom):
    lat = coords["lat"].to_numpy()
    lon = coords["lon"].to_numpy()
    x_m, y_m = bg.project(lat, lon)
    edge_index = bg.knn_edges(x_m, y_m, bg.K)
    dist = bg.distance_matrix(x_m, y_m)
    corr = bg.correlation_matrix(pm_wide)

    bounds = (lon.min() - MARGIN_DEG, lon.max() + MARGIN_DEG,
              lat.min() - MARGIN_DEG, lat.max() + MARGIN_DEG)
    elev, dlon, dlat = fetch_dem(bounds, zoom)

    norm = Normalize(-1, 1)
    cmap = plt.get_cmap("RdBu_r")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 8))

    # --- panel 0: kNN edges colored by endpoint correlation, over terrain ---
    im0 = draw_topography(ax0, elev, dlon, dlat)
    segs, cols = [], []
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if i < j:
            segs.append([(lon[i], lat[i]), (lon[j], lat[j])])
            cols.append(corr[i, j])
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=3.5)
    lc.set_array(np.array(cols))
    ax0.add_collection(lc)
    ax0.scatter(lon, lat, s=55, c="0.1", edgecolors="w", linewidths=0.6, zorder=3)
    for idx, sid in enumerate(ids):
        ax0.annotate(sid, (lon[idx], lat[idx]), fontsize=5.5,
                     xytext=(3, 3), textcoords="offset points")
    fig.colorbar(im0, ax=ax0, label="elevation (m)", fraction=0.046, pad=0.02)
    fig.colorbar(lc, ax=ax0, label="PM2.5 corr of edge endpoints",
                 fraction=0.046, pad=0.10)
    ax0.set(title="Correlation vs. terrain: kNN edges over real topography\n"
            "(red edge = correlated neighbours; contours = elevation)",
            xlabel="longitude", ylabel="latitude")
    ax0.set_xlim(dlon.min(), dlon.max())
    ax0.set_ylim(dlat.min(), dlat.max())

    # --- panel 1: correlation field from the central sensor, over terrain ---
    draw_topography(ax1, elev, dlon, dlat)
    ref = int(np.argmin(dist.mean(axis=1)))
    sc = ax1.scatter(lon, lat, c=corr[ref], cmap=cmap, norm=norm, s=170,
                     edgecolors="k", linewidths=1.0, zorder=3)
    ax1.scatter(lon[ref], lat[ref], marker="*", s=520, c="yellow",
                edgecolors="k", zorder=4, label=f"reference {ids[ref]}")
    for idx, sid in enumerate(ids):
        ax1.annotate(f"{sid}\n{dist[ref, idx] / 1000:.1f}km\n"
                     f"{coords['elevation'].to_numpy()[idx]:.0f}m",
                     (lon[idx], lat[idx]), fontsize=5,
                     xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, ax=ax1, label=f"PM2.5 corr with {ids[ref]}",
                 fraction=0.046, pad=0.02)
    ax1.set(title="Does correlation follow terrain?\n"
            "(each sensor colored by corr to the central node)",
            xlabel="longitude", ylabel="latitude")
    ax1.set_xlim(dlon.min(), dlon.max())
    ax1.set_ylim(dlat.min(), dlat.max())
    ax1.legend(loc="best", fontsize=8)

    dv.VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = dv.VIZ_DIR / "8_topo_correlation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true",
                    help="skip the preprocessing sensor drop")
    ap.add_argument("--zoom", type=int, default=DEFAULT_ZOOM,
                    help="DEM tile zoom (higher = finer, more tiles)")
    args = ap.parse_args()

    pm_wide, coords, ids = dv.load(apply_preprocess=not args.raw)
    fig_topo_correlation(pm_wide, coords, ids, args.zoom)


if __name__ == "__main__":
    main()
