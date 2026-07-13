"""Poster figure: the ACTUAL HRRR (3 km) wind vector at each sensor for a single
representative windy hour, drawn on the real 3 km grid, Fresno and SLC side by
side.

Replaces the idealized common-mode schematic with real data. We read the fetched
HRRR wind (scripts/fetch_wind_hrrr.py output) and, rather than averaging over the
whole record (a year of shifting winds cancels to ~0 and shows nothing), we pick
the single hour with the strongest network-mean wind and quiver the real u/v at
each sensor on a true 3 km grid in local km. The eye sees what the numbers say:
across the flat valley (Fresno) the arrows are near-parallel -- one vector for
the whole network (common-mode) -- while the terrain city (SLC) shows visibly
more turning as flow channels through the valley.
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import FS, apply

apply()

# data lives in the main checkout (untracked in this worktree)
ROOT = Path("/Users/annaypodimatopoulou/Code/side_quests/nasa-sees")
import re

_ROW = re.compile(
    r"\[\s*(\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+)\s*\]"
)

BLUE = "#2a78d6"
GRIDLN = "#c99"
INK = "#0b0b0b"
SECOND = "#52514e"
WIND = "#256abf"

CITIES = [
    ("fresno", "Fresno  (flat valley)"),
    ("slc", "Salt Lake City  (terrain)"),
]


def parse_coords(city):
    txt = (ROOT / f"data/{city}/coords/sensor_lat_long_alt").read_text()
    return {int(i): (float(la), float(lo)) for i, la, lo, _ in _ROW.findall(txt)}


def snapshot_wind(city):
    """Pick the hour with the strongest network-mean wind; return per-sensor
    lat, lon, u, v at that hour plus the timestamp."""
    coords = parse_coords(city)
    wdir = ROOT / f"data/{city}/wind_hrrr"
    series = {}  # sid -> {time: (u, v)}
    for f in sorted(wdir.glob("sensor_*.csv")):
        sid = int(f.stem.split("_")[1])
        if sid not in coords:
            continue
        d = {}
        with open(f) as fh:
            for row in csv.DictReader(fh):
                try:
                    d[row["time"]] = (float(row["u10"]), float(row["v10"]))
                except (ValueError, KeyError):
                    pass
        if d:
            series[sid] = d

    # timestamps present for (nearly) all sensors
    common = set.intersection(*(set(d) for d in series.values()))

    # rank by mean speed across sensors; take the windiest hour
    def mean_speed(t):
        return np.mean([math.hypot(*series[s][t]) for s in series])

    best = max(common, key=mean_speed)

    lat, lon, U, V = [], [], [], []
    for sid, d in series.items():
        la, lo = coords[sid]
        u, v = d[best]
        lat.append(la)
        lon.append(lo)
        U.append(u)
        V.append(v)
    return (*map(np.array, (lat, lon, U, V)), best)


def to_km(lat, lon):
    """Equirectangular projection to local km, centered on the network."""
    lat0 = lat.mean()
    x = (lon - lon.mean()) * 111.32 * math.cos(math.radians(lat0))
    y = (lat - lat.mean()) * 110.57
    return x, y


def draw(ax, title, city):
    lat, lon, u, v, when = snapshot_wind(city)
    x, y = to_km(lat, lon)

    # 3 km HRRR grid spanning the network (+ margin), aligned to origin
    pad = 3.0
    xlo, xhi = math.floor((x.min() - pad) / 3) * 3, math.ceil((x.max() + pad) / 3) * 3
    ylo, yhi = math.floor((y.min() - pad) / 3) * 3, math.ceil((y.max() + pad) / 3) * 3
    for gx in np.arange(xlo, xhi + 0.1, 3.0):
        ax.plot([gx, gx], [ylo, yhi], color=GRIDLN, lw=0.9, zorder=1)
    for gy in np.arange(ylo, yhi + 0.1, 3.0):
        ax.plot([xlo, xhi], [gy, gy], color=GRIDLN, lw=0.9, zorder=1)

    # real wind vectors at each sensor for the chosen hour (same scale both panels)
    ax.quiver(
        x,
        y,
        u,
        v,
        color=WIND,
        angles="xy",
        scale_units="xy",
        scale=2.1,
        width=0.007,
        headwidth=4,
        headlength=5,
        zorder=4,
    )
    ax.scatter(x, y, s=55, color=BLUE, edgecolor="white", linewidth=1.2, zorder=5)

    # honest spread stat: mean speed and directional spread across sensors
    spd = np.hypot(u, v)
    ang = np.arctan2(v, u)
    R = math.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))  # resultant length
    dir_spread = math.degrees(math.sqrt(max(0.0, -2 * math.log(max(R, 1e-9)))))
    ts = when.replace("T", " ") + " UTC"
    ax.text(
        0.5,
        -0.11,
        f"{ts}   ·   n={len(x)} sensors   ·   mean {spd.mean():.1f} m/s\n"
        f"direction spread ±{dir_spread:.0f}° across the network",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FS,
        color=SECOND,
    )

    ax.set_title(title, fontsize=FS)
    ax.set_aspect("equal")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("east–west (km)", fontsize=FS)
    ax.set_ylabel("north–south (km)", fontsize=FS)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(14, 7.6))
for ax, (city, title) in zip(axes, CITIES):
    draw(ax, title, city)

# shared scale bar / legend proxies
axes[0].quiver([], [], [], [], color=WIND, label="HRRR wind (windiest hour)")
axes[0].scatter([], [], s=55, color=BLUE, label="PM2.5 sensor")
axes[0].plot([], [], color=GRIDLN, lw=1.2, label="3 km HRRR grid")
axes[0].legend(loc="upper left", fontsize=FS, frameon=True, framealpha=0.95)

fig.suptitle("HRRR 3 km Wind Across the Sensor Network", fontsize=FS, y=0.99)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
out = Path(__file__).parent / "hrrr_wind_vectors_real.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
