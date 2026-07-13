"""Poster figure (SHOW, don't tell): why elevation is a usable spatial covariate
and reanalysis temp / AOD / wind are not. Pittsburgh, real per-sensor data.

For each covariate we build a [time, sensor] matrix, z-score it GLOBALLY (so all
covariates share one unit = fraction of total std), then plot:
    x = temporal variability  = how much ONE sensor swings over time
    y = spatial variability    = how much sensors DIFFER at the same moment
A spatial (interpolation) model can only use the y-axis. Elevation sits top-left
(all spatial, zero temporal); temp / AOD / wind sit bottom-right (they swing in
time but barely differ between sensors = common-mode). You SEE the separation.

--- EDIT TITLES HERE ---------------------------------------------------------"""
TITLE = "Only Elevation Differs Between Sensors — Temp, AOD & Wind Are Common-Mode"
SUBTITLE = "Pittsburgh, real per-sensor data (each covariate z-scored to a shared unit)"
# -----------------------------------------------------------------------------
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from _style import apply, FS
apply()

PGH = Path(__file__).resolve().parents[1] / "data" / "pittsburgh"
FEET_TO_M = 0.3048


def zscore_std(mat):
    """Global z-score (nan-aware), return (temporal_std, spatial_std) in
    fraction-of-total-std units. spatial = mean over time of var across sensors;
    temporal = mean over sensors of var over time."""
    fin = np.isfinite(mat)
    mu, sd = np.nanmean(mat), np.nanstd(mat)
    z = (mat - mu) / (sd + 1e-12)
    # spatial: variance across sensors at each time, averaged over times with >=2 sensors
    sp = np.array([np.nanvar(z[t, np.isfinite(z[t])]) for t in range(z.shape[0])
                   if fin[t].sum() >= 2])
    # temporal: variance over time for each sensor, averaged over sensors with >=2 obs
    tp = np.array([np.nanvar(z[np.isfinite(z[:, n]), n]) for n in range(z.shape[1])
                   if fin[:, n].sum() >= 2])
    return np.sqrt(tp.mean()), np.sqrt(sp.mean())


# ---- temperature: <id>_temperature.csv (datetime, temperature_C) ----
tcols = {}
for f in glob.glob(str(PGH / "temperature" / "*_temperature.csv")):
    sid = Path(f).name.split("_")[0]
    d = pd.read_csv(f, parse_dates=["datetime"]).set_index("datetime")["temperature_C"]
    tcols[sid] = d
temp = pd.DataFrame(tcols).to_numpy()

# ---- wind: wind_hrrr/sensor_<id>.csv (time, u10, v10) -> speed ----
wcols = {}
for f in glob.glob(str(PGH / "wind_hrrr" / "sensor_*.csv")):
    sid = re.search(r"sensor_(\d+)", f).group(1)
    d = pd.read_csv(f, parse_dates=["time"]).set_index("time")
    wcols[sid] = np.hypot(d["u10"], d["v10"])
wind = pd.DataFrame(wcols).to_numpy()

# ---- AOD: one long csv, daily/sparse -> [day, sensor] on aod_055 ----
a = pd.read_csv(PGH / "aod" / "aod_points_2023-01-01_2023-12-31.csv",
                parse_dates=["datetime"])
a["day"] = a["datetime"].dt.floor("D")
aod = a.pivot_table(index="day", columns="id", values="aod_055", aggfunc="mean").to_numpy()

# ---- elevation: coords file [id, lat, lon, elev_ft] -> static ----
txt = (PGH / "coords" / "pittsburgh_loc_elev.txt").read_text()
elev_ft = [int(m[3]) for m in re.findall(r"\[(\d+),([-\d.]+),([-\d.]+),(\d+)\]", txt)]
elev_m = np.array(elev_ft, dtype=float) * FEET_TO_M
elev_z = (elev_m - elev_m.mean()) / (elev_m.std() + 1e-12)  # spatial std = 1, temporal = 0

points = {
    "Temperature": (*zscore_std(temp), "#eb6834"),
    "AOD":         (*zscore_std(aod),  "#eda100"),
    "Wind speed":  (*zscore_std(wind), "#2a78d6"),
    "Elevation":   (0.0, float(np.std(elev_z)), "#008300"),
}

fig, ax = plt.subplots(figsize=(8.5, 7))

# reference: "spatial = temporal" diagonal; above it a covariate is spatially useful
lim = 1.15
ax.plot([0, lim], [0, lim], ls="--", lw=1, color="#c3c2b7", zorder=1)
ax.text(lim - 0.02, lim - 0.02, "spatial = temporal", ha="right", va="bottom",
        fontsize=FS, color="#898781", rotation=45, rotation_mode="anchor")
# shade the "usable by a spatial model" zone (high spatial)
ax.axhspan(0.5, lim, color="#008300", alpha=0.05, zorder=0)
ax.text(0.98, 0.6, "usable by a spatial model\n(differs between sensors)",
        fontsize=FS, color="#006300", va="center", ha="right")

for name, (tstd, sstd, color) in points.items():
    ax.scatter(tstd, sstd, s=180, color=color, edgecolor="white",
               linewidth=1.5, zorder=4)
    # place each label clear of its point and of neighbours
    off = {"Elevation": (0.04, -0.05, "left"),
           "Wind speed": (0.03, 0.04, "left"),
           "AOD": (-0.03, -0.05, "right"),
           "Temperature": (-0.03, 0.05, "right")}[name]
    ax.annotate(name, (tstd, sstd), xytext=(tstd + off[0], sstd + off[1]),
                fontsize=FS, color=color, va="center", ha=off[2])

ax.set_xlim(-0.05, lim)
ax.set_ylim(-0.05, lim)
ax.set_xlabel("Temporal variability  (how much one sensor swings over time)", fontsize=FS)
ax.set_ylabel("Spatial variability  (how much sensors differ at one moment)", fontsize=FS)
ax.set_title(f"{TITLE}\n{SUBTITLE}", fontsize=FS)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
ax.grid(True, color="#e1e0d9", lw=1); ax.set_axisbelow(True)

fig.tight_layout()
out = "covariate_spatial_temporal.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
for name, (t, s, _) in points.items():
    print(f"  {name:12s} temporal_std={t:.3f}  spatial_std={s:.3f}  ratio(sp/tp)²={ (s/t)**2 if t>0 else float('inf'):.3f}")
