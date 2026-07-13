"""Poster figure (SHOW, don't tell): reanalysis wind is common-mode.

At each hour we take all SLC sensors' HRRR wind speed and plot the NETWORK MEAN
(one line) with a shaded +/-1 std band = how much sensors DIFFER from each other
at that moment. The band is thin (sensors barely differ = no spatial signal) yet
the line swings across the whole range (big temporal change). Thin band under a
swinging line = tiny spatial-vs-temporal variance ratio, made visible.

Right: the same sensors' elevations -- flat in time, but permanently separated
in space. THAT is a signal a spatial model can use; wind is not.

--- EDIT TITLES HERE ---------------------------------------------------------"""
TITLE   = "Reanalysis Wind Is the Same at Every Sensor, but Swings in Time"
TITLE_R = "Elevation: fixed in time,\nseparated in space"
SUPTITLE = ("A spatial model can only use signals that DIFFER between sensors "
            "— elevation does, wind doesn't")
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from _style import apply, FS
apply()

DATA = Path(__file__).resolve().parents[1] / "data" / "slc"
WIND = DATA / "wind_hrrr"
FEET_TO_M = 0.3048
HOURS = 168  # one week

SLC = {
    5758:5139, 6288:4258, 6352:4357, 10808:4418, 18021:4342, 18115:4518,
    18237:4280, 18469:5172, 20897:4224, 22647:5149, 39993:4404, 40713:5178,
    42079:5597, 44157:4533, 46773:4518, 46863:5013, 47171:4473, 49155:4265,
    89717:4773, 97391:4254, 105030:5313, 204009:4393, 205861:4651, 206185:4981,
    235753:4982, 240129:4804, 243975:4235, 251023:4646, 262151:4990, 280142:4835,
    283230:4378, 285111:4459, 298359:5604, 302286:4674, 303336:4916,
}

# [HOURS, n_sensors] wind speed matrix
cols = []
for sid in SLC:
    df = pd.read_csv(WIND / f"sensor_{sid}.csv").iloc[:HOURS]
    cols.append(np.hypot(df["u10"], df["v10"]).to_numpy())
spd = np.column_stack(cols)                 # [HOURS, N]
mean_t = spd.mean(axis=1)
std_t = spd.std(axis=1)
hrs = np.arange(len(mean_t))

# summary numbers for the caption (variance decomposition, matches the ~0.18
# spatial/temporal variance ratio): spatial = spread ACROSS sensors at a moment;
# temporal = how much a SINGLE sensor swings over time.
spatial_var = spd.var(axis=1).mean()          # mean over t of var across sensors
temporal_var = spd.var(axis=0).mean()         # mean over sensors of var over time
spatial = np.sqrt(spatial_var)
temporal = np.sqrt(temporal_var)
var_ratio = spatial_var / temporal_var

BLUE = "#2a78d6"; GREEN = "#008300"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5),
                               gridspec_kw={"width_ratios": [2.6, 1]})

axL.fill_between(hrs, mean_t - std_t, mean_t + std_t, color=BLUE, alpha=0.22,
                 lw=0, label="spread across sensors (±1σ)")
axL.plot(hrs, mean_t, color=BLUE, lw=1.6, label="network-mean wind speed")
axL.set_xlabel("Hours", fontsize=FS)
axL.set_ylabel("HRRR wind speed (m/s)", fontsize=FS)
axL.set_title(TITLE, fontsize=FS)
axL.legend(loc="upper right", fontsize=FS, frameon=True, framealpha=0.95)
axL.text(0.02, 0.96,
         f"spread across sensors ≈ {spatial:.2f} m/s\n"
         f"swing over time ≈ {temporal:.2f} m/s\n"
         f"spatial/temporal variance ≈ {var_ratio:.2f}",
         transform=axL.transAxes, va="top", ha="left", fontsize=FS, color="#52514e")
for s in ["top", "right"]:
    axL.spines[s].set_visible(False)
axL.spines["left"].set_color("#c3c2b7"); axL.spines["bottom"].set_color("#c3c2b7")
axL.grid(True, color="#e1e0d9", lw=1); axL.set_axisbelow(True)

# right: elevation of the same sensors -- flat, separated
elev = np.array([ft * FEET_TO_M for ft in SLC.values()])
axR.scatter(np.zeros_like(elev), elev, s=60, color=GREEN, edgecolor="white",
            linewidth=0.8, alpha=0.85, zorder=3)
axR.set_xlim(-0.5, 0.5); axR.set_xticks([])
axR.set_ylabel("Elevation (m)", fontsize=FS)
axR.set_title(TITLE_R, fontsize=FS)
axR.text(0.0, elev.max() + 15,
         f"range ≈ {elev.max()-elev.min():.0f} m", ha="center", va="bottom",
         fontsize=FS, color=GREEN)
for s in ["top", "right", "bottom"]:
    axR.spines[s].set_visible(False)
axR.spines["left"].set_color("#c3c2b7")
axR.grid(True, axis="y", color="#e1e0d9", lw=1); axR.set_axisbelow(True)

fig.suptitle(SUPTITLE, fontsize=FS, y=1.02)
fig.tight_layout()
out = "wind_common_mode.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
