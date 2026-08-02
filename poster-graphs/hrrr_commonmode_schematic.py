"""Poster schematic: why reanalysis wind is common-mode at sensor scale."""

import matplotlib.pyplot as plt
import numpy as np
from _style import FS, apply
from matplotlib.patches import FancyArrow, Rectangle

apply()

# --- EDIT TITLE HERE ---
TITLE = "Reanalysis Wind Is Common-Mode at Sensor Scale"

BLUE = "#2a78d6"
GRIDLN = "#b02b2b"
INK = "#0b0b0b"
SECOND = "#52514e"
WIND = "#256abf"

fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.axis("off")

# --- full 3 km HRRR grid across the whole domain ---
cell = 3.0
for gx in np.arange(0, 9.1, cell):
    ax.plot([gx, gx], [0, 9], color="#e0b3b3", lw=1.0, zorder=1)
for gy in np.arange(0, 9.1, cell):
    ax.plot([0, 9], [gy, gy], color="#e0b3b3", lw=1.0, zorder=1)
# highlight the center cell (the one we call out)
x0, y0 = 3.0, 3.0
ax.add_patch(
    Rectangle(
        (x0, y0),
        cell,
        cell,
        fill=True,
        facecolor="#f7e3e3",
        edgecolor=GRIDLN,
        lw=2.5,
        zorder=1,
    )
)
ax.text(
    x0 + cell / 2,
    y0 + cell + 0.12,
    "one 3 km HRRR grid cell",
    ha="center",
    va="bottom",
    fontsize=FS,
    color=GRIDLN,
)

# --- wind field EVERYWHERE: identical vector at every grid node ---
wx, wy = 0.75, 0.38
for gx in np.arange(0.6, 9.0, 1.0):
    for gy in np.arange(0.6, 9.0, 1.0):
        ax.add_patch(
            FancyArrow(
                gx,
                gy,
                wx,
                wy,
                width=0.02,
                head_width=0.13,
                head_length=0.12,
                length_includes_head=True,
                color="#b9c9e6",
                zorder=2,
            )
        )

# ~10 sensor dots spread across several cells (1-3 km apart)
rng = np.random.default_rng(7)
sensors = np.array(
    [
        [3.5, 3.6],
        [4.4, 3.9],
        [5.1, 3.4],
        [3.8, 4.8],
        [4.9, 4.9],  # in center cell
        [5.4, 4.3],
        [4.2, 5.4],
        [2.2, 2.4],
        [6.6, 5.8],
        [2.6, 6.3],  # neighbours
    ]
)
ax.scatter(
    sensors[:, 0],
    sensors[:, 1],
    s=150,
    color=BLUE,
    edgecolor="white",
    linewidth=1.8,
    zorder=5,
)

# emphasize the wind AT each sensor (same vector -> common-mode)
for sx, sy in sensors:
    ax.add_patch(
        FancyArrow(
            sx,
            sy,
            wx,
            wy,
            width=0.03,
            head_width=0.17,
            head_length=0.15,
            length_includes_head=True,
            color=WIND,
            zorder=4,
        )
    )

# legend proxies
ax.scatter([], [], s=120, color=BLUE, label="PM2.5 sensor")
ax.plot([], [], color=WIND, lw=2.5, label="HRRR wind at sensor (identical)")
ax.plot([], [], color="#b9c9e6", lw=2.5, label="HRRR wind field")

# scale reference
ax.annotate(
    "",
    xy=(3.5, 2.55),
    xytext=(4.4, 2.55),
    arrowprops=dict(arrowstyle="<->", lw=1.2, color=SECOND),
)
ax.text(
    3.95,
    2.35,
    "~1–3 km sensor spacing",
    ha="center",
    va="top",
    fontsize=FS,
    color=SECOND,
)

ax.text(
    4.5,
    0.4,
    "All sensors inside one HRRR cell receive ~identical wind "
    "(17–18% spatial variation)",
    ha="center",
    va="center",
    fontsize=FS,
    color=INK,
)

ax.set_title(TITLE, fontsize=FS)
ax.legend(loc="upper right", fontsize=FS, frameon=True, framealpha=0.95)

fig.tight_layout()
out = "hrrr_commonmode_schematic.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
