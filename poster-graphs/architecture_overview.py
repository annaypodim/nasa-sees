"""FIGURE 1 -- overall model pipeline (the architecture reported in WORKLOG H2/I:
the physics-GNN that BEATS the strong spatiotemporal-kriging baseline).

    pred = PRIOR + GNN correction
    PRIOR = b * persistence  +  (1-b) * RK-elev spatial      (b learned)
    ST-kriging baseline = PRIOR only (no GNN)  <- the honest floor we beat.

Left  : inputs  ->  graph construction
Right : the two prediction streams (physics prior + learned GNN correction) summed.
"""

import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _diagram import (
    BLUE,
    GOLD,
    GRAY,
    GREEN,
    INK,
    ORANGE,
    PURPLE,
    apply,
    arrow,
    band,
    box,
    label,
    titled_box,
)

apply()
fig, ax = plt.subplots(figsize=(13.5, 6.8))
ax.set_xlim(0, 135)
ax.set_ylim(0, 68)
ax.axis("off")

# ------------------------------------------------------------------ INPUTS ----
label(ax, 15, 65, "INPUTS", fs=12, weight="bold", color="#555")
inp = [
    (58, "PM₂.₅ sensor series\n(with held-out sensors)"),
    (49, "Sensor locations  x, y"),
    (40, "Elevation  (DEM)"),
    (31, "City wind  (HRRR: v, θ)"),
    (22, "Temporal lags\npersistence, lag-1, lag-24"),
]
for yy, txt in inp:
    box(ax, 15, yy, 22, 7.2, txt, color=INK, fs=10)

# ------------------------------------------------------- GRAPH CONSTRUCTION ----
titled_box(
    ax,
    40,
    40,
    19,
    19,
    "Graph\nconstruction",
    "edges (kNN)\n[dist, wind-along, Δelev]\n\nnodes\n[EPA-corrected PM$_{2.5}$]",
    color=GRAY,
    fs=8.2,
    gap=0.24,
)
# collector bus: input stubs -> vertical rail -> single feed into graph box
BUS = "#8a8a8a"
for yy, _ in inp:
    arrow(ax, (26, yy), (29.5, yy), color=BUS, lw=1.2, style="-")
arrow(ax, (29.5, inp[-1][0]), (29.5, inp[0][0]), color=BUS, lw=1.2, style="-")
arrow(ax, (29.5, 40), (32.4, 40), color=BUS, lw=1.4)

# ================================================================ PRIOR band ==
band(ax, 78, 47, 56, 32, color=GREEN, z=0, alpha=0.35)
label(ax, 79, 61, "PHYSICS PRIOR", fs=10.5, weight="bold", color="#2f5e3a")

# spatial RK-elev
titled_box(
    ax,
    65,
    52,
    25,
    11,
    "regression kriging w/ elevation\n(RK-elev)",
    "guess PM$_{2.5}$ from elevation\n"
    r"$\hat z = (\beta_0 + \beta_1\,\mathrm{elev}) + \mathrm{IDW(resid)}$",
    color=GREEN,
    fs=8.6,
    gap=0.26,
)
# temporal persistence
titled_box(
    ax,
    65,
    39.5,
    25,
    9,
    "Persistence  (persist)",
    "carry-forward each sensor's\nlast known value",
    color=GREEN,
    fs=8.8,
    gap=0.24,
)
# learnable blend
titled_box(
    ax,
    92,
    46,
    16,
    12,
    "Learnable blend",
    r"$\mathrm{prior} = b\cdot\mathrm{persist}$"
    "\n"
    r"$+\ (1-b)\cdot$ RK-elev",
    color=GREEN,
    fs=9.3,
    gap=0.24,
)
arrow(ax, (77.8, 52), (84.0, 47.5), color=GREEN[0], lw=1.6)
arrow(ax, (77.8, 39.5), (84.0, 44.5), color=GREEN[0], lw=1.6)

# graph -> prior inputs
arrow(ax, (49.8, 44), (52.2, 52), color=BUS, lw=1.3)
arrow(ax, (49.8, 39), (52.2, 39.5), color=BUS, lw=1.3)

# ============================================================ GNN CORRECTION ==
band(ax, 79, 20, 46, 20, color=BLUE, z=0, alpha=0.22)
label(
    ax,
    79,
    28,
    "GNN CORRECTION",
    fs=10.5,
    weight="bold",
    color="#2c4a72",
)
box(
    ax,
    79,
    18,
    34,
    11,
    "encode  >  K × graph layer  >  head\n\n"
    "diffusion  +  convection  +  local\nsoftmax fusion + residual",
    color=BLUE,
    fs=9.3,
)
arrow(ax, (49.8, 36), (61.8, 20), color=BUS, lw=1.3)

# =================================================================== SUM ======
box(ax, 115, 40, 9, 9, "+", color=INK, fs=22, weight="bold")
arrow(ax, (100.2, 46), (110.5, 41.2), color=GREEN[0], lw=2.0)  # prior ->
arrow(ax, (96.2, 18), (110.5, 38.3), color=BLUE[0], lw=2.0)  # correction ->
label(ax, 106, 45.5, "prior", fs=9, color=GREEN[0])
label(ax, 106, 26, "Δ correction", fs=9, color=BLUE[0])

# =================================================================== OUTPUT ====
box(
    ax,
    128,
    40,
    11,
    12,
    "Predicted\nPM₂.₅\n\nat held-out\nnode × hour",
    color=("#B23B3B", "#F7E4E4"),
    fs=9.5,
    weight="bold",
    text_color="#7a1f1f",
)
arrow(ax, (119.6, 40), (122.4, 40), color="#444", lw=2.0)

# ---- footnote: what the baseline is ----
label(
    ax,
    79,
    6.5,
    "ST-kriging baseline = PRIOR only (space+time interpolation, no GNN).   "
    "OURS = prior + GNN correction.",
    fs=9.2,
    color="#555",
    style="italic",
)

fig.tight_layout()
out = pathlib.Path(__file__).parent / "architecture_overview.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print(f"[saved] {out}")
