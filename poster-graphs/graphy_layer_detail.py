"""FIGURE 2 -- GraPhyNet internals (the GNN CORRECTION block of Fig. 1).

Mirrors GraPhy's own layer figure but reflects OUR code (src/model/):
    encoder -> K x GraPhyLayer -> head
    one GraPhyLayer:  h -> {diffusion, convection, local} -> softmax fusion
                      -> h + fused   (residual)
    two MODULATORS gate BOTH transports: ElevationGate (per-edge, signed Δheight)
    and TemperatureGate (per-node).  local + the outer residual stay ungated.
"""

import itertools
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
fig, ax = plt.subplots(figsize=(13.5, 8.2))
ax.set_xlim(0, 135)
ax.set_ylim(0, 84)
ax.axis("off")

GRAY_L = "#555555"

# ---------------------------------------------------- top: network backbone ---
box(ax, 14, 76, 14, 7, "encoder\nLinear > H", color=GRAY, fs=10)
box(ax, 34, 76, 14, 7, "layer 1", color=GREEN, fs=10)
box(ax, 52, 76, 14, 7, "layer 2", color=GREEN, fs=10)
label(ax, 68, 76, "· · ·", fs=14)
box(ax, 84, 76, 14, 7, "layer K", color=GREEN, fs=10)
box(ax, 104, 76, 14, 7, "head\nlinear > 1", color=GRAY, fs=10)
for a, b in [(21, 27), (41, 45), (59, 63), (73, 77), (91, 97), (111, 116)]:
    arrow(ax, (a, 76), (b, 76), color=GRAY_L, lw=1.6)
label(ax, 6, 76, "X >", fs=11, weight="bold", color=GRAY_L)
label(ax, 120, 76, "Δ >", fs=11, weight="bold", color=BLUE[0])
label(
    ax,
    90,
    70,
    r"residual   $h \leftarrow h + \mathrm{fused}$   "
    r"(nudge rather than overwrite $\to$ no oversmoothing)",
    fs=9,
    style="italic",
    color="#666",
)

# dashed "zoom" cone from layer 2 down to the detail panel
for x0, x1 in [(48, 40), (56, 64)]:
    arrow(
        ax, (x0, 72.3), (x1, 63.3), color="#9ec0a4", lw=1.1, style="-", ls=(0, (4, 3))
    )

# ============================================ ONE GraPhyLayer (detailed) =======
band(ax, 67, 34, 126, 58, color=GREEN, z=0, alpha=0.14)
label(ax, 67, 60.5, "Inside one layer", fs=12, weight="bold", color="#2f5e3a")

# ---- node state h + broadcast bus to the three modules ----
box(ax, 11, 45, 11, 6, "h", color=INK, fs=13, weight="bold")
label(ax, 11, 40, "node state h\n(per sensor)", fs=8.5, color="#666")

arrow(ax, (11, 48), (11, 58), color=GRAY_L, lw=1.5, style="-")  # up to rail
arrow(ax, (11, 58), (102, 58), color=GRAY_L, lw=1.5, style="-")  # rail
for cx in (38, 70, 102):  # drops
    arrow(ax, (cx, 58), (cx, 55.3), color=GRAY_L, lw=1.5)


# ---- three physics modules (parallel columns) ----
def module(cx, color, name, edgetext, tag):
    band(ax, cx, 44, 30, 22, color=color, z=1, alpha=0.28)
    label(ax, cx, 52.5, name, fs=11, weight="bold", color=color[0])
    titled_box(
        ax, cx - 7, 47.8, 15, 5.5, "Node learning", "MLP", color=GRAY, fs=8.2, gap=0.22
    )
    e_title, e_body = edgetext.split("\n", 1)
    titled_box(
        ax, cx - 7, 40.0, 15, 6.5, e_title, e_body, color=color, fs=7.8, gap=0.28
    )
    label(ax, cx, 34.3, tag, fs=8, style="italic", color="#666")
    # simplified feed-forward neural-net glyph on the right, inside the band:
    # three columns of nodes (input -> hidden -> output), fully connected.
    cols = [cx + 3.5, cx + 7.5, cx + 11.5]
    sizes = [3, 4, 2]
    top, bot = 47.6, 40.6
    layers = []
    for xcol, n in zip(cols, sizes):
        ys = [top] if n == 1 else [top - (top - bot) * k / (n - 1) for k in range(n)]
        layers.append([(xcol, yy) for yy in ys])
    for a_layer, b_layer in zip(layers, layers[1:]):        # edges between layers
        for (xa, ya) in a_layer:
            for (xb, yb) in b_layer:
                ax.plot([xa, xb], [ya, yb], "-", color=color[0], lw=0.4,
                        alpha=0.35, zorder=4)
    for lyr in layers:                                      # nodes on top
        xs = [p[0] for p in lyr]; ys = [p[1] for p in lyr]
        ax.plot(xs, ys, "o", ms=4.5, color=color[0], zorder=5)


module(
    38,
    ORANGE,
    "Diffusion",
    "Edge learning\ngraph Laplacian +\nper-feature rate ℓ",
    "spread high > low",
)
module(
    70,
    BLUE,
    "Convection",
    "Edge learning\nEdge-MLP + wind-along\nmessage passing",
    "wind transport",
)
module(
    102,
    PURPLE,
    "Local",
    "Edge learning\ndegree-normalised\ngraph conv",
    "node source / sink",
)

# ---- gates (modulators) feeding diffusion + convection from below ----
box(
    ax,
    53,
    23.5,
    28,
    6,
    "Elevation + Temperature gates\nmodulate diffusion & convection only",
    color=GOLD,
    fs=8,
)
arrow(ax, (46, 26.7), (46, 32.9), color=GOLD[0], lw=1.4)  # -> diffusion
arrow(ax, (60, 26.7), (60, 32.9), color=GOLD[0], lw=1.4)  # -> convection
ax.text(102, 28, "local stays ungated", ha="center", va="center", fontsize=8,
        style="italic", color="#7a6a3a", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

# ---- module outputs -> softmax fusion -> residual ----
outs = [
    (38, r"$d = \mathrm{Diffusion}(h)$"),
    (70, r"$c = \mathrm{Convection}(h)$"),
    (102, r"$l = \mathrm{Local}(h)$"),
]
for cx, eq in outs:
    arrow(ax, (cx, 32.9), (cx, 16.3), color="#4a7a58", lw=1.6)
    ax.text(cx, 18.6, eq, ha="center", va="center", fontsize=9.5,
            color="#2f5e3a", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))

titled_box(
    ax,
    70,
    13,
    78,
    6,
    "Dynamic Softmax Fusion",
    r"$w = \mathrm{softmax}(\mathrm{MLP}(h)),\ \ \sum_i w_i = 1$",
    color=GREEN,
    fs=10,
    title_fs=10,
    gap=0.24,
)
arrow(ax, (70, 10), (70, 8.3), color=GRAY_L, lw=1.6)
box(
    ax,
    70,
    6,
    54,
    4.6,
    r"$\mathrm{fused} = w_D\, d + w_C\, c + w_L\, l"
    r" \qquad h \leftarrow h + \mathrm{fused}$",
    color=INK,
    fs=10,
)

fig.tight_layout()
out = pathlib.Path(__file__).parent / "graphy_layer_detail.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print(f"[saved] {out}")
