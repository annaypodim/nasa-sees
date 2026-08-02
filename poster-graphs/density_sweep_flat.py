"""Poster figure: on flat terrain, no density level lets the model beat IDW."""

import matplotlib.pyplot as plt
import numpy as np
from _style import FS, apply

apply()

# --- EDIT TITLE HERE ---
TITLE = "GNN v. IDW by Density Level on Flat Terrian"

N = [6, 8, 10, 12, 14, 16, 18, 20, 22]
idw = [4.203, 5.122, 3.692, 3.192, 3.359, 3.459, 2.997, 2.993, 3.021]
krig = [4.995, 5.596, 5.195, 3.401, 3.888, 3.837, 3.037, 3.300, 3.107]
idwp = [4.603, 5.476, 4.177, 3.368, 3.349, 3.334, 3.028, 3.210, 3.164]

BLUE = "#2a78d6"  # IDW baseline
AQUA = "#1baf7a"  # GNN + kriging prior
ORANGE = "#eb6834"  # GNN + IDW prior

fig, ax = plt.subplots(figsize=(8, 6))

# shade region where GNN lines lose (above IDW)
upper = np.maximum(krig, idwp)
ax.fill_between(
    N,
    idw,
    upper,
    where=(upper > np.array(idw)),
    interpolate=True,
    color="#e34948",
    alpha=0.12,
    zorder=1,
)

ax.plot(N, idw, color=BLUE, marker="o", ms=7, lw=2, label="IDW baseline", zorder=4)
ax.plot(
    N, krig, color=AQUA, marker="s", ms=7, lw=2, label="GNN + kriging prior", zorder=3
)
ax.plot(
    N, idwp, color=ORANGE, marker="^", ms=7, lw=2, label="GNN + IDW prior", zorder=3
)

ax.text(9, 5.55, "GNN loses here", color="#b02b2b", fontsize=FS, ha="left", va="bottom")

ax.set_xlabel("Number of sensors (N)", fontsize=FS)
ax.set_ylabel("MAE (µg/m³)", fontsize=FS)
ax.set_title(TITLE, fontsize=FS)
ax.set_xticks(N)
ax.legend(loc="upper right", fontsize=FS, frameon=True, framealpha=0.95)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#c3c2b7")
ax.spines["bottom"].set_color("#c3c2b7")
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#e1e0d9", lw=1)

fig.tight_layout()
out = "density_sweep_flat.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
