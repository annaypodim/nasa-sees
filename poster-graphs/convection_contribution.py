"""Poster figure: convection near-zero contribution -> wind uninformative.

MEASURED fusion module shares from real eval runs (fusion_shares.json), captured
by poster-graphs/measure_fusion.py: the per-node softmax weights averaged over
every eval forward pass (3 seeds, 120 epochs). Column order [diffusion,
convection, local] matches model.py's active-module order.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import FS, apply

apply()

# --- EDIT TITLE HERE ---
TITLE = "Physics Module Partial Contributions"

shares = json.loads(
    (Path(__file__).resolve().parent / "fusion_shares.json").read_text()
)
CITY_LABEL = {"fresno": "Fresno", "slc": "Salt Lake City"}
segs = ["Diffusion", "Convection", "Local"]
seg_keys = ["diffusion", "convection", "local"]
bars = [CITY_LABEL[c] for c in shares]
vals = np.array([[shares[c][k] for k in seg_keys] for c in shares])
COLORS = ["#2a78d6", "#898781", "#1baf7a"]  # convection = recessive gray
COLORS = ["#2a78d6", "#898781", "#1baf7a"]  # convection = recessive gray

fig, ax = plt.subplots(figsize=(9, 4))
y = np.arange(len(bars))
left = np.zeros(len(bars))
for j, (seg, color) in enumerate(zip(segs, COLORS)):
    w = vals[:, j]
    ax.barh(
        y,
        w,
        left=left,
        color=color,
        height=0.55,
        edgecolor="white",
        linewidth=2,
        zorder=3,
    )
    for i in range(len(bars)):
        if w[i] > 0.03:
            ax.text(
                left[i] + w[i] / 2,
                y[i],
                f"{w[i] * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=FS,
                color="white",
                zorder=4,
            )
    left += w

ax.set_yticks(y)
ax.set_yticklabels(bars, fontsize=FS)
ax.invert_yaxis()
ax.set_xlim(0, max(1.0, left.max()))
ax.set_xlabel("Share of correction (fraction)", fontsize=FS)
ax.set_title(TITLE, fontsize=FS)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.legend(
    segs,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.42),
    ncol=3,
    fontsize=FS,
    frameon=True,
    framealpha=0.95,
)

fig.tight_layout()
out = "convection_contribution.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
