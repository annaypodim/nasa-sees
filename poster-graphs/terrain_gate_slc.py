"""Poster figure: terrain-aware correction flips loss into a win (Salt Lake City)."""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch

# --- data ---
labels = ["Base,\ngate off", "Base,\ngate on", "Hybrid,\nterrain off", "Hybrid,\nterrain on"]
values = [5.49, 4.96, 4.80, 4.19]
errors = [1.15, 0.94, 0.0, 0.0]
IDW = 4.6

# --- palette (validated defaults; lower MAE is better) ---
GREEN = "#008300"   # beats IDW (below the line)
GRAY  = "#898781"   # loses to IDW
INK   = "#0b0b0b"
SECOND = "#52514e"
colors = [GREEN if v < IDW else GRAY for v in values]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

fig, ax = plt.subplots(figsize=(11, 8))
x = range(len(values))
yerr = [[0, 0, 0, 0], errors]  # error bars only upward where available
bars = ax.bar(x, values, width=0.62, color=colors,
              yerr=yerr, capsize=8,
              error_kw=dict(ecolor=SECOND, elinewidth=2, capthick=2), zorder=3)

# IDW reference line
ax.axhline(IDW, ls="--", lw=2.5, color=INK, zorder=2)
ax.text(len(values) - 0.5, IDW + 0.06, "IDW baseline",
        ha="right", va="bottom", fontsize=15, style="italic", color=INK)

# value labels on bars
for xi, v, e in zip(x, values, errors):
    ax.text(xi, v + (e if e else 0) + 0.12, f"{v:.2f}",
            ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK)

# bracket + annotation for 4.80 -> 4.19
#y_br = 5.15
#ax.annotate("", xy=(3, y_br), xytext=(2, y_br),
#            arrowprops=dict(arrowstyle="-", lw=2, color=INK))
#ax.plot([2, 2], [values[2] + 0.35, y_br], lw=2, color=INK)
#ax.plot([3, 3], [values[3] + 0.35, y_br], lw=2, color=INK)
#ax.text(2.5, y_br + 0.08, "−12.7%", ha="center", va="bottom",
#        fontsize=15, fontweight="bold", color=GREEN)

# axes styling
ax.set_ylabel("MAE (µg/m³)", fontsize=15)
ax.set_title("Terrain-Aware Correction Flips Loss into a Win\n(Salt Lake City)",
             fontsize=15, fontweight="bold", pad=16)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=15)
ax.tick_params(axis="y", labelsize=15)
ax.set_ylim(0, 7)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#c3c2b7")
ax.spines["bottom"].set_color("#c3c2b7")
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#e1e0d9", lw=1)

fig.tight_layout()
out = "terrain_gate_slc.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
