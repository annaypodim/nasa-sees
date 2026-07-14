#!/usr/bin/env python
"""Bar chart of the converged SLC+wind 2x2 ablation vs IDW."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, MAE, std)  -- converged 4-seed results, SLC+wind
data = [
    ("A\npure\nterrain OFF",   5.406, 1.258),
    ("B\npure\nterrain ON",    6.583, 2.853),
    ("C\nhybrid\nterrain OFF", 4.896, 1.111),
    ("D\nhybrid\nterrain ON",  4.863, 0.599),
]
IDW = 4.753
labels = [d[0] for d in data]
maes   = [d[1] for d in data]
stds   = [d[2] for d in data]
colors = ["#c96", "#c66", "#69c", "#39a"]

fig, ax = plt.subplots(figsize=(8.4, 5.4))
x = np.arange(len(data))
bars = ax.bar(x, maes, yerr=stds, capsize=6, color=colors,
              edgecolor="black", linewidth=0.8, width=0.62,
              error_kw=dict(ecolor="#333", lw=1.4))
ax.axhline(IDW, ls="--", lw=1.6, color="#2a9d4a")
ax.text(len(data)-0.35, IDW + 0.06, f"IDW baseline = {IDW:.2f}",
        color="#22803b", fontsize=10, ha="right", va="bottom")

for xi, (m, s) in enumerate(zip(maes, stds)):
    ax.text(xi, m + s + 0.12, f"{m:.2f}\n±{s:.2f}", ha="center",
            va="bottom", fontsize=9.5)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("test MAE (ug/m3)  — lower is better")
ax.set_title("SLC + wind, converged 2x2: terrain gates & IDW-anchoring\n"
             "(4 seeds; terrain helps only the anchored hybrid — and via variance, not mean)")
ax.set_ylim(0, 10.2)
ax.grid(axis="y", alpha=0.25)
# annotate the two key deltas
ax.annotate("terrain on PURE:\n+1.18 MAE, variance 2x  (HURTS)",
            xy=(1, 6.583+2.853), xytext=(1.05, 9.4), fontsize=8.5, color="#a33",
            ha="center")
ax.annotate("terrain on HYBRID:\nvariance halved (robustness win)",
            xy=(3, 4.863+0.599), xytext=(3, 7.2), fontsize=8.5, color="#1a6",
            ha="center", arrowprops=dict(arrowstyle="->", color="#1a6", lw=1.2))
fig.tight_layout()
out = "experiments/logs/faithful/slc_2x2_ablation.png"
fig.savefig(out, dpi=140)
print("wrote", out)
