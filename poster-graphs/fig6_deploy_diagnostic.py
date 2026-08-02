"""FIGURE 6 -- deploy-or-not diagnostic (WORKLOG K3).

LEFT : GNN margin over strong ST-kriging vs gap length, one line per city. The edge
       is an inverted-U (peak ~24-48h); on sparse SLC at 6h it goes NEGATIVE -- the
       OBSERVED CROSSOVER (circled): the one regime where the learned model loses.
RIGHT: margin vs a-priori sensor density across 18 subsampled networks + OLS
       (R2~0.67) -- density predicts the payoff before any training.
Sources: experiments/logs/diagnostic/gap_points.csv  and  .../enrich.log
"""
import csv, pathlib, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _style import apply, FS

apply()
REPO = pathlib.Path(__file__).resolve().parents[1]
GAP = REPO / "experiments/logs/diagnostic/gap_points.csv"
ENRICH = REPO / "experiments/logs/diagnostic/enrich.log"
CITYCOL = {"pittsburgh": "#3F6FB0", "fresno": "#3F7E4E", "slc": "#B23B3B"}
CITYLAB = {"pittsburgh": "Pittsburgh (dense)", "fresno": "Fresno (medium)",
           "slc": "SLC (sparse terrain)"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.6))

# ---- LEFT: margin vs gap length, per city ----
rows = list(csv.DictReader(open(GAP)))
by_city = {}
for r in rows:
    by_city.setdefault(r["city"], []).append((int(r["gap"]), float(r["margin_pct"])))
axL.axhline(0, color="#c33", lw=1.2, ls="--", zorder=1)
for city, pts in by_city.items():
    pts.sort()
    gx = [p[0] for p in pts]; my = [p[1] for p in pts]
    axL.plot(gx, my, "-o", color=CITYCOL[city], lw=2, ms=6, label=CITYLAB[city], zorder=3)
# circle the observed crossover: SLC 6h
cx = [p for p in by_city["slc"] if p[0] == 6][0]
axL.scatter([cx[0]], [cx[1]], s=260, facecolor="none", edgecolor="#B23B3B",
            lw=2.2, zorder=4)
axL.annotate("observed crossover\nSLC 6h: −3.0%, 0/5 folds\n(GNN not worth it)",
             (cx[0], cx[1]), xytext=(18, -6), textcoords="offset points",
             fontsize=FS-3, color="#B23B3B",
             arrowprops=dict(arrowstyle="->", color="#B23B3B", lw=1.2))
axL.set_xscale("log")
axL.set_xticks([6, 12, 24, 48, 72])
axL.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axL.minorticks_off()
axL.set_xlabel("gap length (hours, log scale)")
axL.set_ylabel("GNN margin over ST-kriging (%)   ·   >0 = GNN wins")
axL.set_title("Where the correction pays: inverted-U in gap length")
axL.legend(loc="upper right", fontsize=FS-3)

# ---- RIGHT: margin vs density across 18 subsampled networks ----
dens, marg, cols = [], [], []
pat = re.compile(r"\[stat\]\s+(\w+)\s+N=\s*\d+\s+ss=\d+\s+nrmse=[\d.]+\s+dens=([\d.]+).*margin=([+\-][\d.]+)")
for line in open(ENRICH):
    m = pat.search(line)
    if m:
        dens.append(float(m.group(2))); marg.append(float(m.group(3)))
        cols.append(CITYCOL[m.group(1)])
dens, marg = np.array(dens), np.array(marg)
# OLS + R2
A = np.vstack([dens, np.ones_like(dens)]).T
(slope, b), *_ = np.linalg.lstsq(A, marg, rcond=None)
pred = slope * dens + b
r2 = 1 - ((marg - pred) ** 2).sum() / ((marg - marg.mean()) ** 2).sum()
gx = np.linspace(dens.min(), dens.max(), 50)
axR.plot(gx, slope * gx + b, "-", color="#888", lw=1.6, zorder=1,
         label=f"OLS  R²={r2:.2f}  (n={len(dens)})")
axR.scatter(dens, marg, c=cols, s=70, edgecolor="white", lw=0.8, zorder=3)
axR.axhline(0, color="#c33", lw=1, ls="--", zorder=0)
axR.set_xlabel("sensor density  (nodes / km²)  ·  a-priori, label-free")
axR.set_ylabel("GNN margin over ST-kriging (%)")
axR.set_title("Density predicts the payoff, before training")
axR.legend(loc="lower right", fontsize=FS-3)

fig.tight_layout()
out = REPO / "poster-graphs/fig6_deploy_diagnostic.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print(f"[saved] {out}  (density R2={r2:.3f})")
