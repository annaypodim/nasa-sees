"""FIGURE 4 -- covariates are NULL on the temporal task too.

Bars = Delta MAE vs the winning temporal model when each covariate is added
(temperature gate, AOD node-feature) or toggled (wind hrrr-vs-zero), under the same
5-fold CV. Everything sits inside a ~+/-1% noise band and mostly the WRONG way ->
temp/AOD/wind add nothing once the sensor's own persistence+lags are present.
Source: experiments/logs/temporal_covariates/results.csv (cherry-picked, WORKLOG J).
"""
import csv, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _style import apply, FS

apply()
REPO = pathlib.Path(__file__).resolve().parents[1]
CSV = REPO / "experiments/logs/temporal_covariates/results.csv"

rows = list(csv.DictReader(open(CSV)))
# (label, delta%, covariate) — temp/AOD Pittsburgh-only; wind all 3 cities
COL = {"temp": "#B8892A", "aod": "#7A66AE", "wind": "#3F6FB0"}
bars = []
for r in rows:
    v = r["variant"]
    if v == "base":
        continue
    d = float(r["delta_vs_baseline_pct"])
    tag = f"{r['city'][:3]} {r['gap_len']}h"
    if v == "temp":
        bars.append(("temp", f"temp\n{tag}", d))
    elif v == "aod":
        bars.append(("aod", f"AOD\n{tag}", d))
    elif v == "wind_hrrr_vs_zero":
        bars.append(("wind", f"wind\n{tag}", d))

# order: temp, aod, wind
order = {"temp": 0, "aod": 1, "wind": 2}
bars.sort(key=lambda b: (order[b[0]], b[1]))

fig, ax = plt.subplots(figsize=(8.2, 4.4))
xs = range(len(bars))
ax.axhspan(-1, 1, color="#dddddd", alpha=0.5, zorder=0)          # ~noise floor
ax.axhline(0, color="#444", lw=1.2, zorder=1)
ax.bar(list(xs), [b[2] for b in bars], color=[COL[b[0]] for b in bars],
       edgecolor="white", lw=0.8, zorder=2, width=0.72)
ax.set_xticks(list(xs))
ax.set_xticklabels([b[1] for b in bars], fontsize=FS-2)
ax.set_ylabel("Δ MAE vs baseline (%)")
ax.set_ylim(-1.6, 1.6)
ax.set_title("Covariates are null on the temporal task  (no consistent gain, |Δ| at most ~1%)")
ax.text(0.985, 0.05, "shaded = ±1% fold-noise band;  >0 = covariate HURTS",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=FS-3, color="#555")
# legend
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=COL["temp"], label="temperature gate (Pgh only)"),
                   mp.Patch(color=COL["aod"], label="AOD feature (Pgh only)"),
                   mp.Patch(color=COL["wind"], label="wind on vs off (3 cities)")],
          loc="upper left", fontsize=FS-3, framealpha=0.9)
fig.tight_layout()
out = REPO / "poster-graphs/fig4_covariate_null.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print(f"[saved] {out}")
