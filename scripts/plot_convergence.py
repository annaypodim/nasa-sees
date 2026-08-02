#!/usr/bin/env python
"""Plot val-MAE convergence curve(s) from val_curve_seed*.csv files."""
import sys, csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = sys.argv[1:]
out = next((a for a in args if a.endswith(".png")), "experiments/logs/faithful/convergence_curve.png")
paths = [a for a in args if a.endswith(".csv")] or \
        sorted(glob.glob("experiments/logs/faithful/curve_*/val_curve_seed*.csv"))
fig, ax = plt.subplots(figsize=(9, 5.2))

for p in paths:
    steps, val, best = [], [], []
    with open(p) as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["step"])); val.append(float(row["val_mae"]))
            best.append(float(row["running_best"]))
    lbl = os.path.basename(os.path.dirname(p))
    ax.plot(steps, val, lw=1.2, alpha=0.55, color="#3b7dd8", label=f"{lbl}: val MAE (raw)")
    ax.plot(steps, best, lw=2.4, color="#d8453b", label=f"{lbl}: best-so-far")
    bi = min(range(len(best)), key=lambda i: best[i])
    ax.scatter([steps[bi]], [best[bi]], s=70, zorder=5, color="#d8453b",
               edgecolor="white", linewidth=1.3)
    ax.annotate(f"best {best[bi]:.2f}@{steps[bi]}",
                (steps[bi], best[bi]), textcoords="offset points",
                xytext=(8, -14), fontsize=10, color="#b8342b")

# reference baselines
ax.axhline(3.229, ls="--", lw=1.3, color="#2a9d4a", label="IDW baseline (seed 0) = 3.23")
ax.axhline(4.52, ls=":", lw=1.3, color="#888",
           label="pure terrain-OFF seed 0 best-val = 4.52")

ax.set_xlabel("training step"); ax.set_ylabel("validation MAE (ug/m3)")
ax.set_title("Faithful GraPhy convergence — SLC, terrain-gates ON, wind ON (seed 0)")
ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="upper right")
ax.set_ylim(2.8, min(11, max(v for p in paths for v in
    [float(r["val_mae"]) for r in csv.DictReader(open(p))]) * 1.05))
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
