"""Aggregate per-seed logs from eval_graphy_faithful.py (run one seed per process,
in parallel) into the 8-seed mean +- std summary. Usage:
    python scripts/aggregate_faithful.py experiments/logs/faithful/<run>/seed_*.log
"""
import re
import sys

import numpy as np

pat = re.compile(
    r"seed (\d+): OURS mae=([\d.]+) rmse=([\d.]+) r2=([-\d.]+) corr=([-\d.]+) "
    r"\(n=(\d+)\)\s+IDW mae=([\d.]+)\s+w\(D,C,L\)=\(([\d.]+),([\d.]+),([\d.]+)\)(.*)"
)
rows = []
for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            mobj = pat.search(line)
            if mobj:
                g = mobj.groups()
                rows.append(dict(seed=int(g[0]), mae=float(g[1]), rmse=float(g[2]),
                                 r2=float(g[3]), corr=float(g[4]), n=int(g[5]),
                                 idw=float(g[6]), wD=float(g[7]), wC=float(g[8]),
                                 wL=float(g[9]), diverged="DIVERGED" in g[10]))

if not rows:
    print("no seed result lines found in:", sys.argv[1:]); sys.exit(1)
rows.sort(key=lambda r: r["seed"])
seeds = [r["seed"] for r in rows]


def ms(key):
    v = np.array([r[key] for r in rows]); return v.mean(), v.std()


print("=" * 72)
print(f"FAITHFUL GraPhy  aggregated over {len(rows)} seeds {seeds}")
print("=" * 72)
for r in rows:
    d = "  DIVERGED" if r["diverged"] else ""
    print(f"  seed {r['seed']}: MAE={r['mae']:.3f}  R2={r['r2']:.3f}  "
          f"corr={r['corr']:.3f}  IDW={r['idw']:.3f}  "
          f"w=({r['wD']:.2f},{r['wC']:.2f},{r['wL']:.2f}){d}")
mae_m, mae_s = ms("mae"); r2_m, r2_s = ms("r2"); rmse_m, rmse_s = ms("rmse")
idw_m, idw_s = ms("idw")
print("-" * 72)
print(f"FAITHFUL GraPhy   MAE={mae_m:.3f}+-{mae_s:.3f}  RMSE={rmse_m:.3f}+-{rmse_s:.3f}  "
      f"R2={r2_m:.3f}+-{r2_s:.3f}")
print(f"IDW baseline      MAE={idw_m:.3f}+-{idw_s:.3f}")
print(f"repo best (ref)   MAE=3.054+-0.29   GraPhy(paper) MAE=2.380")
print(f"mean fusion  w_D={np.mean([r['wD'] for r in rows]):.3f}  "
      f"w_C={np.mean([r['wC'] for r in rows]):.3f}  "
      f"w_L={np.mean([r['wL'] for r in rows]):.3f}")
print(f"diverged seeds: {sum(r['diverged'] for r in rows)}/{len(rows)}")
print("=" * 72)
