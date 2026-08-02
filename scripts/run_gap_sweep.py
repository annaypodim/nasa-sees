"""
run_gap_sweep.py  --  expand the diagnostic's range along the TEMPORAL axis.

The 18-network diagnostic fixed gap=24h, so every point WON (margins 1.8-11.5%) and
the fit was a line through a narrow band (R2~0.39, crossover un-observed). Gap length
is the missing knob: at VERY SHORT gaps persistence is near-perfect -> the ST-kriging
baseline is near-optimal -> the GNN's margin should collapse toward ZERO (maybe cross
it). That gives us the low end / an actual observed crossover, and a second predictor
axis (temporal staleness) for a 2-feature diagnostic.

For each (full-network city, gap in {6,12,24,48,72}): run the winning config under
5-fold CV; record the realized margin plus two a-priori predictors:
  spatial_nrmse  (per city, gap-independent: spatial interpolation headroom)
  stale_gap      = 1 - autocorr(gap)  (per gap: how stale persistence gets = temporal
                   headroom the GNN can exploit over persistence)

USAGE
    PYTHONPATH=. .venv/bin/python scripts/run_gap_sweep.py
"""
from __future__ import annotations
import csv, pathlib, warnings
import numpy as np
from src.graph import build_graph2 as bg
from src.model import train as tr
import scripts.eval_temporal as et
from scripts.run_temporal_density import winning_args
from scripts.diagnostic import spatial_loo_nrmse, temporal_stats

warnings.filterwarnings("ignore")
CITIES = ["pittsburgh", "fresno", "slc"]
GAPS = [6, 12, 24, 48, 72]
KFOLD = 5


def run_city_gaps(city, out_path, fields):
    bg.use_city(city); bg.SENSOR_SET = "urban"; bg.EPA_CORRECT = True
    tr.WIND_SOURCE = "hrrr"; tr.STRICT_INPUTS = False; tr.USE_CACHE = False
    tr.SUBSAMPLE_N = None
    graph = tr.build_static_graph()
    ids, pm, observed = graph[0], graph[1], graph[2]
    x_m, y_m = graph[8], graph[9]
    N, T = len(ids), len(pm)
    z = np.sqrt(np.clip(pm.to_numpy(dtype=np.float64), 0, None))
    obs = observed.to_numpy()
    nrmse = spatial_loo_nrmse(z, obs, np.asarray(x_m, float), np.asarray(y_m, float))

    for gap in GAPS:
        rho, vario = temporal_stats(z, obs, gap)
        stale = 1.0 - rho if np.isfinite(rho) else np.nan
        args = winning_args(); args.city = city; args.gap_len = gap
        folds = et.kfold_gaps(obs, N, T, gap, KFOLD)
        ours, stks = [], []
        for k in range(KFOLD):
            if not folds[k].any():
                continue
            m, _mp, _b, mstk = et.run_seed(k, graph, args, masks=folds[k])
            ours.append(m["mae"]); stks.append(mstk["mae"])
        if not ours:
            print(f"[skip] {city} gap={gap}: no folds"); continue
        om, sm = float(np.mean(ours)), float(np.mean(stks))
        margin = -(om / sm - 1.0) * 100.0
        won = sum(o < s for o, s in zip(ours, stks))
        row = dict(city=city, gap=gap, spatial_nrmse=round(nrmse, 4),
                   stale_gap=round(stale, 4), ours_mae=round(om, 3),
                   stk_mae=round(sm, 3), margin_pct=round(margin, 2),
                   folds_won=f"{won}/{len(ours)}")
        with open(out_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"[done] {city:11s} gap={gap:>2}h  nrmse={nrmse:.3f} stale={stale:.3f}  "
              f"margin={margin:+.2f}%  won={won}/{len(ours)}")


def main():
    out = pathlib.Path("experiments/logs/diagnostic/gap_points.csv")
    fields = ["city", "gap", "spatial_nrmse", "stale_gap", "ours_mae", "stk_mae",
              "margin_pct", "folds_won"]
    with open(out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    for city in CITIES:
        try:
            run_city_gaps(city, out, fields)
        except Exception as ex:
            print(f"[skip city] {city}: {ex}")
    print(f"\n[complete] -> {out}")


if __name__ == "__main__":
    main()
