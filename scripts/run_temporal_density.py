"""
run_temporal_density.py  --  densify the deploy-or-not diagnostic (scripts/diagnostic.py).

The diagnostic fit had only 3 distinct networks (the 3 cities) -> n_effective ~ 3.
This driver subsamples each city to several densities x seeds, so each becomes a
DISTINCT network with its own (spatial_nrmse, realized-margin) pair. That turns the
diagnostic from a suggestive 3-point trend into a ~20-point regression spanning the
crossover threshold -- the difference between an anecdote and a rule.

For each (city, N, subsample_seed): thin the net to N sensors (kNN + kriging rebuilt
on the subset), run the WINNING temporal config (--temporal --rk-spatial, wind hrrr)
under 5-fold CV at gap=24h, record OURS vs ST-kriging margin, and the a-priori
spatial_nrmse of that exact subset. Appends rows to CSV as it goes.

USAGE
    PYTHONPATH=. .venv/bin/python scripts/run_temporal_density.py
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import warnings

import numpy as np

from src.graph import build_graph2 as bg
from src.model import train as tr
import scripts.eval_temporal as et
from scripts.diagnostic import spatial_loo_nrmse

warnings.filterwarnings("ignore")

GAP = 24
KFOLD = 5
# full N ~ pgh 59, slc 36, fresno 18; subsample below full, a couple seeds each.
PLAN = {
    "pittsburgh": ([12, 20, 30, 45], [0, 1]),
    "slc":        ([10, 16, 24],     [0, 1]),
    "fresno":     ([8, 12],          [0, 1]),
}


def winning_args():
    """Namespace matching the WORKLOG section-I winning config (defaults + the flags)."""
    return argparse.Namespace(
        city=None, sensor_set="urban", wind="hrrr",
        epochs=120, steps=96, hidden=16, layers=4, lr=0.01,
        gap_len=GAP, n_gaps=6, transform="sqrt", loss="huber", huber_beta=1.0,
        clip=1.0, corr_reg=0.6, no_convection=False, no_local=False,
        temporal=True, long_lags=False, temp_gate=False, aod_feature=False,
        elev_gate=False, elev_kernel=False, elev_kernel_h=150.0,
        rk_spatial=True, max_nodes=None, subsample_seed=0,
    )


def run_cell(city, N, ss):
    bg.use_city(city); bg.SENSOR_SET = "urban"; bg.EPA_CORRECT = True
    tr.WIND_SOURCE = "hrrr"; tr.STRICT_INPUTS = False; tr.USE_CACHE = False
    tr.SUBSAMPLE_N = N; tr.SUBSAMPLE_SEED = ss
    graph = tr.build_static_graph()

    ids, pm, observed = graph[0], graph[1], graph[2]
    elev, x_m, y_m = graph[7], graph[8], graph[9]
    n_actual = len(ids)

    # a-priori predictor on THIS subset (label-free, no training)
    values = np.clip(pm.to_numpy(dtype=np.float64), 0, None)
    z = np.sqrt(values)
    nrmse = spatial_loo_nrmse(z, observed.to_numpy(),
                              np.asarray(x_m, float), np.asarray(y_m, float))

    # realized margin: 5-fold CV, winning config
    args = winning_args(); args.city = city
    obs = observed.to_numpy(); T = len(pm)
    folds = et.kfold_gaps(obs, n_actual, T, GAP, KFOLD)
    ours, stks = [], []
    for k in range(KFOLD):
        if not folds[k].any():
            continue
        m, _mp, _b, mstk = et.run_seed(k, graph, args, masks=folds[k])
        ours.append(m["mae"]); stks.append(mstk["mae"])
    if not ours:
        return None
    om, sm = float(np.mean(ours)), float(np.mean(stks))
    margin = -(om / sm - 1.0) * 100.0                     # + = GNN beats ST-kriging
    won = sum(o < s for o, s in zip(ours, stks))
    return dict(city=city, N_requested=N, N_actual=n_actual, seed=ss,
                spatial_nrmse=round(nrmse, 4), ours_mae=round(om, 3),
                stk_mae=round(sm, 3), margin_pct=round(margin, 2),
                folds_won=f"{won}/{len(ours)}")


def main():
    out = pathlib.Path("experiments/logs/diagnostic")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "density_points.csv"
    fields = ["city", "N_requested", "N_actual", "seed", "spatial_nrmse",
              "ours_mae", "stk_mae", "margin_pct", "folds_won"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()

    for city, (densities, seeds) in PLAN.items():
        for N in densities:
            for ss in seeds:
                try:
                    row = run_cell(city, N, ss)
                except Exception as ex:
                    print(f"[skip] {city} N={N} ss={ss}: {ex}")
                    continue
                if row is None:
                    print(f"[skip] {city} N={N} ss={ss}: no folds")
                    continue
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)
                print(f"[done] {city:11s} N={row['N_actual']:>2} ss={ss}  "
                      f"nrmse={row['spatial_nrmse']:.3f}  margin={row['margin_pct']:+.2f}%  "
                      f"won={row['folds_won']}")
    print(f"\n[complete] -> {csv_path}")


if __name__ == "__main__":
    main()
