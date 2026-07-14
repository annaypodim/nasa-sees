"""
run_temporal_covariates.py  -  paired covariate test on the TEMPORAL gap-fill task.

Drives eval_temporal.run_seed under proper K-fold CV, running a BASELINE and one or
more covariate VARIANTS on the SAME folds and SAME per-fold seeds (fold index k), so
the only thing that differs between baseline and variant is the covariate flag. This
is a strictly paired comparison: per fold we get (ours_baseline, ours_variant) and can
count how many folds the covariate actually improves.

Baseline model = the project's strongest reported config:  --temporal --rk-spatial.
The prior-only prediction is the ST-kriging baseline (reported by run_seed as m_stk).

Covariate variants (only run where the data exists):
    temp   -> args.temp_gate   (per-node temperature gate; pittsburgh only)
    aod    -> args.aod_feature  (per-node AOD node feature; pittsburgh only)
    wind   -> handled by running the whole thing under --wind zero vs --wind hrrr
              (wind never changes the node/fold set, so those two are also paired)
    windfeat -> args.wind_feature (per-node standardized wind-speed node feature)

Output: one JSON blob per (city, gap, variant) to stdout (tagged RESULT:) plus a
human table. The caller aggregates the RESULT: lines into the CSV.
"""
from __future__ import annotations

import argparse
import json
import types

import numpy as np

from src.graph import build_graph2 as bg
from src.model import train as tr
from scripts import eval_temporal as et


BASE_ARGS = dict(
    city="pittsburgh", sensor_set="urban", wind="hrrr", seeds="0,1,2,3", kfold=5,
    epochs=120, steps=96, hidden=16, layers=4, lr=0.01, gap_len=24, n_gaps=6,
    transform="sqrt", loss="huber", huber_beta=1.0, clip=1.0, corr_reg=0.6,
    no_convection=False, no_local=False, temporal=True, long_lags=False,
    temp_gate=False, aod_feature=False, wind_feature=False, elev_gate=False,
    elev_kernel=False, elev_kernel_h=150.0, rk_spatial=True,
)


def make_args(**over):
    d = dict(BASE_ARGS)
    d.update(over)
    return types.SimpleNamespace(**d)


def run_variant(graph, folds, gap_len, **over):
    """Run all K folds for one variant; return per-fold ours/stk mae arrays."""
    args = make_args(gap_len=gap_len, **over)
    ours, stk = [], []
    for k in range(len(folds)):
        m, mp, b, mstk = et.run_seed(k, graph, args, masks=folds[k])
        ours.append(m["mae"]); stk.append(mstk["mae"])
    return np.array(ours), np.array(stk)


def emit(city, gap_len, variant, ours, stk, base_ours):
    delta = (ours.mean() / base_ours.mean() - 1.0) * 100.0 if base_ours is not None else 0.0
    # folds where THIS variant's ours beats the baseline's ours (paired, per fold)
    if base_ours is not None:
        improved = int(np.sum(ours < base_ours))
    else:
        improved = int(np.sum(ours < ours) )  # n/a for the baseline row -> 0
    rec = dict(
        city=city, gap_len=int(gap_len), variant=variant,
        ours_mae=round(float(ours.mean()), 4), ours_std=round(float(ours.std()), 4),
        stk_mae=round(float(stk.mean()), 4),
        baseline_ours_mae=(round(float(base_ours.mean()), 4) if base_ours is not None else round(float(ours.mean()), 4)),
        delta_vs_baseline_pct=round(float(delta), 3),
        folds_improved_over_baseline=(improved if base_ours is not None else "n/a"),
        ours_per_fold=[round(float(x), 4) for x in ours],
    )
    print("RESULT: " + json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--sensor-set", default="urban")
    ap.add_argument("--wind", choices=["hrrr", "zero"], default="hrrr")
    ap.add_argument("--gap-len", type=int, required=True)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--variants", default="base",
                    help="comma list from {base,temp,aod,windfeat}. base is always run first.")
    args = ap.parse_args()

    bg.use_city(args.city); bg.SENSOR_SET = args.sensor_set; bg.EPA_CORRECT = True
    tr.WIND_SOURCE = args.wind; tr.STRICT_INPUTS = (args.wind != "zero"); tr.USE_CACHE = False
    graph = tr.build_static_graph()
    (ids, pm, observed, *_rest) = graph
    obs = observed.to_numpy()
    N, T = len(ids), len(pm)
    folds = et.kfold_gaps(obs, N, T, args.gap_len, args.kfold)
    print(f"[cov] city={args.city} wind={args.wind} gap={args.gap_len} N={N} "
          f"folds={args.kfold} ids_hash={hash(tuple(ids)) & 0xffffffff}", flush=True)

    common = dict(epochs=args.epochs, kfold=args.kfold, wind=args.wind,
                  city=args.city, sensor_set=args.sensor_set)

    # BASELINE (no covariate) -- tagged with the wind source so wind isolation pairs
    b_ours, b_stk = run_variant(graph, folds, args.gap_len, **common)
    base_tag = f"base_wind_{args.wind}"
    emit(args.city, args.gap_len, base_tag, b_ours, b_stk, None)

    wanted = [v.strip() for v in args.variants.split(",") if v.strip() and v.strip() != "base"]
    flag = {"temp": dict(temp_gate=True), "aod": dict(aod_feature=True),
            "windfeat": dict(wind_feature=True)}
    for v in wanted:
        try:
            o, s = run_variant(graph, folds, args.gap_len, **common, **flag[v])
        except SystemExit as e:
            print("RESULT: " + json.dumps(dict(
                city=args.city, gap_len=int(args.gap_len), variant=v,
                ours_mae="N/A - no data", ours_std="N/A", stk_mae="N/A",
                baseline_ours_mae=round(float(b_ours.mean()), 4),
                delta_vs_baseline_pct="N/A", folds_improved_over_baseline="N/A",
                note=str(e))), flush=True)
            continue
        emit(args.city, args.gap_len, v, o, s, b_ours)


if __name__ == "__main__":
    main()
