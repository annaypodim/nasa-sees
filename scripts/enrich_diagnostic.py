"""
enrich_diagnostic.py  --  add more a-priori predictors to the 18 subsampled networks
in experiments/logs/diagnostic/density_points.csv (NO training; just rebuild each
subsampled graph and compute label-free stats), then refit the margin on each single
predictor and on the best 2-feature pair. Answers: is spatial_nrmse the best a-priori
predictor, or does density / spacing / relief do better once we have 18 real networks?

USAGE
    PYTHONPATH=. .venv/bin/python scripts/enrich_diagnostic.py
"""
from __future__ import annotations
import csv, itertools, pathlib, warnings
import numpy as np
from src.graph import build_graph2 as bg
from src.model import train as tr
from scripts.diagnostic import (convex_hull_area, mean_nn_km, temporal_stats,
                                 spatial_loo_nrmse, ols_fit)

warnings.filterwarnings("ignore")
CSV = pathlib.Path("experiments/logs/diagnostic/density_points.csv")


def stats_for(city, N, ss):
    bg.use_city(city); bg.SENSOR_SET = "urban"; bg.EPA_CORRECT = True
    tr.WIND_SOURCE = "hrrr"; tr.STRICT_INPUTS = False; tr.USE_CACHE = False
    tr.SUBSAMPLE_N = N; tr.SUBSAMPLE_SEED = ss
    g = tr.build_static_graph()
    ids, pm, observed = g[0], g[1], g[2]
    elev, x_m, y_m = g[7], g[8], g[9]
    x = np.asarray(x_m, float); y = np.asarray(y_m, float)
    z = np.sqrt(np.clip(pm.to_numpy(dtype=np.float64), 0, None))
    obs = observed.to_numpy()
    area = convex_hull_area(x, y)
    rho24, vario24 = temporal_stats(z, obs, 24)
    return dict(
        N=len(ids),
        density=len(ids) / area if area and not np.isnan(area) else np.nan,
        mean_nn_km=mean_nn_km(x, y),
        relief_m=float(np.std(np.asarray(elev, float))),
        spatial_nrmse=spatial_loo_nrmse(z, obs, x, y),
        rho24=rho24, vario24=vario24,
    )


def main():
    rows = list(csv.DictReader(open(CSV)))
    enriched = []
    for r in rows:
        city, N, ss = r["city"], int(r["N_requested"]), int(r["seed"])
        try:
            s = stats_for(city, N, ss)
        except Exception as ex:
            print(f"[skip] {city} N={N} ss={ss}: {ex}"); continue
        s["margin"] = float(r["margin_pct"]); s["city"] = city
        enriched.append(s)
        print(f"[stat] {city:11s} N={s['N']:>2} ss={ss}  nrmse={s['spatial_nrmse']:.3f} "
              f"dens={s['density']:.3f} nn={s['mean_nn_km']:.2f} relief={s['relief_m']:.0f} "
              f"margin={s['margin']:+.2f}")

    preds = ["spatial_nrmse", "density", "mean_nn_km", "relief_m", "rho24", "vario24"]
    y = [e["margin"] for e in enriched]
    print(f"\nSINGLE-predictor OLS across {len(enriched)} networks:")
    print(f"{'predictor':>14}  {'R2':>6}  {'Spearman':>9}")
    singles = {}
    for p in preds:
        f = ols_fit([e[p] for e in enriched], y); singles[p] = f
        print(f"{p:>14}  {f['r2']:>6.3f}  {f['spearman']:>9.3f}" if f else f"{p:>14}   --")

    # best 2-feature pair (plain multiple OLS, R2)
    def r2_multi(cols):
        X = np.array([[e[c] for c in cols] for e in enriched], float)
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X, yy = X[ok], np.array(y)[ok]
        if len(yy) < len(cols) + 2:
            return None
        A = np.hstack([X, np.ones((len(X), 1))])
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        pred = A @ coef
        ss_res = ((yy - pred) ** 2).sum(); ss_tot = ((yy - yy.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"\nBEST 2-feature pairs (multiple OLS R2):")
    pairs = sorted(((r2_multi(list(c)), c) for c in itertools.combinations(preds, 2)
                    if r2_multi(list(c)) is not None), key=lambda t: -t[0])
    for r2, c in pairs[:5]:
        print(f"  {r2:>6.3f}   {c[0]} + {c[1]}")


if __name__ == "__main__":
    main()
