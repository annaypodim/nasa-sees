"""
diagnostic.py  --  a DEPLOY-OR-NOT rule for the learned GNN correction.

THE CONTRIBUTION (the novel bit):
    IGNNK-style work says qualitatively "a learned graph model beats kriging more
    when the field is richer." Nobody publishes a NUMBER: an a-priori statistic you
    can compute from an UNLABELLED sensor network, before training anything, that
    predicts whether the GNN correction will actually beat a strong space+time
    kriging baseline -- and the crossover THRESHOLD below which you should not bother.

    This script computes, per city, only label-free statistics of the OBSERVED data
    (no training, no held-out labels), pairs them with the realized 5-fold-CV margin
    (OURS vs ST-kriging, from WORKLOG section I), and fits the margin as a function of
    each candidate statistic. The best single predictor + its zero-crossing is the
    deploy-or-not rule.

    Candidate a-priori statistics (all computed on KNOWN cells only):
      N              network size
      density        nodes per km^2 (N / convex-hull area)
      mean_nn_km     mean nearest-neighbour spacing
      relief_m       spatial std of sensor elevation (terrain relief)
      rho_h          temporal autocorrelation at the gap horizon h (persistence
                     quality: corr(z[t], z[t+h]) averaged over sensors)
      vario_h        temporal semivariance at horizon h: mean (z[t+h]-z[t])^2
      spatial_nrmse  leave-one-out IDW error on observed cells / field std
                     (how much spatial structure the prior ALREADY captures)
      richness       proposed compound index = log10(N) * rho_h
                     (graph structure x temporally-predictable structure)

USAGE
    PYTHONPATH=. .venv/bin/python scripts/diagnostic.py
"""
from __future__ import annotations

import warnings
import numpy as np

from src.graph import build_graph2 as bg
from src.model import train as tr

warnings.filterwarnings("ignore")

# Realized 5-fold-CV margins from WORKLOG section (I): OURS vs ST-kriging.
# improvement% = -(OURS/STK - 1)*100, i.e. POSITIVE = GNN beats the strong baseline.
# All six cells won 5/5 folds.
MARGINS = {
    ("pittsburgh", 24): 9.9, ("pittsburgh", 48): 8.2,
    ("fresno",     24): 7.2, ("fresno",     48): 6.1,
    ("slc",        24): 3.3, ("slc",        48): 1.8,
}
CITIES = ["pittsburgh", "fresno", "slc"]
GAPS = [24, 48]


def convex_hull_area(x, y):
    """Shoelace area of the 2-D convex hull, km^2 (x,y in metres). Monotone chain."""
    pts = sorted(set(zip(x.tolist(), y.tolist())))
    if len(pts) < 3:
        return np.nan
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i+1) % len(hull)]
        area += x1*y2 - x2*y1
    return abs(area) / 2.0 / 1e6  # m^2 -> km^2


def mean_nn_km(x, y):
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1).mean() / 1e3


def temporal_stats(z, obs, h):
    """rho_h = mean over sensors of corr(z[t], z[t+h]) on pairs both observed;
    vario_h = mean over sensors of mean((z[t+h]-z[t])^2) on those pairs."""
    T, N = z.shape
    rhos, varios = [], []
    for n in range(N):
        c = obs[:, n]
        pair = c[:-h] & c[h:]
        if pair.sum() < 30:
            continue
        a = z[:-h, n][pair]
        b = z[h:, n][pair]
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        rhos.append(np.corrcoef(a, b)[0, 1])
        varios.append(np.mean((b - a) ** 2))
    return (float(np.nanmean(rhos)) if rhos else np.nan,
            float(np.nanmean(varios)) if varios else np.nan)


def spatial_loo_nrmse(z, obs, x, y, n_hours=200, seed=0):
    """Leave-one-out IDW error on OBSERVED cells, averaged over sampled hours,
    normalised by the field's std. Low = field is spatially compressible (prior
    already explains it -> little room for a learned correction)."""
    rng = np.random.default_rng(seed)
    T, N = z.shape
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    w = 1.0 / (d + 1.0)
    np.fill_diagonal(w, 0.0)
    hours = np.where(obs.sum(axis=1) >= 4)[0]
    if len(hours) == 0:
        return np.nan
    hours = rng.choice(hours, size=min(n_hours, len(hours)), replace=False)
    errs, scale = [], []
    for t in hours:
        vis = obs[t]
        idx = np.where(vis)[0]
        if len(idx) < 4:
            continue
        zt = z[t]
        for n in idx:
            m = vis.copy(); m[n] = False
            num = (w[n] * zt * m).sum()
            den = (w[n] * m).sum()
            if den <= 0:
                continue
            errs.append(abs(zt[n] - num / den))
        scale.append(zt[idx].std())
    if not errs:
        return np.nan
    return float(np.mean(errs) / (np.mean(scale) + 1e-8))


def city_stats(city):
    bg.use_city(city)
    bg.SENSOR_SET = "urban"
    bg.EPA_CORRECT = True
    tr.WIND_SOURCE = "hrrr"
    tr.STRICT_INPUTS = True
    tr.USE_CACHE = False
    g = tr.build_static_graph()
    ids, pm, observed = g[0], g[1], g[2]
    elev, x_m, y_m = g[7], g[8], g[9]
    values = np.clip(pm.to_numpy(dtype=np.float64), 0, None)
    z = np.sqrt(values)                      # match eval_temporal default transform
    obs = observed.to_numpy()
    x = np.asarray(x_m, float); y = np.asarray(y_m, float)
    e = np.asarray(elev, float)

    area = convex_hull_area(x, y)
    N = len(ids)
    s = dict(
        N=N,
        density=N / area if area and not np.isnan(area) else np.nan,
        mean_nn_km=mean_nn_km(x, y),
        relief_m=float(np.std(e)),
        spatial_nrmse=spatial_loo_nrmse(z, obs, x, y),
    )
    for h in GAPS:
        rho, vario = temporal_stats(z, obs, h)
        s[f"rho_{h}"] = rho
        s[f"vario_{h}"] = vario
        s[f"richness_{h}"] = np.log10(N) * rho
    return s


def ols_fit(px, py):
    px = np.asarray(px, float); py = np.asarray(py, float)
    ok = np.isfinite(px) & np.isfinite(py)
    px, py = px[ok], py[ok]
    if len(px) < 3 or px.std() < 1e-12:
        return None
    A = np.vstack([px, np.ones_like(px)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, py, rcond=None)
    pred = slope * px + intercept
    ss_res = ((py - pred) ** 2).sum()
    ss_tot = ((py - py.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # Spearman via rank Pearson
    rx = np.argsort(np.argsort(px)); ry = np.argsort(np.argsort(py))
    rho_s = np.corrcoef(rx, ry)[0, 1]
    thr = -intercept / slope if abs(slope) > 1e-12 else np.nan  # margin = 0 crossing
    return dict(slope=slope, intercept=intercept, r2=r2, spearman=rho_s, threshold=thr)


def main():
    print("Computing a-priori (label-free) network statistics per city ...\n")
    stats = {c: city_stats(c) for c in CITIES}

    # Assemble the per-(city,gap) design matrix.
    rows = []
    for c in CITIES:
        for h in GAPS:
            s = stats[c]
            rows.append(dict(
                city=c, gap=h, improvement=MARGINS[(c, h)],
                N=s["N"], density=s["density"], mean_nn_km=s["mean_nn_km"],
                relief_m=s["relief_m"], spatial_nrmse=s["spatial_nrmse"],
                rho_h=s[f"rho_{h}"], vario_h=s[f"vario_{h}"],
                richness=s[f"richness_{h}"],
            ))

    # --- print raw stats ---
    cols = [("city", "city"), ("gap", "gap"), ("improvement", "impr%"),
            ("N", "N"), ("density", "density"), ("mean_nn_km", "mean_nn_km"),
            ("relief_m", "relief_m"), ("spatial_nrmse", "spat_nrmse"),
            ("rho_h", "rho_h"), ("vario_h", "vario_h"), ("richness", "richness")]
    print("  ".join(f"{lbl:>10}" for _, lbl in cols))
    for r in rows:
        print("  ".join(f"{r[k]:>10.3f}" if isinstance(r[k], float) else f"{str(r[k]):>10}"
                        for k, _ in cols))

    # --- fit each candidate predictor against the realized margin ---
    print("\nPredictor -> realized GNN margin (improvement%%), OLS across 6 cells:")
    print(f"{'predictor':>14}  {'R2':>6}  {'Spearman':>9}  {'slope':>9}  {'thresh(impr=0)':>15}")
    y = [r["improvement"] for r in rows]
    candidates = ["N", "density", "mean_nn_km", "relief_m", "spatial_nrmse",
                  "rho_h", "vario_h", "richness"]
    fits = {}
    for k in candidates:
        f = ols_fit([r[k] for r in rows], y)
        fits[k] = f
        if f:
            print(f"{k:>14}  {f['r2']:>6.3f}  {f['spearman']:>9.3f}  "
                  f"{f['slope']:>9.3f}  {f['threshold']:>15.3f}")
        else:
            print(f"{k:>14}  {'--':>6}")

    best = max((k for k in candidates if fits[k]), key=lambda k: fits[k]["r2"])
    bf = fits[best]
    print(f"\nBEST a-priori predictor: '{best}'  (R2={bf['r2']:.3f}, "
          f"Spearman={bf['spearman']:.3f})")
    print(f"DEPLOY-OR-NOT RULE: predicted improvement% = {bf['slope']:.3f} * {best} "
          f"{bf['intercept']:+.3f}")
    print(f"  crossover: deploy the GNN when {best} "
          f"{'>' if bf['slope'] > 0 else '<'} {bf['threshold']:.3f}  "
          f"(below/above -> ST-kriging is as good; save the model).")

    _save_figure(rows, best, bf)
    _densified_fit()


def _save_figure(rows, best, bf):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pathlib
    except Exception as ex:
        print(f"[fig skipped: {ex}]")
        return
    colors = {"pittsburgh": "#3F6FB0", "fresno": "#3F7E4E", "slc": "#B23B3B"}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = np.array([r[best] for r in rows])
    ys = np.array([r["improvement"] for r in rows])
    lo, hi = xs.min(), xs.max()
    grid = np.linspace(lo - 0.05*(hi-lo), hi + 0.05*(hi-lo), 100)
    ax.plot(grid, bf["slope"]*grid + bf["intercept"], "-", color="#888",
            lw=1.5, zorder=1, label=f"OLS  R²={bf['r2']:.2f}")
    ax.axhline(0, color="#c33", lw=1, ls="--", zorder=0)
    if np.isfinite(bf["threshold"]) and lo <= bf["threshold"] <= hi:
        ax.axvline(bf["threshold"], color="#c33", lw=1, ls="--", zorder=0)
    for r in rows:
        ax.scatter(r[best], r["improvement"], s=90, color=colors[r["city"]],
                   edgecolor="white", lw=1.2, zorder=3,
                   marker=("o" if r["gap"] == 24 else "s"))
        ax.annotate(f"{r['city'][:3]} {r['gap']}h", (r[best], r["improvement"]),
                    fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"a-priori predictor:  {best}")
    ax.set_ylabel("realized GNN margin over ST-kriging  (improvement %)")
    ax.set_title("Deploy-or-not diagnostic: predicting the GNN's marginal value")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = pathlib.Path("experiments/logs/diagnostic")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "diagnostic.png", dpi=200)
    print(f"\n[saved] {out/'diagnostic.png'}")


def _densified_fit():
    """If scripts/run_temporal_density.py has produced subsampled-network points,
    refit the deploy-or-not rule on the ~20-point cloud (spatial_nrmse vs realized
    margin) -- the honest, n>3 version of the diagnostic."""
    import csv
    import pathlib
    p = pathlib.Path("experiments/logs/diagnostic/density_points.csv")
    if not p.exists():
        print("\n[densified fit skipped: run scripts/run_temporal_density.py first]")
        return
    xs, ys, cts = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                xs.append(float(r["spatial_nrmse"])); ys.append(float(r["margin_pct"]))
                cts.append(r["city"])
            except (ValueError, KeyError):
                continue
    if len(xs) < 4:
        print(f"\n[densified fit: only {len(xs)} points so far]")
        return
    f = ols_fit(xs, ys)
    print("\n" + "=" * 68)
    print(f"DENSIFIED FIT  ({len(xs)} distinct networks, spatial_nrmse -> margin%)")
    print("=" * 68)
    print(f"  R2={f['r2']:.3f}  Spearman={f['spearman']:.3f}  "
          f"slope={f['slope']:.2f}  crossover nrmse={f['threshold']:.3f}")
    print(f"  RULE: deploy the GNN when spatial_nrmse > {f['threshold']:.3f}")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"pittsburgh": "#3F6FB0", "fresno": "#3F7E4E", "slc": "#B23B3B"}
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        xa = np.array(xs); grid = np.linspace(xa.min(), xa.max(), 100)
        ax.plot(grid, f["slope"]*grid + f["intercept"], "-", color="#888", lw=1.5,
                label=f"OLS  R²={f['r2']:.2f}  (n={len(xs)})")
        ax.axhline(0, color="#c33", lw=1, ls="--")
        if np.isfinite(f["threshold"]):
            ax.axvline(f["threshold"], color="#c33", lw=1, ls="--")
        for x, y, c in zip(xs, ys, cts):
            ax.scatter(x, y, s=55, color=colors.get(c, "#888"),
                       edgecolor="white", lw=0.8, zorder=3)
        ax.set_xlabel("a-priori predictor:  spatial_nrmse (interpolation headroom)")
        ax.set_ylabel("GNN margin over ST-kriging (improvement %)")
        ax.set_title(f"Deploy-or-not diagnostic, densified ({len(xs)} networks)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(p.parent / "diagnostic_densified.png", dpi=200)
        print(f"  [saved] {p.parent/'diagnostic_densified.png'}")
    except Exception as ex:
        print(f"  [fig skipped: {ex}]")


if __name__ == "__main__":
    main()
