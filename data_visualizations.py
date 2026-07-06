"""
sanity-check visualizations for the PM2.5 sensor-graph GNN inputs
=============================================================================
Four figure groups, each targeting a class of "is anything wrong?" bug for
graph inputs. Everything is built from the SAME loaders the real pipeline uses
(build_graph2 + preprocessing), so what you see is what the model eats.

  1. MISSINGNESS / COVERAGE   outputs/viz/1_missingness.png
       - availability heatmap [sensor x hour]: who is online when
       - per-sensor coverage bars: the long tail of sparse / dead sensors
     Catches: the overlap problem, systematic gaps, the 8736 ceiling.

  2. VALUE DISTRIBUTIONS       outputs/viz/2_distributions.png
       - pooled PM2.5 histogram, linear + log
       - per-sensor boxplots
       - per-sensor mean-vs-std scatter
     Catches: negatives, stuck-at-zero, off-scale spikes, miscalibrated sensors.

  3. TIME-SERIES BEHAVIOUR     outputs/viz/3_timeseries.png
       - a few sensors overlaid: do they co-vary on regional events?
       - diurnal profile (mean by hour-of-day)
       - seasonal profile (mean by month)
     Catches: flat/dead series, timezone shifts, physically-wrong daily/annual shape.

  4. GRAPH STRUCTURE           outputs/viz/4_graph.png
       - spatial node map with kNN edges (real lat/lon)
       - node degree distribution
       - PM2.5 correlation vs. edge distance
     Catches: bad coordinates, isolated / hub nodes, a graph that ignores physics.

  6. MONTHLY MEANS PER SENSOR  outputs/viz/6_monthly_by_sensor.png
       - [sensor x month] heatmap of mean PM2.5
       - the same means overlaid as one line per sensor
     Catches: sensors that only wake up for part of the year, per-sensor
     seasonal offsets, and whether the wildfire-season peak is shared.

run:   .venv/bin/python data_visualizations.py
       .venv/bin/python data_visualizations.py --raw   # skip preprocessing drop
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_graph2 as bg
import preprocessing as pp

# --- name this run -----------------------------------------------------------
# Write whatever describes the current experiment here. Every figure for this
# run is saved under outputs/viz/<RUN_LABEL>/ so manual runs don't overwrite
# each other -- change it before each run to keep the graphs as you go.
RUN_LABEL = "above 1 sigma removed"

# drop sensors whose yearly-mean PM2.5 exceeds (cross-sensor mean + this many
# std). Set to None to keep every sensor. 1.0 -> "above 1 sigma removed".
DROP_ABOVE_SIGMA = 1.0

VIZ_DIR = bg.OUT_DIR / "viz" / RUN_LABEL if RUN_LABEL else bg.OUT_DIR / "viz"


# ---------------------------------------------------------------------------
# load once: the wide [time x sensor] PM2.5 table + coords, via the real loaders
# ---------------------------------------------------------------------------
def load(apply_preprocess: bool = True):
    cfg = bg.GROUP_CONFIG[bg.SENSOR_SET]
    coords = bg.parse_sensor_coords(bg.COORDS_FILE)
    coords = coords[coords["group"] == bg.SENSOR_SET]

    long_pm = bg.load_air_quality(cfg["purple_air_dir"], coords["station_id"].tolist())
    station_ids = sorted(set(coords["station_id"]) & set(long_pm["station_id"]))
    coords = coords[coords["station_id"].isin(station_ids)].sort_values("station_id")
    station_ids = coords["station_id"].tolist()
    pm_wide = (
        long_pm.pivot_table(index="timestamp", columns="station_id", values="pm25")
        .reindex(columns=station_ids)
        .sort_index()
    )

    if apply_preprocess:
        # mask implausible + drop full-year duds -> the node set the model sees.
        clean, _ = pp.mask_implausible(pm_wide)
        pm_wide, dropped = pp.drop_dud_sensors(clean)
        if dropped:
            print(f"[viz] preprocessing dropped {len(dropped)} dud sensors: {dropped}")
        station_ids = list(pm_wide.columns)
        coords = coords[coords["station_id"].isin(station_ids)]

        # drop mean-level outliers: sensors whose yearly avg sits above the
        # cross-sensor mean by more than DROP_ABOVE_SIGMA std -> unhealthy node.
        if DROP_ABOVE_SIGMA is not None:
            per_sensor = pm_wide.mean()
            cutoff = per_sensor.mean() + DROP_ABOVE_SIGMA * per_sensor.std()
            hot = per_sensor[per_sensor > cutoff].index.tolist()
            if hot:
                print(f"[viz] dropped {len(hot)} sensors above "
                      f"+{DROP_ABOVE_SIGMA:g} sigma (cutoff={cutoff:.1f}): {hot}")
            pm_wide = pm_wide.drop(columns=hot)
            station_ids = list(pm_wide.columns)
            coords = coords[coords["station_id"].isin(station_ids)]
    coords = coords.set_index("station_id").reindex(station_ids).reset_index()
    print(f"[viz] {pm_wide.shape[0]} hours x {pm_wide.shape[1]} sensors")
    return pm_wide, coords, station_ids


# ---------------------------------------------------------------------------
# 1. missingness / coverage
# ---------------------------------------------------------------------------
def fig_missingness(pm_wide, ids):
    # plausibility-mask so stuck-garbage doesn't falsely count as "reporting"
    # (harmless if pm_wide was already masked by preprocessing).
    masked, _ = pp.mask_implausible(pm_wide)
    obs = masked.notna()

    cov = obs.mean().sort_values()  # ascending coverage
    order = cov.index.tolist()
    present = obs[order].to_numpy().T  # [sensor x hour], bool

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [3, 1, 1.5]}
    )

    ax0.imshow(
        present,
        aspect="auto",
        cmap="Greens",
        interpolation="nearest",
        origin="lower",
        extent=[0, present.shape[1], 0, present.shape[0]],
    )
    ax0.set_yticks(np.arange(len(order)) + 0.5)
    ax0.set_yticklabels(order, fontsize=7)
    ax0.set(
        title="Data availability throughout 2023",
        xlabel="Hour index",
        ylabel="Sensor ID",
    )

    ax1.barh(np.arange(len(cov)), cov.values * 100, color="tab:green")
    ax1.set_yticks(np.arange(len(cov)))
    ax1.set_yticklabels(cov.index, fontsize=7)
    ax1.set(
        title="Sensor coverage by percentage",
        xlabel="Percent of total year",
        xlim=(0, 100),
    )
    ax1.legend(fontsize=8)

    # --- tradeoff: keep top-N most-reliable sensors, how many hours have ALL
    #     N reporting at once -> that's the usable full-graph pool. -----------
    desc = cov.index[::-1].tolist()  # most-reliable first
    H = len(obs)
    ns = list(range(2, len(desc) + 1))
    full = [int(obs[desc[:n]].all(axis=1).sum()) for n in ns]
    ax2.plot(ns, full, marker="o", color="tab:blue")
    knee = 16 if max(ns) >= 16 else max(ns)
    ki = ns.index(knee)
    ax2.axvline(knee, color="red", ls="--", lw=1)
    ax2.annotate(
        f"{knee} sensors\n{full[ki]} full graphs\n({full[ki] / H:.0%} of year)",
        (knee, full[ki]),
        fontsize=8,
        xytext=(8, -30),
        textcoords="offset points",
        color="red",
    )
    ax2.set(
        title="Size of full graph vs. hours",
        xlabel="Sensors",
        ylabel="Hours",
        ylim=(0, H * 1.02),
    )
    ax2.grid(alpha=0.3)
    _save(fig, "1_missingness.png")


# ---------------------------------------------------------------------------
# 2. value distributions
# ---------------------------------------------------------------------------
def fig_distributions(pm_wide, ids):
    vals = pm_wide.to_numpy().ravel()
    vals = vals[~np.isnan(vals)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(vals, bins=80, color="tab:blue")
    axes[0, 0].set(title="Pooled PM2.5 (linear)", xlabel="ug/m3", ylabel="count")

    pos = vals[vals > 0]
    axes[0, 1].hist(
        pos,
        bins=np.logspace(np.log10(pos.min()), np.log10(pos.max()), 80),
        color="tab:purple",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(title="Pooled PM2.5 (log x)", xlabel="ug/m3", ylabel="count")

    per_sensor = [pm_wide[c].dropna().to_numpy() for c in pm_wide.columns]
    axes[1, 0].boxplot(
        per_sensor,
        vert=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    axes[1, 0].set_xticklabels(pm_wide.columns, rotation=90, fontsize=6)
    axes[1, 0].set(title="Per-sensor distribution", ylabel="PM2.5 ug/m3")

    means = pm_wide.mean()
    stds = pm_wide.std()
    axes[1, 1].scatter(means, stds, color="tab:red")
    for sid in pm_wide.columns:
        axes[1, 1].annotate(
            sid,
            (means[sid], stds[sid]),
            fontsize=6,
            xytext=(2, 2),
            textcoords="offset points",
        )
    axes[1, 1].set(title="Sensor mean vs. std", xlabel="mean PM2.5", ylabel="std PM2.5")
    _save(fig, "2_distributions.png")


# ---------------------------------------------------------------------------
# 3. time-series behaviour
# ---------------------------------------------------------------------------
def fig_timeseries(pm_wide, ids):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # overlay the 5 best-covered sensors over ~2 weeks so events are visible.
    best = pm_wide.notna().mean().sort_values(ascending=False).index[:5]
    window = pm_wide[best].iloc[: 24 * 14]
    for sid in best:
        axes[0].plot(window.index, window[sid], lw=0.9, label=sid)
    axes[0].set(title="PM2.5 sensor covariance", ylabel="PM2.5 ug/m3")
    axes[0].legend(fontsize=8, ncol=5)

    idx = pd.DatetimeIndex(pm_wide.index)
    diurnal = pm_wide.groupby(idx.hour).mean().mean(axis=1)
    axes[1].plot(diurnal.index, diurnal.values, marker="o", color="tab:orange")
    axes[1].set(
        title="Diurnal profile (mean over all sensors by hour-of-day)",
        xlabel="hour of day (UTC)",
        ylabel="mean PM2.5",
        xticks=range(0, 24, 2),
    )

    monthly = pm_wide.groupby(idx.month).mean().mean(axis=1)
    axes[2].bar(monthly.index, monthly.values, color="tab:green")
    axes[2].set(
        title="Seasonal profile (mean over all sensors by month)",
        xlabel="month",
        ylabel="mean PM2.5",
        xticks=range(1, 13),
    )
    _save(fig, "3_timeseries.png")


# ---------------------------------------------------------------------------
# 4. graph structure
# ---------------------------------------------------------------------------
def fig_graph(pm_wide, coords, ids):
    lat = coords["lat"].to_numpy()
    lon = coords["lon"].to_numpy()
    x_m, y_m = bg.project(lat, lon)
    edge_index = bg.knn_edges(x_m, y_m, bg.K)
    dist = bg.distance_matrix(x_m, y_m)
    corr = bg.correlation_matrix(pm_wide)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # spatial map with edges
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        axes[0].plot([lon[i], lon[j]], [lat[i], lat[j]], color="0.7", lw=0.7, zorder=1)
    axes[0].scatter(lon, lat, s=70, c="tab:blue", edgecolors="k", zorder=3)
    for idx, sid in enumerate(ids):
        axes[0].annotate(
            sid,
            (lon[idx], lat[idx]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axes[0].set(
        title=f"kNN sensor graph (k={bg.K})", xlabel="longitude", ylabel="latitude"
    )
    axes[0].set_aspect("equal", adjustable="datalim")

    # degree distribution
    deg = np.bincount(edge_index[0].numpy(), minlength=len(ids))
    axes[1].bar(np.arange(len(ids)), deg, color="tab:purple")
    axes[1].set_xticks(np.arange(len(ids)))
    axes[1].set_xticklabels(ids, rotation=90, fontsize=6)
    axes[1].set(title="Node degree", ylabel="degree")

    # correlation heatmap: how similarly does every pair of sensors move?
    # ordered by longitude so geographically-near sensors sit next to each
    # other -> a bright diagonal block would mean "near = correlated".
    o = np.argsort(lon)
    corr_o = corr[np.ix_(o, o)]
    im = axes[2].imshow(corr_o, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_xticks(np.arange(len(ids)))
    axes[2].set_yticks(np.arange(len(ids)))
    axes[2].set_xticklabels([ids[i] for i in o], rotation=90, fontsize=6)
    axes[2].set_yticklabels([ids[i] for i in o], fontsize=6)
    axes[2].set(title="PM2.5 correlation heatmap (sensors ordered W->E)")
    fig.colorbar(im, ax=axes[2], fraction=0.046, label="Pearson corr")
    _save(fig, "4_graph.png")


# ---------------------------------------------------------------------------
# 5. spatial correlation: is physical proximity actually predictive of PM2.5
#    similarity? overlay correlation onto real positions two ways.
# ---------------------------------------------------------------------------
def fig_spatial_corr(pm_wide, coords, ids):
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    lat = coords["lat"].to_numpy()
    lon = coords["lon"].to_numpy()
    x_m, y_m = bg.project(lat, lon)
    edge_index = bg.knn_edges(x_m, y_m, bg.K)
    dist = bg.distance_matrix(x_m, y_m)
    corr = bg.correlation_matrix(pm_wide)
    norm = Normalize(-1, 1)
    cmap = plt.get_cmap("RdBu_r")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(17, 7))

    # --- panel 0: kNN edges colored by endpoint PM2.5 correlation -----------
    segs, cols = [], []
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if i < j:  # each undirected edge once
            segs.append([(lon[i], lat[i]), (lon[j], lat[j])])
            cols.append(corr[i, j])
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=3)
    lc.set_array(np.array(cols))
    ax0.add_collection(lc)
    ax0.scatter(lon, lat, s=60, c="0.2", zorder=3)
    for idx, sid in enumerate(ids):
        ax0.annotate(
            sid,
            (lon[idx], lat[idx]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    fig.colorbar(lc, ax=ax0, label="PM2.5 corr of the two endpoints")
    ax0.set(
        title="Are physical neighbors actually correlated?\n"
        "(kNN edges colored by correlation -- red = yes)",
        xlabel="longitude",
        ylabel="latitude",
    )
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.autoscale_view()

    # --- panel 1: correlation field from one reference sensor ----------------
    # reference = the most geographically central node (min mean distance).
    ref = int(np.argmin(dist.mean(axis=1)))
    c_ref = corr[ref].copy()
    sc = ax1.scatter(
        lon, lat, c=c_ref, cmap=cmap, norm=norm, s=140, edgecolors="k", zorder=3
    )
    ax1.scatter(
        lon[ref],
        lat[ref],
        marker="*",
        s=500,
        c="yellow",
        edgecolors="k",
        zorder=4,
        label=f"reference {ids[ref]}",
    )
    for idx, sid in enumerate(ids):
        ax1.annotate(
            f"{sid}\n{dist[ref, idx] / 1000:.1f}km",
            (lon[idx], lat[idx]),
            fontsize=5.5,
            xytext=(3, 3),
            textcoords="offset points",
        )
    fig.colorbar(sc, ax=ax1, label=f"PM2.5 corr with {ids[ref]}")
    ax1.set(
        title="Does correlation fade with distance?\n"
        "(every sensor colored by corr to the central one)",
        xlabel="longitude",
        ylabel="latitude",
    )
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(loc="best", fontsize=8)
    _save(fig, "5_spatial_corr.png")


# ---------------------------------------------------------------------------
# 6. monthly mean PM2.5 per sensor: each sensor's average for every month of
#    the year, both as a heatmap and as one line per sensor.
# ---------------------------------------------------------------------------
def fig_monthly_by_sensor(pm_wide, ids):
    idx = pd.DatetimeIndex(pm_wide.index)
    # [month x sensor] mean, then transpose to [sensor x month] for the heatmap.
    monthly = pm_wide.groupby(idx.month).mean().reindex(range(1, 13))
    grid = monthly.to_numpy().T  # [sensor x month], NaN where a sensor was silent
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    im = ax0.imshow(grid, aspect="auto", cmap="YlOrRd", origin="lower")
    ax0.set_xticks(np.arange(12))
    ax0.set_xticklabels(months)
    ax0.set_yticks(np.arange(len(ids)))
    ax0.set_yticklabels(ids, fontsize=7)
    ax0.set(
        title="Mean PM2.5 per sensor by month of 2023",
        xlabel="month",
        ylabel="Sensor ID",
    )
    fig.colorbar(im, ax=ax0, fraction=0.046, label="mean PM2.5 ug/m3")

    for col, sid in enumerate(ids):
        ax1.plot(range(1, 13), grid[col], marker="o", lw=1, label=sid)
    ax1.set(
        title="Monthly mean PM2.5, one line per sensor",
        xlabel="month",
        ylabel="mean PM2.5 ug/m3",
        xticks=range(1, 13),
    )
    ax1.set_xticklabels(months)
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)
    _save(fig, "6_monthly_by_sensor.png")


# ---------------------------------------------------------------------------
def _save(fig, name):
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = VIZ_DIR / name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        action="store_true",
        help="skip preprocessing (mask/drop) and show every sensor",
    )
    args = ap.parse_args()

    pm_wide, coords, ids = load(apply_preprocess=not args.raw)
    fig_missingness(pm_wide, ids)
    fig_distributions(pm_wide, ids)
    fig_timeseries(pm_wide, ids)
    fig_graph(pm_wide, coords, ids)
    fig_spatial_corr(pm_wide, coords, ids)
    fig_monthly_by_sensor(pm_wide, ids)
    print(f"\n[done] 6 figures in {VIZ_DIR}")


if __name__ == "__main__":
    main()
