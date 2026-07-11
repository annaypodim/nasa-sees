"""
Preprocessing for the PM2.5 sensor graph  ->  clean node-feature tables.
=============================================================================
Two jobs, both grounded in the earlier data analysis:

  1. DROP FULL-YEAR DUD SENSORS
     A sensor whose file is missing most of the year (e.g. 125725 at ~28%
     coverage) can never be a reliable node. Missingness that is a property of
     the *whole sensor* (not just a few hours) is removed once, up front, as a
     node-level decision. This is safe because it is static: the node is gone
     from every timestep, so the fixed kNN topology is simply rebuilt on the
     survivors.

  2. PER-CELL OBSERVED MASK  (the imputation-safe way to handle NaN)
     Most missingness is per (sensor, hour) cell, and it shifts hour to hour
     (only ~2.4% of hours have all sensors reporting). We do NOT delete those
     cells: a value we never observed has no ground truth, so it can never be a
     training target -- but the sensor is still a valid node the model should
     learn to impute. So:
        - `observed`  : boolean [time x node], True only where we have a real
                        reading. Loss / metrics must be restricted to these.
        - `filled`    : the same table with missing inputs filled to a finite
                        number (so the forward pass is finite) -- NEVER used as
                        a target, only as an input feature.

Nothing that we cannot confirm ever enters supervision; nothing recoverable is
thrown away.

run the demo:   .venv/bin/python preprocessing.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
# a sensor must report at least this fraction of the year to be kept as a node.
# 0.40 drops 125725 (~0.28) but keeps the ~0.56 borderline sensors; raise it to
# also prune 147456 / 180637 if you want a stricter node set.
COVERAGE_THRESHOLD = 0.40

# physical-plausibility bounds for a single hourly PM2.5 reading (ug/m3).
# The EPA AQI scale tops out at ~500 ug/m3; sustained hourly values above that
# are off-scale and, in this dataset, come from stuck/faulty sensors (e.g.
# 124223 reads >500 for basically the whole year). Readings outside [0, MAX]
# are treated as NOT observed -- never an input, never a target. A sensor whose
# real coverage then falls below COVERAGE_THRESHOLD is dropped by the dud step,
# so a mostly-garbage sensor prunes itself without a separate rule.
PLAUSIBLE_MIN_UG = 0.0
PLAUSIBLE_MAX_UG = 500.0

# how to fill missing *inputs* (targets are never filled). "zero" mirrors what
# model.py did before; "mean" uses each sensor's own mean, which is a gentler
# prior for message passing. Filled values are flagged by `observed`, so this
# choice only affects the forward pass, never the loss.
FILL_METHOD = "zero"

# TEMPORAL DESPIKE (robust MAD outlier removal).  Raw PurpleAir series carry
# isolated single-hour noise spikes / dropouts that survive the coarse [0,500]
# plausibility clip but still corrupt supervision and inflate LINEAR MAE (a lone
# 300 ug/m3 blip is a big absolute error at eval and a bad training target).
# GraPhy's benchmark uses QA'd data; this is the QA we can apply to our own bytes
# without any new fetch.  For each sensor we compare every reading to a CENTRED
# per-sensor rolling median and flag cells whose deviation exceeds DESPIKE_K
# robust-sigma (MAD-based, robust_sigma = 1.4826*MAD).  A cell flagged as a spike
# is set NOT observed (never an input, never a target) -- identical treatment to
# an implausible reading.  Conservative by design: a REAL multi-hour pollution
# episode drags the rolling median up WITH it, so its readings stay within K
# robust-sigma and survive; only short isolated excursions relative to the local
# level are removed.  OFF by default so existing pipelines are unchanged; the
# eval driver flips DESPIKE on via --despike.
DESPIKE = False
DESPIKE_WINDOW = 13     # hours, centred rolling window (odd -> symmetric)
DESPIKE_K = 6.0         # robust-z threshold; higher = more conservative
DESPIKE_MIN_SIGMA = 1.0  # ug/m3 floor on robust_sigma so a flat clean stretch
#                          (MAD~0) doesn't flag ordinary small wiggles as spikes


# ---------------------------------------------------------------------------
# 0. mask physically-implausible readings
# ---------------------------------------------------------------------------
def mask_implausible(pm_wide: pd.DataFrame,
                     lo: float = PLAUSIBLE_MIN_UG,
                     hi: float = PLAUSIBLE_MAX_UG):
    """Set out-of-range PM2.5 cells to NaN so they count as *not observed*.

    Returns (cleaned_wide, n_masked). Runs before the coverage drop, so a
    sensor whose readings are mostly garbage sees its coverage collapse and is
    then removed by drop_dud_sensors -- no special per-sensor rule needed. Only
    finite readings strictly outside [lo, hi] are masked; existing NaNs stay
    NaN and are not counted.
    """
    bad = ((pm_wide < lo) | (pm_wide > hi)) & pm_wide.notna()
    n_masked = int(bad.to_numpy().sum())
    return pm_wide.mask(bad), n_masked


# ---------------------------------------------------------------------------
# 0b. temporal despike (per-sensor robust MAD outlier removal)
# ---------------------------------------------------------------------------
def despike(pm_wide: pd.DataFrame,
            window: int = DESPIKE_WINDOW,
            k: float = DESPIKE_K,
            min_sigma: float = DESPIKE_MIN_SIGMA):
    """Flag isolated single-hour spikes as NaN (not observed).

    Returns (cleaned_wide, n_masked). Per column: robust-z = |x - rolling_median|
    / max(1.4826*rolling_MAD, min_sigma). Cells with robust-z > k are masked. The
    rolling median tracks genuine multi-hour episodes (they move it up with them),
    so only short excursions relative to the LOCAL level are removed; min_sigma
    prevents a quiet clean stretch (MAD~0) from flagging normal small variation.
    Runs before the coverage drop, like mask_implausible, so a chronically noisy
    sensor loses coverage and self-prunes.
    """
    minp = max(3, window // 2)
    med = pm_wide.rolling(window, center=True, min_periods=minp).median()
    absdev = (pm_wide - med).abs()
    mad = absdev.rolling(window, center=True, min_periods=minp).median()
    robust_sigma = (1.4826 * mad).clip(lower=min_sigma)
    bad = (absdev > k * robust_sigma) & pm_wide.notna()
    n_masked = int(bad.to_numpy().sum())
    return pm_wide.mask(bad), n_masked


# ---------------------------------------------------------------------------
# 1. drop full-year dud sensors
# ---------------------------------------------------------------------------
def sensor_coverage(pm_wide: pd.DataFrame) -> pd.Series:
    """Fraction of timesteps each sensor (column) actually reported."""
    return pm_wide.notna().mean().sort_values()


def drop_dud_sensors(pm_wide: pd.DataFrame,
                     threshold: float = COVERAGE_THRESHOLD):
    """Remove sensors below `threshold` yearly coverage.

    Returns (kept_wide, dropped_ids). Node order (column order) is preserved
    for the survivors so downstream coords/edges line up.
    """
    cov = sensor_coverage(pm_wide)
    dropped = cov[cov < threshold].index.tolist()
    kept = [c for c in pm_wide.columns if c not in dropped]
    return pm_wide[kept], dropped


# ---------------------------------------------------------------------------
# 2. per-cell observed mask + input fill
# ---------------------------------------------------------------------------
def observed_mask(pm_wide: pd.DataFrame) -> pd.DataFrame:
    """Boolean [time x node]: True exactly where a real reading exists.

    This is the record that reconciles "can't train without full data": the
    loss is restricted to True cells, so genuinely-missing values -- which we
    have no way to confirm -- can never be supervised.
    """
    return pm_wide.notna()


def fill_missing(pm_wide: pd.DataFrame, method: str = FILL_METHOD) -> pd.DataFrame:
    """Fill missing *inputs* so the forward pass is finite. Never a target."""
    if method == "zero":
        return pm_wide.fillna(0.0)
    if method == "mean":
        # each sensor's own mean; any all-NaN column (shouldn't survive the
        # dud drop) falls back to 0 so nothing stays NaN.
        return pm_wide.fillna(pm_wide.mean()).fillna(0.0)
    raise ValueError(f"unknown fill method {method!r} (use 'zero' or 'mean')")


# ---------------------------------------------------------------------------
# convenience: run both steps and report
# ---------------------------------------------------------------------------
def preprocess(pm_wide: pd.DataFrame,
               threshold: float = COVERAGE_THRESHOLD,
               fill_method: str = FILL_METHOD,
               plausible_range: tuple[float, float] = (PLAUSIBLE_MIN_UG,
                                                       PLAUSIBLE_MAX_UG),
               verbose: bool = True):
    """Full pipeline -> (filled, observed, kept_ids, dropped_ids).

    `filled` and `observed` share the same [time x kept_node] shape; feed
    `filled` to the model as input and use `observed` to mask the loss.
    Implausible readings are masked to NaN first, so stuck/faulty sensors lose
    coverage and are then dropped by the dud step.
    """
    lo, hi = plausible_range
    clean_wide, n_masked = mask_implausible(pm_wide, lo, hi)
    n_spikes = 0
    if DESPIKE:
        clean_wide, n_spikes = despike(clean_wide)
        if verbose:
            cells = pm_wide.size
            print(f"[preprocess] despike: masked {n_spikes}/{cells} spike cells "
                  f"({n_spikes / cells:.2%}) > {DESPIKE_K:g} robust-sigma from a "
                  f"{DESPIKE_WINDOW}h rolling median -> treated as not observed")
    kept_wide, dropped = drop_dud_sensors(clean_wide, threshold)
    observed = observed_mask(kept_wide)
    filled = fill_missing(kept_wide, fill_method)
    kept_ids = list(kept_wide.columns)

    if verbose:
        cov = sensor_coverage(clean_wide)  # coverage AFTER masking garbage
        cells = pm_wide.size
        print(f"[preprocess] masked {n_masked}/{cells} implausible cells "
              f"({n_masked / cells:.2%}) outside [{lo:g}, {hi:g}] ug/m3 "
              f"-> treated as not observed")
        print(f"[preprocess] sensors: {pm_wide.shape[1]} -> {len(kept_ids)} "
              f"kept  (threshold={threshold:.0%} yearly coverage)")
        if dropped:
            print("[preprocess] dropped full-year duds "
                  f"(coverage): {[(d, round(float(cov[d]), 3)) for d in dropped]}")
        cells = observed.size
        obs = int(observed.to_numpy().sum())
        print(f"[preprocess] observed cells: {obs}/{cells} "
              f"({obs / cells:.1%}) -> only these can enter the loss; "
              f"the rest are filled inputs ({fill_method}), never targets")
    return filled, observed, kept_ids, dropped


# ---------------------------------------------------------------------------
# 3. persist the processed tables so runs stop recomputing them
# ---------------------------------------------------------------------------
# saved as CSV (not parquet: no pyarrow dependency, and CSV is human-readable +
# git-diffable, which is the point -- one traceable artifact per city-group).
def processed_dir(data_dir: Path, city: str, sensor_set: str) -> Path:
    return Path(data_dir) / city / "processed" / sensor_set


def save_processed(filled: pd.DataFrame, observed: pd.DataFrame,
                   data_dir: Path, city: str, sensor_set: str,
                   dropped_ids: list | None = None, extra: dict | None = None) -> Path:
    """Write filled/observed tables + a meta.json describing how they were made.

    Returns the directory written. `filled` and `observed` share index/columns;
    meta.json records the settings + kept/dropped ids so the artifact is
    self-explaining (which raw data + which thresholds produced it).
    """
    out = processed_dir(data_dir, city, sensor_set)
    out.mkdir(parents=True, exist_ok=True)
    filled.to_csv(out / "pm_filled.csv", index_label="timestamp")
    observed.astype(int).to_csv(out / "observed.csv", index_label="timestamp")
    meta = {
        "city": city,
        "sensor_set": sensor_set,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_timesteps": int(filled.shape[0]),
        "n_nodes": int(filled.shape[1]),
        "kept_ids": list(map(str, filled.columns)),
        "dropped_ids": list(map(str, dropped_ids or [])),
        "coverage_threshold": COVERAGE_THRESHOLD,
        "plausible_range_ug": [PLAUSIBLE_MIN_UG, PLAUSIBLE_MAX_UG],
        "fill_method": FILL_METHOD,
        "observed_cells": int(observed.to_numpy().sum()),
        **(extra or {}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load_processed(data_dir: Path, city: str, sensor_set: str):
    """Return (filled, observed, meta) if a processed cache exists, else None.

    observed comes back boolean; both tables use a parsed DatetimeIndex so they
    drop straight into the training pipeline in place of a fresh preprocess().
    """
    d = processed_dir(data_dir, city, sensor_set)
    fp, op, mp = d / "pm_filled.csv", d / "observed.csv", d / "meta.json"
    if not (fp.exists() and op.exists()):
        return None
    filled = pd.read_csv(fp, index_col="timestamp", parse_dates=["timestamp"])
    observed = pd.read_csv(op, index_col="timestamp", parse_dates=["timestamp"]).astype(bool)
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    return filled, observed, meta


# ---------------------------------------------------------------------------
# demo -- runs the good (build_graph2) pipeline just to get a real wide table
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.graph import build_graph2 as bg

    cfg = bg.GROUP_CONFIG[bg.SENSOR_SET]
    coords = bg.parse_sensor_coords(bg.COORDS_FILE)
    coords = coords[coords["group"] == bg.SENSOR_SET]
    long_pm = bg.load_air_quality(cfg["purple_air_dir"], coords["station_id"].tolist())
    station_ids = sorted(set(coords["station_id"]) & set(long_pm["station_id"]))
    pm_wide = (long_pm.pivot_table(index="timestamp", columns="station_id",
                                   values="pm25")
               .reindex(columns=station_ids).sort_index())

    print(f"\nraw wide table: {pm_wide.shape[0]} hours x {pm_wide.shape[1]} sensors")
    filled, observed, kept_ids, dropped = preprocess(pm_wide)
    print(f"\nkept node ids ({len(kept_ids)}): {kept_ids}")
    print(f"filled shape: {filled.shape}   observed shape: {observed.shape}")
