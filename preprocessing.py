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

import pandas as pd


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
# a sensor must report at least this fraction of the year to be kept as a node.
# 0.40 drops 125725 (~0.28) but keeps the ~0.56 borderline sensors; raise it to
# also prune 147456 / 180637 if you want a stricter node set.
COVERAGE_THRESHOLD = 0.40

# how to fill missing *inputs* (targets are never filled). "zero" mirrors what
# model.py did before; "mean" uses each sensor's own mean, which is a gentler
# prior for message passing. Filled values are flagged by `observed`, so this
# choice only affects the forward pass, never the loss.
FILL_METHOD = "zero"


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
               verbose: bool = True):
    """Full pipeline -> (filled, observed, kept_ids, dropped_ids).

    `filled` and `observed` share the same [time x kept_node] shape; feed
    `filled` to the model as input and use `observed` to mask the loss.
    """
    kept_wide, dropped = drop_dud_sensors(pm_wide, threshold)
    observed = observed_mask(kept_wide)
    filled = fill_missing(kept_wide, fill_method)
    kept_ids = list(kept_wide.columns)

    if verbose:
        cov = sensor_coverage(pm_wide)
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
# demo -- runs the good (build_graph2) pipeline just to get a real wide table
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import build_graph2 as bg

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
