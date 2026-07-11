"""
Train GraPhyNet as a PM2.5 imputer  ->  watch it actually learn.
=============================================================================
model.py only does ONE random forward pass, so its predictions are noise. This
adds the missing piece: a self-supervised imputation training loop.

THE OBJECTIVE (why it's honest):
    each step we take a timestep's KNOWN sensor values, randomly HIDE a subset
    of them from the input (set to the "unknown" placeholder), run the model,
    and score it ONLY on those hidden-but-known nodes. The model must therefore
    reconstruct values it cannot see in its own input -- so it can't cheat by
    copying the input through (the identity-trap). Genuinely-missing cells
    (observed == False) are never used as targets; we can't confirm them.

NORMALISATION:
    PM2.5 here spans 0 .. ~1568, so raw MSE would be ruled by one outlier. We
    train in log1p + z-score space (standard for PM2.5) and invert back to
    ug/m3 only when writing the human-readable predictions.csv.

run:   .venv/bin/python train.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save a PNG instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.graph import build_graph2 as bg
from src.graph import preprocessing as pp
from src.model.diffusion import inverse_distance_weights
from src.model.model import GraPhyNet

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
# --- levers: edit these to change a run ---------------------------------------
SEED = 0
EPOCHS = 80  # training length
VAL_FRAC = 0.15  # train/test split: last fraction of timeline held out for eval
MASK_FRAC = 0.20  # fraction of a timestep's KNOWN nodes to hide + predict
SPATIAL_HOLDOUT = False  # eval regime. False: leave-one-node-out (dense; neighbours
#   dominate, covariates redundant). True: hide a target AND its HOLDOUT_RING nearest
#   sensors, then predict the target -- it has no nearby known sensor, so the model
#   must lean on farther nodes + covariates (temperature). This is the "no monitor
#   here" interpolation regime where met covariates can actually add signal.
#   (CLI: --spatial-holdout, --holdout-ring N)
HOLDOUT_RING = 4  # how many nearest sensors to blank around the target, spatial mode
STEPS_PER_EPOCH = 64  # random timesteps sampled per epoch
LR = 0.01
UNKNOWN = 0.0  # placeholder fed for hidden/missing nodes (in z-space)
EXCEED_UG = 35.4  # PM2.5 exceedance threshold for the ROC-AUC label (EPA USG)
USE_ELEVATION = True  # elevation gate on/off (CLI: --no-elevation zeroes Δelev -> gate=1)
USE_TEMPERATURE = True  # temperature gate on/off (CLI: --no-temperature -> node_temp=None,
#                         an exact no-op). Silently inert if the city has no temperature.
USE_TEMP_FEATURE = True  # temperature as a NEVER-masked node input channel (CLI:
#                          --no-temp-feature). The target's own temperature stays
#                          visible when its PM2.5 is hidden, so the model can predict
#                          from it directly -- the lever the gate structurally lacks.
USE_ELEV_FEATURE = True  # DEM elevation as a NEVER-masked node channel (CLI:
#                          --no-elev-feature). Elevation is spatially varying AND
#                          knowable at any location from a DEM, so it's the covariate
#                          that actually pays off for interpolation in complex terrain.
WIND_SOURCE = "era5"  # which wind feeds the convection module (CLI: --wind).
#   "era5": the original ~30 km reanalysis -> one vector city-wide (spatially flat).
#   "hrrr": HRRR 3 km -> real spatial structure; its convergence zones carry PM
#           signal (spatial-anomaly corr -0.119) the single ERA5 vector can't show.
#   "zero": all-zero wind -> convection inert (distance only); the convection-OFF
#           floor for the ablation. Needs --allow-missing-inputs (all-zero wind).
USE_CACHE = True  # load data/<city>/processed/<group> if present (CLI: --rebuild)
STRICT_INPUTS = True  # abort if PM2.5 / distance / wind inputs are missing or all-zero
#                       (CLI: --allow-missing-inputs runs anyway with zero-filled features)
SYNTH_ELEV_K = 0.0  # SANITY TEST: inject a known elevation-dependent PM2.5 offset
#   (ug/m3 per metre of elevation above the node mean) so the field genuinely
#   decouples by height. If the gate works, ON should beat OFF here; if ON==OFF
#   even on this planted gradient, the module -- not the data -- is the problem.
#   CLI: --synth-elev-grad K  (0 = off, the real data). See notes below.
SYNTH_TEMP_K = 0.0  # SANITY TEST for the TEMPERATURE gate. Unlike the elevation
#   synth (a static per-node offset a per-EDGE gate recovers from same-height
#   neighbours), our temperature gate is per-NODE -- it can only turn a node's
#   mixing up/down, not route from similar-temp neighbours. So we plant the
#   signal a node gate CAN exploit: temperature controls how much a node's own
#   spatial anomaly survives. gain = clip(1 + K*s, 0) scales each cell's anomaly
#   about its timestep's spatial mean -- COLD (s<0) shrinks it toward the mean
#   (trapped/flat), HOT (s>0) amplifies it. A masked COLD node then sits flat
#   among possibly-HOT, anomalous neighbours: averaging them misleads, and only
#   a gate that suppresses the cold node's transport recovers the flat truth.
#   If tempON still can't beat tempOFF here, the gate architecture is the problem.
#   CLI: --synth-temp-grad K  (0 = off, the real data).
SYNTH_TEMPDIRECT_K = 0.0  # SANITY TEST for the temperature FEATURE. Plant PM2.5
#   as a DIRECT function of each node's OWN temperature: value = gmean + K*20*s.
#   A masked node's temperature stays visible, so the feature can read it and
#   predict the value EXACTLY, while a neighbour-only baseline can't (a node's
#   own temp isn't fully given by its neighbours). If temp_feature ON beats the
#   bare baseline here, the feature mechanism works -- the earlier real-data null
#   is the DATA (common-mode temperature), not the wiring.
#   CLI: --synth-temp-direct K  (0 = off).


# ---------------------------------------------------------------------------
# graph setup  (same static topology model.py builds, via preprocessing)
# ---------------------------------------------------------------------------
def build_static_graph():
    """Rebuild the graph from build_graph2's pipeline + return node/edge tables.

    Unlike the old placeholder (one city-wide wind for every edge, every hour),
    this carries build_graph2's REAL per-sensor wind, interpolated onto the
    hourly grid. Everything except the wind is static (topology, distances); the
    convection edge features are therefore time-varying -> `edge_attr_t` is
    [T, E, 3] = [distance, wind_along, wind_speed] rather than one [E, 3].
    """
    cfg = bg.GROUP_CONFIG[bg.SENSOR_SET]

    coords = bg.parse_sensor_coords(bg.COORDS_FILE)
    coords = coords[coords["group"] == bg.SENSOR_SET].sort_values("station_id")

    # Prefer the saved processed cache (scripts/build_processed.py) so we don't
    # re-parse every raw CSV each run. USE_CACHE=False (CLI --rebuild) forces a
    # fresh preprocess from the raw data.
    cached = pp.load_processed(bg.DATA_DIR, bg.CITY, bg.SENSOR_SET) if USE_CACHE else None
    if cached is not None:
        pm, observed, _meta = cached
        kept_ids = list(map(str, pm.columns))
        print(f"[cache] loaded processed {bg.CITY}-{bg.SENSOR_SET}: "
              f"{pm.shape[0]} hours x {pm.shape[1]} nodes "
              f"(built {_meta.get('generated_at', '?')})")
    else:
        long_pm = bg.load_air_quality(cfg["purple_air_dir"], coords["station_id"].tolist())
        # keep only ids present in BOTH coordinates and air-quality data
        station_ids = sorted(set(coords["station_id"]) & set(long_pm["station_id"]))
        pm_raw = (
            long_pm.pivot_table(index="timestamp", columns="station_id", values="pm25")
            .reindex(columns=station_ids)
            .sort_index()
        )
        # drop full-year duds + get the per-cell observed mask
        pm, observed, kept_ids, _ = pp.preprocess(pm_raw)

    # wind is interpolated onto the (now-known) PM2.5 timeline per surviving sensor.
    # WIND_SOURCE picks the field: ERA5 (~30 km, flat), HRRR (3 km, spatially
    # structured), or zero (convection-off ablation floor).
    if WIND_SOURCE == "zero":
        u10_wide = pd.DataFrame(0.0, index=pm.index, columns=kept_ids)
        v10_wide = pd.DataFrame(0.0, index=pm.index, columns=kept_ids)
        has_wind = False
        print("[wind] source=zero -> convection inert (distance only)")
    elif WIND_SOURCE == "hrrr":
        hrrr_dir = cfg.get("wind_hrrr_dir")
        u10_wide, v10_wide, has_wind = bg.load_wind(None, hrrr_dir, kept_ids, pm.index)
        print(f"[wind] source=HRRR 3km ({hrrr_dir})")
    else:
        u10_wide, v10_wide, has_wind = bg.load_wind(
            cfg["wind_zip"], cfg["wind_dir"], kept_ids, pm.index
        )
        print("[wind] source=ERA5 ~30km")
    # surface temperature on the same timeline for the temperature gate (live-
    # loaded like wind, not cached). cfg may omit temp_* -> have_temp=False.
    temp_wide, has_temp = bg.load_temperature(
        cfg.get("temp_zip"), cfg.get("temp_dir"), kept_ids, pm.index
    )

    # restrict everything to the surviving nodes; node order = sorted kept ids
    coords = coords[coords["station_id"].isin(kept_ids)].sort_values("station_id")
    ids = coords["station_id"].tolist()
    pm = pm.reindex(columns=ids)
    observed = observed.reindex(columns=ids)
    u10_wide = u10_wide.reindex(columns=ids)
    v10_wide = v10_wide.reindex(columns=ids)
    temp_wide = temp_wide.reindex(columns=ids)

    x_m, y_m = bg.project(coords["lat"].to_numpy(), coords["lon"].to_numpy())
    edge_index = bg.knn_edges(x_m, y_m, bg.K)
    dist = bg.distance_matrix(x_m, y_m)
    edge_dist = np.array([dist[i, j] for i, j in edge_index.t()])
    edge_weight = inverse_distance_weights(torch.tensor(edge_dist, dtype=torch.float))

    # per-timestep convection features from the REAL interpolated wind field:
    # project each edge's mean-endpoint wind vector onto the edge direction, so
    # wind_along = +speed with the wind (src->dst), -speed against, 0 crosswind.
    src, dst = edge_index.numpy()
    dx, dy = x_m[dst] - x_m[src], y_m[dst] - y_m[src]
    inv_len = 1.0 / np.maximum(np.hypot(dx, dy), 1e-9)
    ux, uy = dx * inv_len, dy * inv_len  # unit edge direction (src -> dst)

    # signed Δelevation per edge (dst − src, metres) for the elevation gate.
    # real DEM here (build_graph2 altitudes), so the gate is genuinely active.
    elev = coords["elevation"].to_numpy()
    edge_delev = torch.tensor(elev[dst] - elev[src], dtype=torch.float)

    U, V = u10_wide.to_numpy(), v10_wide.to_numpy()  # [T, N]
    u_edge = 0.5 * (U[:, src] + U[:, dst])           # [T, E] mean wind on the edge
    v_edge = 0.5 * (V[:, src] + V[:, dst])
    speed = np.hypot(u_edge, v_edge)                              # [T, E]
    wind_along = u_edge * ux[None, :] + v_edge * uy[None, :]      # [T, E]
    dist_col = np.broadcast_to(edge_dist, speed.shape)            # [T, E]
    edge_attr_t = torch.tensor(
        np.stack([dist_col, wind_along, speed], axis=-1), dtype=torch.float
    )  # [T, E, 3]

    print(
        f"[graph] set={bg.SENSOR_SET!r}  nodes={len(ids)}  "
        f"edges={edge_index.shape[1]}  timesteps={len(pm)}  "
        f"has_wind={has_wind}  has_temp={has_temp}"
    )

    validate_inputs(pm, observed, edge_dist, edge_attr_t, has_wind)
    return (ids, pm, observed, edge_index, edge_weight, edge_attr_t, edge_delev,
            elev, x_m, y_m, has_wind, temp_wide, has_temp)


# ---------------------------------------------------------------------------
# input guard: fail loudly instead of silently training on zero-filled inputs
# ---------------------------------------------------------------------------
def validate_inputs(pm, observed, edge_dist, edge_attr_t, has_wind):
    """Raise if a required model input (PM2.5, distance, wind) is missing or
    degenerate. Wind especially fails silently: build_graph2 zero-fills absent
    wind, so a whole physics channel can vanish without any error. Bypass with
    STRICT_INPUTS=False (CLI --allow-missing-inputs) for intentional ablations.
    """
    if not STRICT_INPUTS:
        if not has_wind:
            print("[warn] STRICT_INPUTS off: wind is missing -> zero-filled "
                  "(convection runs on distance only)")
        return

    problems = []
    # --- PM2.5 -------------------------------------------------------------
    if pm.shape[1] == 0 or int(observed.to_numpy().sum()) == 0:
        problems.append("PM2.5: no observed readings after preprocessing")
    fully_missing = [c for c in pm.columns if int(observed[c].sum()) == 0]
    if fully_missing:
        problems.append(f"PM2.5: {len(fully_missing)} node(s) with zero observed "
                        f"cells: {fully_missing}")
    # --- distance ----------------------------------------------------------
    if not np.isfinite(edge_dist).all():
        problems.append("distance: non-finite edge distance(s)")
    elif not (edge_dist > 0).all():
        problems.append("distance: zero-length edge(s) (duplicate sensor coords?)")
    # --- wind (the silent one) ---------------------------------------------
    if not has_wind:
        problems.append("wind: no wind files found -> edge wind features are all "
                        "zero (convection blind to wind)")
    elif float(edge_attr_t[..., 1:3].abs().sum()) == 0.0:
        problems.append("wind: wind_along/wind_speed are identically zero")

    if problems:
        raise ValueError(
            "STRICT_INPUTS: required model inputs are missing or degenerate:\n  - "
            + "\n  - ".join(problems)
            + "\n\nFix the data, or pass --allow-missing-inputs (STRICT_INPUTS=False) "
              "to train anyway with zero-filled features."
        )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def main():
    print(f"[config] epochs={EPOCHS}  val_frac={VAL_FRAC}  mask_frac={MASK_FRAC}")

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    (ids, pm, observed, edge_index, edge_weight, edge_attr_t, edge_delev, elev,
     x_m, y_m, has_wind, temp_wide, has_temp) = build_static_graph()
    N = len(ids)

    # elevation-gate toggle: zeroing Δelev makes gate=exp(0)=1 everywhere (exact
    # no-op), so --no-elevation is a clean ablation against the identical model.
    delev = edge_delev if USE_ELEVATION else torch.zeros_like(edge_delev)
    print(f"[config] elevation_gate={'ON' if USE_ELEVATION else 'OFF'}")

    # temperature-gate toggle: active only when requested AND the city has temp.
    # OFF -> node_temp=None each step -> the gate is skipped entirely (no-op).
    temp_on = USE_TEMPERATURE and has_temp
    if USE_TEMPERATURE and not has_temp:
        print("[config] temperature_gate=OFF (no temperature data for this city)")
    else:
        print(f"[config] temperature_gate={'ON' if temp_on else 'OFF'}")

    # normalise in log1p space using ONLY observed training cells (no leakage).
    T = len(pm)
    n_val = int(T * VAL_FRAC)
    train_ts = np.arange(0, T - n_val)
    val_ts = np.arange(T - n_val, T)

    values = pm.to_numpy(dtype=np.float64)  # [T, N], already NaN-filled
    obs = observed.to_numpy()  # [T, N] bool

    # --- SANITY TEST: plant a known elevation gradient -----------------------
    # Add a per-node offset K*(elev - mean_elev) to the OBSERVED cells only, so
    # the true field decouples by height: same-elevation neighbours agree, far-
    # apart-in-height neighbours are biased. This is exactly the structure the
    # elevation gate exists to exploit -- a well-mixed regime (real rural data)
    # has none of it, which is why ON==OFF there. If ON still can't beat OFF on
    # THIS field, the gate itself (architecture), not the data, is the problem.
    if SYNTH_ELEV_K != 0.0:
        offset = SYNTH_ELEV_K * (elev - elev.mean())  # [N] ug/m3
        values = values + np.where(obs, offset[None, :], 0.0)
        values = np.clip(values, 0, None)
        print(
            f"[synth] planted elevation gradient K={SYNTH_ELEV_K} ug/m3 per m -> "
            f"node offsets span [{offset.min():+.2f}, {offset.max():+.2f}] ug/m3 "
            f"(elev spread {elev.max() - elev.min():.0f} m)"
        )

    # --- SANITY TEST: plant a temperature-controlled mixing signal -----------
    # The field a per-NODE transport gate can actually exploit needs BOTH regimes:
    #   HOT node  -> value = global mean + a SPATIALLY SMOOTH pattern. smooth so a
    #               hot node's (also-hot, since temperature clusters in space)
    #               neighbours carry ~the same value -> recoverable by averaging.
    #   COLD node -> value = global mean (flat), i.e. the UNKNOWN=0 fallback a
    #               suppressed masked node lands on.
    # So neighbours are informative in the hot regime and misleading in the cold
    # regime, and ONLY the target's own temperature says which -- the gate must
    # pass transport when hot, suppress when cold. (v1 collapsed to the spatial
    # mean -> neighbours solved it, gate redundant. v2 amplified the rough real
    # anomalies -> hot regime unrecoverable, so predict-the-mean won. This fixes
    # both: a smooth, recoverable hot signal against a global-mean cold prior.)
    # gate=exp(a*s): a>0 gives hot>1 (pass) / cold<1 (suppress), the sign we want.
    if SYNTH_TEMP_K != 0.0 and has_temp:
        temp_c = temp_wide.to_numpy(dtype=np.float64)
        s_syn = np.nan_to_num(
            (temp_c - np.nanmean(temp_c[train_ts])) / (np.nanstd(temp_c[train_ts]) + 1e-6),
            nan=0.0,
        )                                                  # [T, N] standardized
        gmean = float(np.nanmean(np.where(obs, values, np.nan)[train_ts]))  # scalar prior
        # smooth per-node spatial pattern from UTM coords: one low-frequency bump
        # across the sensor field, so neighbours share values (recoverable).
        xn = (x_m - x_m.mean()) / (x_m.std() + 1e-6)
        yn = (y_m - y_m.mean()) / (y_m.std() + 1e-6)
        pattern = np.sin(1.3 * xn) + np.cos(1.1 * yn)      # [N] smooth in space
        pattern = SYNTH_TEMP_K * 20.0 * pattern             # -> ug/m3 amplitude
        hot = np.clip(s_syn, 0.0, None)                     # 0 when cold, >0 when hot
        synthetic = gmean + hot * pattern[None, :]          # cold->gmean, hot->smooth bump
        values = np.where(obs, np.clip(synthetic, 0, None), values)
        print(
            f"[synth] planted temp mixing signal K={SYNTH_TEMP_K}: cold->global mean "
            f"{gmean:.1f} ug/m3, hot-> +smooth spatial pattern "
            f"(|amp|<={np.abs(pattern).max():.1f} ug/m3, recoverable from hot neighbours)"
        )

    # --- SANITY TEST: plant PM2.5 as a DIRECT function of each node's own temp -
    # value = gmean + K*20*s[t,n]. The target's temperature is never masked, so
    # the temp FEATURE can read it and predict the value exactly; a neighbour-only
    # model cannot (a node's own temp isn't fully recoverable from neighbours).
    # This isolates the feature's unique power -- reading the hidden node itself.
    if SYNTH_TEMPDIRECT_K != 0.0 and has_temp:
        temp_c = temp_wide.to_numpy(dtype=np.float64)
        s_syn = np.nan_to_num(
            (temp_c - np.nanmean(temp_c[train_ts])) / (np.nanstd(temp_c[train_ts]) + 1e-6),
            nan=0.0,
        )                                                  # [T, N] standardized
        gmean = float(np.nanmean(np.where(obs, values, np.nan)[train_ts]))
        synthetic = gmean + SYNTH_TEMPDIRECT_K * 20.0 * s_syn
        values = np.where(obs, np.clip(synthetic, 0, None), values)
        print(
            f"[synth] planted DIRECT temp->PM2.5 K={SYNTH_TEMPDIRECT_K}: "
            f"value = {gmean:.1f} + {SYNTH_TEMPDIRECT_K*20:.0f}*s ug/m3 "
            f"(a node's own temperature IS its answer)"
        )

    logv = np.log1p(np.clip(values, 0, None))
    train_obs_vals = logv[np.ix_(train_ts, np.arange(N))][obs[train_ts]]
    mu, sigma = train_obs_vals.mean(), train_obs_vals.std() + 1e-8
    z = (logv - mu) / sigma  # standardised targets/inputs

    z_t = torch.tensor(z, dtype=torch.float)
    obs_t = torch.tensor(obs)

    # standardize surface temperature with TRAIN-window stats only (same
    # no-leakage rule as the PM z-scoring), so s=0 is the neutral airmass. NaNs
    # (missing sensor/gap) -> 0. This ONE array feeds two consumers:
    #   * the temperature GATE (node_temp): scales transport strength.
    #   * the temperature FEATURE: an extra, NEVER-masked node input channel so
    #     the model can read a node's temperature even when its PM2.5 is hidden.
    temp_feat_on = USE_TEMP_FEATURE and has_temp
    if has_temp:
        temp = temp_wide.to_numpy(dtype=np.float64)          # [T, N] °C
        tmu = np.nanmean(temp[train_ts])
        tsd = np.nanstd(temp[train_ts]) + 1e-6
        temp_std = torch.tensor(np.nan_to_num((temp - tmu) / tsd, nan=0.0),
                                dtype=torch.float)           # [T, N]
        print(f"[temp] standardized surface temperature: train mean={tmu:.1f}C "
              f"std={tsd:.1f}C  ->  gate_input=feature")
    else:
        temp_std = None
    temp_z_t = temp_std if temp_on else None                 # gate input (None -> inert)
    if USE_TEMP_FEATURE and not has_temp:
        print("[config] temp_feature=OFF (no temperature data for this city)")
    else:
        print(f"[config] temp_feature={'ON' if temp_feat_on else 'OFF'}")

    # ELEVATION FEATURE: a never-masked node channel from the static DEM elevation.
    # Unlike ERA5 temperature, elevation genuinely varies in space AND is knowable
    # at any (even sensor-free) location from a DEM -- the standard covariate in
    # land-use-regression / kriging-with-external-drift. Standardized across nodes;
    # static in time, so it's the same column every timestep.
    elev_feat_on = USE_ELEV_FEATURE
    if elev_feat_on:
        e = np.asarray(elev, dtype=np.float64)
        elev_std = torch.tensor((e - e.mean()) / (e.std() + 1e-6),
                                dtype=torch.float).reshape(-1, 1)   # [N, 1]
        print(f"[elev] standardized DEM elevation feature: spread "
              f"{e.max() - e.min():.0f} m across {N} nodes")
    print(f"[config] elev_feature={'ON' if elev_feat_on else 'OFF'}")

    # never-masked feature channels appended after PM2.5 (col 0). masked_step hides
    # only col 0, so temperature/elevation stay visible at the hidden node -- exactly
    # what's available when interpolating to a location with no PM sensor.
    node_in = 1 + int(temp_feat_on) + int(elev_feat_on)

    def node_features(t: int):
        """Node input rows for timestep t: [N, node_in]. col 0 = PM2.5 (z, maskable);
        then temperature and/or elevation channels (never masked)."""
        cols = [z_t[t].reshape(-1, 1)]
        if temp_feat_on:
            cols.append(temp_std[t].reshape(-1, 1))
        if elev_feat_on:
            cols.append(elev_std)
        return torch.cat(cols, dim=1) if len(cols) > 1 else cols[0].clone()

    model = GraPhyNet(node_in=node_in, edge_in=edge_attr_t.shape[-1], hidden=8, layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    # SPATIAL HOLDOUT: for each node, the indices of ITSELF + its HOLDOUT_RING
    # nearest sensors (by UTM distance). Hiding this whole disk leaves the target
    # with no nearby known sensor -> the "no monitor here" interpolation regime.
    node_dist = np.hypot(x_m[:, None] - x_m[None, :], y_m[:, None] - y_m[None, :])
    hide_idx = torch.tensor(np.argsort(node_dist, axis=1)[:, :HOLDOUT_RING + 1],
                            dtype=torch.long)                # [N, ring+1], col0=self
    print(f"[config] eval={'SPATIAL_HOLDOUT ring=' + str(HOLDOUT_RING) if SPATIAL_HOLDOUT else 'leave-one-node-out'}")

    def masked_step(t: int, train: bool):
        """One masked-imputation step at timestep t; returns MSE over the targets.

        leave-one-node-out: hide a random subset of known nodes, score them.
        spatial holdout: pick one known target, hide it + its nearest ring, score
        just the target (it now has no nearby known sensor)."""
        known = torch.nonzero(obs_t[t], as_tuple=False).squeeze(-1)
        node_temp = None if temp_z_t is None else temp_z_t[t]
        if SPATIAL_HOLDOUT:
            if len(known) < HOLDOUT_RING + 2:
                return None
            target = int(known[torch.randint(len(known), (1,))])
            x = node_features(t)
            x[hide_idx[target], 0] = UNKNOWN          # blank the target + its ring
            pred = model(x, edge_index, edge_weight, edge_attr_t[t], delev, node_temp)
            return loss_fn(pred[target, 0], z_t[t][target])

        if len(known) < 3:
            return None
        n_hide = max(1, int(len(known) * MASK_FRAC))
        target_nodes = known[torch.randperm(len(known))[:n_hide]]
        x = node_features(t)
        x[target_nodes, 0] = UNKNOWN  # hide only the PM column; temp stays visible
        # this hour's wind (edge_attr) + elevation gate (delev) + temperature gate (node_temp)
        pred = model(x, edge_index, edge_weight, edge_attr_t[t], delev, node_temp)
        return loss_fn(pred[target_nodes, 0], z_t[t][target_nodes])

    print(
        f"\ntraining: N={N} nodes, {len(train_ts)} train / {len(val_ts)} val "
        f"timesteps, hide {MASK_FRAC:.0%} of known nodes per step\n"
    )

    hist = {"epoch": [], "train": [], "val": []}  # loss curves for the plot
    val_sample = rng.choice(val_ts, size=min(200, len(val_ts)), replace=False)
    for epoch in range(EPOCHS):
        model.train()
        batch = rng.choice(train_ts, size=STEPS_PER_EPOCH, replace=False)
        opt.zero_grad()
        losses = [masked_step(int(t), train=True) for t in batch]
        losses = [l for l in losses if l is not None]
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vl = (
                torch.stack(
                    [
                        l
                        for t in val_sample
                        if (l := masked_step(int(t), train=False)) is not None
                    ]
                )
                .mean()
                .item()
            )
        hist["epoch"].append(epoch + 1)
        hist["train"].append(loss.item())
        hist["val"].append(vl)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch + 1:3d}  train_mse(z)={loss.item():.4f}  "
                f"val_mse(z)={vl:.4f}"
            )

    # learned gate parameters: did the optimizer actually USE the gates, or
    # leave them at their inert init (elev h_up/h_down ~ init, temp a ~ 0)?
    with torch.no_grad():
        temp_a = [round(float(l.temp_gate.a), 3) for l in model.layers]
        elev_h = [tuple(round(float(v), 1) for v in l.elev_gate.scales()) for l in model.layers]
    print(f"\n[gates] learned temp a per layer (0=inert): {temp_a}")
    print(f"[gates] learned elev (h_up,h_down) m per layer: {elev_h}")

    # -----------------------------------------------------------------------
    # evaluate: held-out imputation on every val timestep, in real ug/m3
    # -----------------------------------------------------------------------
    model.eval()
    rows = []
    with torch.no_grad():
        for t in val_ts:
            known = torch.nonzero(obs_t[t], as_tuple=False).squeeze(-1)
            node_temp = None if temp_z_t is None else temp_z_t[t]

            if SPATIAL_HOLDOUT:
                # up to 5 random targets/timestep, each with its own hidden disk,
                # scored alone -- the "no nearby monitor" interpolation regime.
                if len(known) < HOLDOUT_RING + 2:
                    continue
                sel = known[torch.randperm(len(known))[:5]].tolist()
                for target in sel:
                    x = node_features(t)
                    x[hide_idx[target], 0] = UNKNOWN
                    pred_z = model(x, edge_index, edge_weight, edge_attr_t[t], delev, node_temp)[:, 0]
                    rows.append({
                        "timestamp": pm.index[t], "station_id": ids[target],
                        "pm25_true": np.expm1(z[t, target] * sigma + mu),
                        "pm25_pred": np.expm1(pred_z[target].item() * sigma + mu),
                    })
                continue

            if len(known) < 3:
                continue
            n_hide = max(1, int(len(known) * MASK_FRAC))
            target_nodes = known[torch.randperm(len(known))[:n_hide]]
            x = node_features(t)
            x[target_nodes, 0] = UNKNOWN
            pred_z = model(x, edge_index, edge_weight, edge_attr_t[t], delev, node_temp)[:, 0]
            for node in target_nodes.tolist():
                true_ug = np.expm1(z[t, node] * sigma + mu)
                pred_ug = np.expm1(pred_z[node].item() * sigma + mu)
                rows.append(
                    {
                        "timestamp": pm.index[t],
                        "station_id": ids[node],
                        "pm25_true": true_ug,
                        "pm25_pred": pred_ug,
                    }
                )

    ev = pd.DataFrame(rows)
    mae = (ev["pm25_pred"] - ev["pm25_true"]).abs().mean()
    corr = ev["pm25_true"].corr(ev["pm25_pred"])
    baseline = (
        (ev["pm25_true"] - ev["pm25_true"].mean()).abs().mean()
    )  # predict-the-mean

    # ROC-AUC of the exceedance task: can pred rank which nodes cross EXCEED_UG?
    # (regression -> ranking metric; pred value is the score, true>thr the label)
    labels = (ev["pm25_true"] > EXCEED_UG).to_numpy()
    n_pos = int(labels.sum())
    if 0 < n_pos < len(labels):
        auc = roc_auc_score(labels, ev["pm25_pred"].to_numpy())
    else:
        auc = float("nan")  # AUC undefined when one class is empty

    print(f"\nHELD-OUT IMPUTATION ({len(ev)} masked nodes over val set):")
    print(
        f"  MAE            = {mae:6.2f} ug/m3   (predict-the-mean baseline = {baseline:6.2f})"
    )
    print(f"  corr(true,pred)= {corr:6.3f}")
    print(
        f"  ROC-AUC        = {auc:6.3f}   (exceedance >{EXCEED_UG} ug/m3: "
        f"{n_pos}/{len(labels)} positive)"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    elev_tag = "elevON" if USE_ELEVATION else "elevOFF"   # folder says which config
    temp_tag = "tempON" if temp_on else "tempOFF"
    if temp_feat_on:
        temp_tag += "_tfeatON"
    if elev_feat_on:
        temp_tag += "_efeatON"
    # trace the run to its data: which city + which sensor group produced it,
    # e.g. train_20260707_..._slc-urban_elevON_tempOFF (so runs never blur together).
    data_tag = f"{bg.CITY}-{bg.SENSOR_SET}"
    run_dir = (Path(__file__).resolve().parents[2] / "outputs" / "runs" /
               f"train_{run_id}_{data_tag}_{elev_tag}_{temp_tag}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # the numbers we actually read: a small metrics.json summarising the run,
    # plus the settings that produced it so a folder is self-explaining.
    metrics = {
        "run_id": run_id,
        "city": bg.CITY,
        "sensor_set": bg.SENSOR_SET,
        "has_wind": bool(has_wind),
        "has_temp": bool(has_temp),
        "use_elevation": USE_ELEVATION,
        "use_temperature": bool(temp_on),
        "use_temp_feature": bool(temp_feat_on),
        "use_elev_feature": bool(elev_feat_on),
        "n_nodes": int(N),
        "epochs": EPOCHS,
        "val_frac": VAL_FRAC,
        "mask_frac": MASK_FRAC,
        "n_masked_eval": int(len(ev)),
        "mae_ug": round(float(mae), 3),
        "mae_baseline_ug": round(float(baseline), 3),
        "skill_vs_baseline": round(float(1 - mae / baseline), 3),
        "corr": round(float(corr), 3),
        "roc_auc": round(float(auc), 3),
        "exceed_ug": EXCEED_UG,
        "n_exceed": n_pos,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    ev.to_csv(run_dir / "predictions.csv", index=False)  # raw rows, for recompute
    print(f"\n[saved] metrics  -> {run_dir / 'metrics.json'}")
    print(f"[saved] raw csv  -> {run_dir / 'predictions.csv'}")

    plot_results(hist, ev, mae, baseline, corr, run_dir)
    return metrics


def plot_results(hist, ev, mae, baseline, corr, run_dir: Path):
    """Two panels: the loss curve (is it learning?) and true-vs-pred (is it right?)."""
    INK, MUTED, GRID = "#1f2933", "#6b7280", "#d9dee3"
    TRAIN_C, VAL_C, PT_C = "#3b7dd8", "#e8833a", "#3b7dd8"  # blue=train, orange=val
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))

    # --- panel 1: training vs validation loss (log y: epoch-1 loss is huge) ----
    axL.plot(hist["epoch"], hist["train"], color=TRAIN_C, lw=2, label="train")
    axL.plot(hist["epoch"], hist["val"], color=VAL_C, lw=2, label="validation")
    axL.set_yscale("log")
    axL.set(
        xlabel="epoch",
        ylabel="masked MSE (log1p z-space, log scale)",
        title="Is it learning?  loss per epoch",
    )
    axL.grid(True, color=GRID, lw=0.6, alpha=0.7)
    axL.legend(frameon=False)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # --- panel 2: held-out true vs predicted PM2.5 (log-log: values span 0..1600) ---
    t = ev["pm25_true"].to_numpy() + 1.0  # +1 so zeros are plottable on log axis
    p = ev["pm25_pred"].to_numpy() + 1.0
    lim = [1, max(t.max(), p.max()) * 1.1]
    axR.plot(lim, lim, color=MUTED, lw=1.5, ls="--", zorder=1, label="perfect (y = x)")
    axR.scatter(t, p, s=18, color=PT_C, alpha=0.35, edgecolors="none", zorder=2)
    axR.set(
        xscale="log",
        yscale="log",
        xlim=lim,
        ylim=lim,
        xlabel="true PM2.5 + 1  (ug/m3)",
        ylabel="predicted PM2.5 + 1  (ug/m3)",
        title="Is it right?  held-out imputation",
    )
    axR.grid(True, color=GRID, lw=0.6, alpha=0.7)
    axR.legend(frameon=False, loc="upper left")
    axR.text(
        0.97,
        0.05,
        f"MAE = {mae:.1f}  (mean-baseline {baseline:.1f})\ncorr = {corr:.3f}",
        transform=axR.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID),
    )
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    fig.suptitle("PM2.5 training results", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    path = run_dir / "training_results.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[saved] plot -> {path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="train GraPhyNet PM2.5 imputer")
    p.add_argument("--city", choices=list(bg.CITY_CONFIG), default=bg.CITY,
                   help=f"which city to train on (default: {bg.CITY}). also swaps "
                        f"coords/PM2.5/wind/temperature dirs + UTM zone.")
    p.add_argument("--sensor-set", default=None,
                   help="which sensor group in that city (e.g. urban/rural). "
                        "default: the city's first group.")
    p.add_argument("--wind", choices=["era5", "hrrr", "zero"], default=None,
                   help="wind field feeding convection: era5 (~30km, flat), "
                        "hrrr (3km, spatially structured), zero (convection off). "
                        "Use with --allow-missing-inputs for zero.")
    p.add_argument("--no-elevation", action="store_true",
                   help="disable the elevation gate (Δelev=0 -> gate=1); for ablation")
    p.add_argument("--no-temperature", action="store_true",
                   help="disable the temperature gate (node_temp=None -> gate=1); for ablation")
    p.add_argument("--no-temp-feature", action="store_true",
                   help="disable the temperature node-feature channel (node_in back to 1); for ablation")
    p.add_argument("--no-elev-feature", action="store_true",
                   help="disable the DEM elevation node-feature channel; for ablation")
    p.add_argument("--spatial-holdout", action="store_true",
                   help="eval by hiding a target + its nearest neighbours (no nearby "
                        "known sensor), the interpolation regime where covariates matter")
    p.add_argument("--holdout-ring", type=int, default=None, metavar="N",
                   help="how many nearest sensors to blank around the target (spatial mode)")
    p.add_argument("--rebuild", action="store_true",
                   help="ignore the processed-data cache and preprocess the raw "
                        "CSVs fresh (USE_CACHE=False)")
    p.add_argument("--allow-missing-inputs", action="store_true",
                   help="don't abort when wind/distance/PM2.5 are missing; "
                        "train anyway with zero-filled features")
    p.add_argument("--synth-elev-grad", type=float, default=0.0, metavar="K",
                   help="sanity test: inject a known PM2.5 gradient of K ug/m3 "
                        "per metre of elevation (0=off/real data). Run with "
                        "--synth-elev-grad K both WITH and WITHOUT --no-elevation "
                        "to see if the gate can recover a planted gradient.")
    p.add_argument("--synth-temp-grad", type=float, default=0.0, metavar="K",
                   help="sanity test for the TEMPERATURE gate: scale each cell's "
                        "spatial anomaly by clip(1+K*s) so cold nodes go flat and "
                        "hot nodes amplify (0=off/real data). Run both WITH and "
                        "WITHOUT --no-temperature to see if the gate can exploit it.")
    p.add_argument("--synth-temp-direct", type=float, default=0.0, metavar="K",
                   help="sanity test for the temp FEATURE: plant PM2.5 = mean + "
                        "K*20*s directly from each node's own temperature. Run "
                        "with vs --no-temp-feature to see if the feature exploits it.")
    a = p.parse_args()
    # point the build_graph2 globals at the requested city/group before main()
    # reads bg.CITY / bg.SENSOR_SET (mirrors how the smoke tests drove it).
    bg.use_city(a.city)
    bg.SENSOR_SET = a.sensor_set or next(iter(bg.GROUP_CONFIG))
    if bg.SENSOR_SET not in bg.GROUP_CONFIG:
        p.error(f"--sensor-set {bg.SENSOR_SET!r} not in {a.city} groups "
                f"{list(bg.GROUP_CONFIG)}")
    if a.wind is not None:
        WIND_SOURCE = a.wind
    if a.rebuild:
        USE_CACHE = False
    if a.no_elevation:
        USE_ELEVATION = False
    if a.no_temperature:
        USE_TEMPERATURE = False
    if a.no_temp_feature:
        USE_TEMP_FEATURE = False
    if a.no_elev_feature:
        USE_ELEV_FEATURE = False
    if a.spatial_holdout:
        SPATIAL_HOLDOUT = True
    if a.holdout_ring is not None:
        HOLDOUT_RING = a.holdout_ring
    if a.allow_missing_inputs:
        STRICT_INPUTS = False
    SYNTH_ELEV_K = a.synth_elev_grad
    SYNTH_TEMP_K = a.synth_temp_grad
    SYNTH_TEMPDIRECT_K = a.synth_temp_direct
    main()
