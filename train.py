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

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save a PNG instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import build_graph2 as bg
import preprocessing as pp
from diffusion import inverse_distance_weights
from model import GraPhyNet

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
SEED = 0
EPOCHS = 80
STEPS_PER_EPOCH = 64  # random timesteps sampled per epoch
MASK_FRAC = 0.20  # fraction of a timestep's KNOWN nodes to hide + predict
LR = 0.01
VAL_FRAC = 0.15  # last chunk of timesteps held out for evaluation
UNKNOWN = 0.0  # placeholder fed for hidden/missing nodes (in z-space)


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

    long_pm = bg.load_air_quality(cfg["purple_air_dir"], coords["station_id"].tolist())
    # keep only ids present in BOTH coordinates and air-quality data
    station_ids = sorted(set(coords["station_id"]) & set(long_pm["station_id"]))
    coords = coords[coords["station_id"].isin(station_ids)].sort_values("station_id")
    station_ids = coords["station_id"].tolist()

    pm_raw = (
        long_pm.pivot_table(index="timestamp", columns="station_id", values="pm25")
        .reindex(columns=station_ids)
        .sort_index()
    )

    # real wind, interpolated onto the hourly PM2.5 grid (per surviving sensor)
    u10_wide, v10_wide, has_wind = bg.load_wind(
        cfg["wind_zip"], cfg["wind_dir"], station_ids, pm_raw.index
    )

    # drop full-year duds + get the per-cell observed mask
    pm, observed, kept_ids, _ = pp.preprocess(pm_raw)

    # restrict everything to the surviving nodes; node order = sorted kept ids
    coords = coords[coords["station_id"].isin(kept_ids)].sort_values("station_id")
    ids = coords["station_id"].tolist()
    pm = pm.reindex(columns=ids)
    observed = observed.reindex(columns=ids)
    u10_wide = u10_wide.reindex(columns=ids)
    v10_wide = v10_wide.reindex(columns=ids)

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
        f"edges={edge_index.shape[1]}  timesteps={len(pm)}  has_wind={has_wind}"
    )
    return ids, pm, observed, edge_index, edge_weight, edge_attr_t


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    ids, pm, observed, edge_index, edge_weight, edge_attr_t = build_static_graph()
    N = len(ids)

    # normalise in log1p space using ONLY observed training cells (no leakage).
    T = len(pm)
    n_val = int(T * VAL_FRAC)
    train_ts = np.arange(0, T - n_val)
    val_ts = np.arange(T - n_val, T)

    values = pm.to_numpy(dtype=np.float64)  # [T, N], already NaN-filled
    obs = observed.to_numpy()  # [T, N] bool
    logv = np.log1p(np.clip(values, 0, None))
    train_obs_vals = logv[np.ix_(train_ts, np.arange(N))][obs[train_ts]]
    mu, sigma = train_obs_vals.mean(), train_obs_vals.std() + 1e-8
    z = (logv - mu) / sigma  # standardised targets/inputs

    z_t = torch.tensor(z, dtype=torch.float)
    obs_t = torch.tensor(obs)

    model = GraPhyNet(node_in=1, edge_in=edge_attr_t.shape[-1], hidden=8, layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    def masked_step(t: int, train: bool):
        """Hide a random subset of known nodes at timestep t; predict them."""
        known = torch.nonzero(obs_t[t], as_tuple=False).squeeze(-1)
        if len(known) < 3:
            return None
        n_hide = max(1, int(len(known) * MASK_FRAC))
        perm = torch.randperm(len(known))[:n_hide]
        target_nodes = known[perm]

        x = z_t[t].clone().reshape(-1, 1)
        x[target_nodes, 0] = UNKNOWN  # hide them from the input
        pred = model(x, edge_index, edge_weight, edge_attr_t[t])  # this hour's wind
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

    # -----------------------------------------------------------------------
    # evaluate: held-out imputation on every val timestep, in real ug/m3
    # -----------------------------------------------------------------------
    model.eval()
    rows = []
    with torch.no_grad():
        for t in val_ts:
            known = torch.nonzero(obs_t[t], as_tuple=False).squeeze(-1)
            if len(known) < 3:
                continue
            n_hide = max(1, int(len(known) * MASK_FRAC))
            target_nodes = known[torch.randperm(len(known))[:n_hide]]
            x = z_t[t].clone().reshape(-1, 1)
            x[target_nodes, 0] = UNKNOWN
            pred_z = model(x, edge_index, edge_weight, edge_attr_t[t])[:, 0]
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
    print(f"\nHELD-OUT IMPUTATION ({len(ev)} masked nodes over val set):")
    print(
        f"  MAE            = {mae:6.2f} ug/m3   (predict-the-mean baseline = {baseline:6.2f})"
    )
    print(f"  corr(true,pred)= {corr:6.3f}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parent / "outputs" / "runs" / f"train_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "predictions.csv"
    ev.to_csv(out, index=False)
    print(f"\n[saved] held-out predictions -> {out}")
    print(ev.head(12).round(2).to_string(index=False))

    plot_results(hist, ev, mae, baseline, corr, run_dir)


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
    main()
