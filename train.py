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

import build_graph as bg
import preprocessing as pp
from convection import edge_bearings, wind_edge_features
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
    """Return everything that doesn't change across timesteps + the node tables."""
    long_df = bg.load_sensor_data()
    ids = sorted(long_df["station_id"].unique())
    pm_raw = (
        long_df.pivot_table(index="timestamp", columns="station_id", values="pm25")
        .reindex(columns=ids)
        .sort_index()
    )

    # drop full-year duds + get the per-cell observed mask
    pm, observed, ids, _ = pp.preprocess(pm_raw)

    coords = bg.get_coordinates(ids)
    ids = coords["station_id"].tolist()
    x_m, y_m = bg.project(coords["lat"].to_numpy(), coords["lon"].to_numpy())
    pm = pm.reindex(columns=ids)
    observed = observed.reindex(columns=ids)

    edge_index = bg.knn_edges(x_m, y_m, bg.K)
    dist = bg.distance_matrix(x_m, y_m)
    edge_dist = np.array([dist[i, j] for i, j in edge_index.t()])
    edge_weight = inverse_distance_weights(torch.tensor(edge_dist, dtype=torch.float))

    bearing = edge_bearings(x_m, y_m, edge_index.numpy())
    WIND_DIR, WIND_SPEED = np.deg2rad(270.0), 3.0  # one city-wide wind (placeholder)
    edge_attr = wind_edge_features(edge_dist, bearing, WIND_DIR, WIND_SPEED)

    return ids, pm, observed, edge_index, edge_weight, edge_attr


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    ids, pm, observed, edge_index, edge_weight, edge_attr = build_static_graph()
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

    model = GraPhyNet(node_in=1, edge_in=edge_attr.shape[1], hidden=8, layers=3)
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
        pred = model(x, edge_index, edge_weight, edge_attr)
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
            pred_z = model(x, edge_index, edge_weight, edge_attr)[:, 0]
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
