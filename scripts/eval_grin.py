"""
eval_grin.py  -  run GRIN (Cini et al., ICLR 2022; the field-standard graph
imputation model, still benchmarked in 2024-25 papers) as a LEARNED baseline on
OUR data, under OUR EXACT protocol, so its MAE is directly comparable to the
OURS / ST-kriging numbers from eval_temporal.py.

WHY THIS DRIVER
    Published GRIN numbers (e.g. AQI-36 Beijing MAE ~12) are on different data and
    tell us nothing about Fresno/SLC/Pittsburgh. The only rigorous "we beat a
    learned SOTA baseline" claim is to retrain GRIN's ARCHITECTURE on our panels
    with our splits. We borrow only the model class (tsl.nn.models.stgn.GRINModel)
    and drive it ourselves: same city panel, same sqrt transform, same z-scoring on
    KNOWN cells, the SAME 5-fold gap partition (kfold_gaps) and the SAME MC train
    gaps as eval_temporal, and the SAME metrics()/inverse-transform. GRIN is
    TERRAIN-BLIND (no elevation), so beating it isolates our terrain contribution.

PROTOCOL
    For each fold k: known = obs & ~test_gap & ~train_gap are GRIN's observed inputs
    (mask=1); train_gap cells are the training targets (hidden from input, truth
    known); test_gap cells are scored. We tile the T-hour series into windows of
    length --window (default 168h = 1 week) so every gap sits inside a window with
    flanking observed temporal context. GRIN reconstructs the whole window; loss is
    MAE on train_gap cells, evaluation is MAE on test_gap cells (real ug/m3).

USAGE
    PYTHONPATH=. .venv/bin/python scripts/eval_grin.py --city fresno --wind hrrr \
        --gap-len 24 --kfold 5 --epochs 80
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import torch

from src.graph import build_graph2 as bg
from src.model import train as tr
from scripts.eval_inductive import metrics
from scripts.eval_temporal import kfold_gaps, make_gaps

warnings.filterwarnings("ignore")


def windows(T, W):
    """Non-overlapping window starts tiling [0,T); last window right-aligned so
    every hour is covered exactly once (the tail window may overlap the previous)."""
    starts = list(range(0, T - W + 1, W))
    if not starts or starts[-1] + W < T:
        starts.append(max(0, T - W))
    return starts


def run_fold(seed, graph, args, test_gap):
    (ids, pm, observed, edge_index, edge_weight, edge_attr_t, edge_delev, elev,
     x_m, y_m, has_wind, temp_wide, has_temp) = graph
    N, T = len(ids), len(pm)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    values = np.clip(pm.to_numpy(dtype=np.float64), 0, None)
    obs = observed.to_numpy()

    if args.transform == "sqrt":
        tv = np.sqrt(values); inv = lambda r: np.clip(r, 0, None) ** 2
    elif args.transform == "linear":
        tv = values; inv = lambda r: np.clip(r, 0, None)
    else:
        tv = np.log1p(values); inv = lambda r: np.clip(np.expm1(r), 0, None)

    # MC train gaps drawn exactly as eval_temporal (leak-safe, disjoint from test)
    train_gap = make_gaps(obs & ~test_gap, N, T, args.gap_len, args.n_gaps * 3, rng)
    known = obs & ~test_gap & ~train_gap

    mu, sigma = tv[known].mean(), tv[known].std() + 1e-8
    z = (tv - mu) / sigma
    z_in = np.where(known, z, 0.0)                       # GRIN sees 0 where hidden

    def to_ug(zv):
        return inv(zv * sigma + mu)

    z_t = torch.tensor(z, dtype=torch.float)
    zin_t = torch.tensor(z_in, dtype=torch.float)
    known_t = torch.tensor(known)
    train_t = torch.tensor(train_gap)
    test_t = torch.tensor(test_gap)
    ei = edge_index
    ew = edge_weight

    from tsl.nn.models.stgn import GRINModel
    model = GRINModel(input_size=1, hidden_size=args.hidden, ff_size=args.hidden * 2,
                      embedding_size=8, n_nodes=N, n_layers=args.layers,
                      kernel_size=2, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    W = min(args.window, T)
    starts = windows(T, W)

    def make_batch(s):
        sl = slice(s, s + W)
        x = zin_t[sl].reshape(1, W, N, 1)                # [B=1,W,N,1]
        m = known_t[sl].reshape(1, W, N, 1).float()      # observed-input mask
        return x, m, sl

    # ---- train: reconstruct train_gap cells within each window -----------------
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(starts))
        losses = []
        opt.zero_grad()
        for wi in order:
            s = starts[wi]
            x, m, sl = make_batch(s)
            tgt = train_t[sl]                            # [W,N] bool targets
            if not tgt.any():
                continue
            out = model(x, ei, ew, mask=m)
            imp = out[0] if isinstance(out, (list, tuple)) else out
            imp = imp.reshape(W, N)
            pred = imp[tgt]
            true = z_t[sl][tgt]
            losses.append(torch.nn.functional.l1_loss(pred, true))
        if not losses:
            continue
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # ---- eval: predict test_gap cells ------------------------------------------
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for s in starts:
            x, m, sl = make_batch(s)
            tgt = test_t[sl]
            if not tgt.any():
                continue
            out = model(x, ei, ew, mask=m)
            imp = out[0] if isinstance(out, (list, tuple)) else out
            imp = imp.reshape(W, N)
            idx = tgt.nonzero()
            for r, c in idx.tolist():
                trues.append(to_ug(z_t[sl][r, c].item()))
                preds.append(to_ug(imp[r, c].item()))
    return metrics(np.array(trues), np.array(preds))


def main():
    ap = argparse.ArgumentParser(description="GRIN learned-baseline eval on our splits.")
    ap.add_argument("--city", default="fresno")
    ap.add_argument("--sensor-set", default="urban")
    ap.add_argument("--wind", choices=["era5", "hrrr", "zero"], default="hrrr")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--gap-len", type=int, default=24)
    ap.add_argument("--n-gaps", type=int, default=6)
    ap.add_argument("--window", type=int, default=168, help="GRIN window length (hours)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--transform", choices=["log", "linear", "sqrt"], default="sqrt")
    ap.add_argument("--probe", action="store_true",
                    help="timing probe: run only fold 0 with --epochs, report wall time")
    args = ap.parse_args()

    bg.use_city(args.city); bg.SENSOR_SET = args.sensor_set; bg.EPA_CORRECT = True
    tr.WIND_SOURCE = args.wind; tr.STRICT_INPUTS = (args.wind != "zero"); tr.USE_CACHE = False
    print(f"[grin] city={args.city} gap={args.gap_len}h kfold={args.kfold} "
          f"window={args.window} epochs={args.epochs} hidden={args.hidden}")
    graph = tr.build_static_graph()
    (ids, pm, observed, *_rest) = graph
    obs = observed.to_numpy()
    N, T = len(ids), len(pm)
    folds = kfold_gaps(obs, N, T, args.gap_len, args.kfold)

    rows = []
    fold_range = [0] if args.probe else range(args.kfold)
    for k in fold_range:
        t0 = time.time()
        m = run_fold(k, graph, args, folds[k])
        dt = time.time() - t0
        rows.append(m)
        print(f"  fold {k}: GRIN mae={m['mae']:.3f}  rmse={m['rmse']:.3f}  "
              f"r2={m['r2']:.3f}  (n={m['n']})  [{dt:.0f}s]")
        if args.probe:
            print(f"[probe] one fold = {dt:.0f}s @ {args.epochs} epochs -> "
                  f"full {args.kfold}-fold ~= {dt*args.kfold/60:.1f} min")
            return

    mae = np.array([r["mae"] for r in rows])
    print("\n" + "=" * 60)
    print(f"GRIN baseline  city={args.city}  gap={args.gap_len}h  {args.kfold}-fold CV")
    print(f"GRIN MAE = {mae.mean():.3f} +- {mae.std():.3f}  (compare to OURS/ST-kriging)")
    print("=" * 60)


if __name__ == "__main__":
    main()
