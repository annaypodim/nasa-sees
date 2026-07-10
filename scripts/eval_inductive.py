"""
eval_inductive.py  -  reproduce GraPhy's INDUCTIVE spatial-kriging protocol so our
MAE is directly comparable to their reported 2.38 ug/m3 on Fresno.

WHY A SEPARATE DRIVER (vs train.py)
    train.py evaluates TRANSDUCTIVELY: it masks nodes inside ONE sensor set that
    is used for both training and scoring (leave-one-out / spatial-holdout). GraPhy
    instead holds out whole SENSORS: train on ~28 sensors, then predict PM2.5 at
    ~9 sensors the model never saw during training, across ALL hours. That is a
    different, harder, and standard "kriging to an unmonitored location" protocol,
    and it is the one whose number (MAE 2.38 ug/m3, Fresno, Oct'23-Jan'24) we are
    trying to reproduce. Mixing the two protocols is exactly why our earlier
    numbers were not comparable to the paper.

PROTOCOL (matches GraPhy)
    * Split the N sensors into DISJOINT train/val/test sets (default 28:4:9 ->
      scaled to N). The split is by SENSOR, not by time.
    * The val+test sensors' PM2.5 input is ALWAYS masked (never fed), so the model
      is genuinely inductive -- it has never seen their values. Their graph edges
      to train sensors still carry message passing (that is how kriging works).
    * Train: each step, additionally hide a random subset of TRAIN sensors and
      regress them (self-supervised imputation, same objective as train.py).
    * Eval: over EVERY hour, predict the held-out TEST sensors and score in real
      ug/m3. Report MAE / RMSE / R2 / corr, plus an inverse-distance-weighting
      (IDW) baseline for a sanity floor (GraPhy also benchmarks IDW).

USAGE
    .venv/bin/python scripts/eval_inductive.py --city fresno --wind hrrr \
        --epochs 120 --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from src.graph import build_graph2 as bg
from src.model import train as tr
from src.model.model import GraPhyNet

UNKNOWN = 0.0  # placeholder fed for masked nodes, in z-space (train.py uses the same)


def split_nodes(N: int, n_val: int, n_test: int, rng: np.random.Generator):
    """Disjoint train/val/test node-index sets (by sensor). Test+val are held out."""
    perm = rng.permutation(N)
    test = np.sort(perm[:n_test])
    val = np.sort(perm[n_test:n_test + n_val])
    train = np.sort(perm[n_test + n_val:])
    return train, val, test


def idw_baseline(values_ug, obs, x_m, y_m, train_idx, test_idx, power=2.0):
    """Inverse-distance-weighting kriging floor: predict each test sensor at each
    hour as the IDW of the OBSERVED train sensors that hour. Model-free sanity."""
    d = np.hypot(x_m[test_idx, None] - x_m[None, train_idx],
                 y_m[test_idx, None] - y_m[None, train_idx])  # [n_test, n_train]
    w_all = 1.0 / np.maximum(d, 1.0) ** power
    preds, trues = [], []
    for t in range(values_ug.shape[0]):
        tr_obs = obs[t, train_idx]
        if tr_obs.sum() == 0:
            continue
        w = w_all[:, tr_obs]
        vals = values_ug[t, train_idx][tr_obs]
        est = (w * vals[None, :]).sum(1) / w.sum(1)  # [n_test]
        for k, node in enumerate(test_idx):
            if obs[t, node]:
                preds.append(est[k]); trues.append(values_ug[t, node])
    return np.array(trues), np.array(preds)


def metrics(true, pred):
    err = pred - true
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    ss_res = (err ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = np.corrcoef(true, pred)[0, 1] if len(true) > 1 else float("nan")
    return dict(mae=mae, rmse=rmse, r2=r2, corr=corr, n=len(true))


def run_seed(seed, graph, args):
    (ids, pm, observed, edge_index, edge_weight, edge_attr_t, edge_delev, elev,
     x_m, y_m, has_wind, temp_wide, has_temp) = graph
    N, T = len(ids), len(pm)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # scale the 28:4:9 GraPhy split to N sensors
    n_test = max(1, round(N * 9 / 41))
    n_val = max(1, round(N * 4 / 41))
    train_idx, val_idx, test_idx = split_nodes(N, n_val, n_test, rng)
    hidden_input = np.concatenate([val_idx, test_idx])  # never fed their PM

    values = pm.to_numpy(dtype=np.float64)  # [T, N] ug/m3, NaN already filled to 0
    obs = observed.to_numpy()

    # transform + z-score using ONLY observed TRAIN-sensor cells (no leakage).
    # log-space MSE optimizes the geometric error and underpredicts peaks -> it can
    # inflate LINEAR MAE on pollution episodes (and lose to IDW). --linear trains in
    # raw ug/m3 z-space, which optimizes the metric we report (closer to GraPhy).
    base = np.clip(values, 0, None)
    tv = base if args.linear else np.log1p(base)
    train_cells = tv[:, train_idx][obs[:, train_idx]]
    mu, sigma = train_cells.mean(), train_cells.std() + 1e-8
    z = (tv - mu) / sigma
    z_t = torch.tensor(z, dtype=torch.float)
    obs_t = torch.tensor(obs)

    def to_ug(zval):  # invert z-score (+ optional log) back to ug/m3
        raw = zval * sigma + mu
        return raw if args.linear else np.expm1(raw)

    # optional never-masked elevation feature (GraPhy base = OFF; ablation = ON)
    node_in = 1
    elev_col = None
    if args.elev_feature:
        e = np.asarray(elev, dtype=np.float64)
        elev_col = torch.tensor((e - e.mean()) / (e.std() + 1e-6),
                                dtype=torch.float).reshape(-1, 1)
        node_in = 2

    def node_features(t):
        x = z_t[t].reshape(-1, 1).clone()
        x[hidden_input, 0] = UNKNOWN          # val+test never reveal their PM
        if elev_col is not None:
            return torch.cat([x, elev_col], dim=1)
        return x

    delev = edge_delev if args.elev_gate else torch.zeros_like(edge_delev)

    model = GraPhyNet(node_in=node_in, edge_in=edge_attr_t.shape[-1],
                      hidden=args.hidden, layers=args.layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    train_ts = np.arange(T)
    for epoch in range(args.epochs):
        model.train()
        batch = rng.choice(train_ts, size=min(args.steps, T), replace=False)
        opt.zero_grad()
        losses = []
        for t in batch:
            known_train = train_idx[obs[t, train_idx]]
            if len(known_train) < 3:
                continue
            n_hide = max(1, int(len(known_train) * args.mask_frac))
            targets = rng.choice(known_train, size=n_hide, replace=False)
            x = node_features(int(t))
            x[targets, 0] = UNKNOWN
            pred = model(x, edge_index, edge_weight, edge_attr_t[t], delev,
                         None)[:, 0]
            losses.append(loss_fn(pred[targets], z_t[t][targets]))
        if not losses:
            continue
        torch.stack(losses).mean().backward()
        opt.step()

    # -------- inductive eval: predict TEST sensors over every hour ----------
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for t in range(T):
            if obs[t, test_idx].sum() == 0:
                continue
            x = node_features(t)                  # test PM already masked
            pred_z = model(x, edge_index, edge_weight, edge_attr_t[t], delev,
                           None)[:, 0]
            for node in test_idx:
                if obs[t, node]:
                    trues.append(to_ug(z[t, node]))
                    preds.append(to_ug(pred_z[node].item()))
    trues, preds = np.array(trues), np.array(preds)
    m = metrics(trues, preds)

    # IDW floor on the same held-out test sensors
    it, ip = idw_baseline(values, obs, x_m, y_m, train_idx, test_idx)
    m_idw = metrics(it, ip)
    return m, m_idw, (len(train_idx), len(val_idx), len(test_idx))


def main():
    ap = argparse.ArgumentParser(description="Inductive kriging eval (GraPhy protocol).")
    ap.add_argument("--city", default="fresno")
    ap.add_argument("--sensor-set", default="urban")
    ap.add_argument("--wind", choices=["era5", "hrrr", "zero"], default="hrrr")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--mask-frac", type=float, default=0.25)
    ap.add_argument("--linear", action="store_true",
                    help="train/z-score in raw ug/m3 instead of log1p (optimizes "
                         "linear MAE, the reported metric; likely closer to GraPhy)")
    ap.add_argument("--elev-feature", action="store_true",
                    help="add never-masked DEM elevation channel (ablation; GraPhy base=off)")
    ap.add_argument("--elev-gate", action="store_true",
                    help="enable the elevation gate (ablation; GraPhy base=off)")
    args = ap.parse_args()

    bg.use_city(args.city)
    bg.SENSOR_SET = args.sensor_set
    tr.WIND_SOURCE = args.wind
    tr.STRICT_INPUTS = (args.wind != "zero")
    tr.USE_CACHE = False  # Fresno has no processed cache; build fresh from raw

    print(f"[repro] city={args.city} wind={args.wind} hidden={args.hidden} "
          f"layers={args.layers} epochs={args.epochs} elev_feat={args.elev_feature} "
          f"elev_gate={args.elev_gate}")
    graph = tr.build_static_graph()

    seeds = [int(s) for s in args.seeds.split(",")]
    ours, idws = [], []
    split = None
    for s in seeds:
        m, m_idw, split = run_seed(s, graph, args)
        ours.append(m); idws.append(m_idw)
        print(f"  seed {s}: OURS mae={m['mae']:.3f} rmse={m['rmse']:.3f} "
              f"r2={m['r2']:.3f} corr={m['corr']:.3f} (n={m['n']})   "
              f"IDW mae={m_idw['mae']:.3f}")

    def agg(rows, k):
        v = np.array([r[k] for r in rows])
        return v.mean(), v.std()

    print("\n" + "=" * 68)
    print(f"INDUCTIVE KRIGING  city={args.city}  split(train/val/test)={split}  "
          f"seeds={seeds}")
    print("=" * 68)
    for name, rows in [("OURS (GraPhyNet)", ours), ("IDW baseline", idws)]:
        mae_m, mae_s = agg(rows, "mae")
        rmse_m, _ = agg(rows, "rmse")
        r2_m, _ = agg(rows, "r2")
        corr_m, _ = agg(rows, "corr")
        print(f"{name:20s}  MAE={mae_m:.3f}±{mae_s:.3f}  RMSE={rmse_m:.3f}  "
              f"R2={r2_m:.3f}  corr={corr_m:.3f}")
    print(f"{'GraPhy (paper)':20s}  MAE=2.380   (Fresno, 41 sensors, Oct23-Jan24)")
    print("=" * 68)


if __name__ == "__main__":
    main()
