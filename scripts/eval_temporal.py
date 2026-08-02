"""
eval_temporal.py  -  LONG-GAP TEMPORAL imputation (the project's novelty axis).

WHY A SEPARATE DRIVER (vs eval_inductive.py)
    eval_inductive holds out whole SENSORS and predicts them from OTHER sensors at
    the same hour -- pure SPATIAL kriging, memoryless (GraPhy's protocol). That task
    can only use a covariate's SPATIAL variation, which is why common-mode temp and
    (redundant) AOD were dead there. This driver holds out contiguous TIME WINDOWS
    ("gaps") from otherwise-present sensors and predicts each sensor's OWN missing
    hours. Now the target's own HISTORY is a signal, and covariates whose variance
    lives in TIME (temp: temporal std 8.8C vs spatial 0.06; AOD: 0.22 vs 0.02) can
    finally act. This is where GraPhy's memorylessness is a real gap to beat.

WHAT STAYS THE SAME (the whole point)
    The model is UNCHANGED: same GraPhyNet, same message passing, same gates. Time
    enters as FEATURES (the node's own lags + persistence) and as a PRIOR term, not
    as recurrence -- so the per-timestep architecture is preserved exactly. Only the
    masking (time windows, not sensors) and a few lag features are new.

PRIOR = learnable blend of SPATIAL IDW and TEMPORAL PERSISTENCE (both feed the
    workhorse prior, per the kernel lesson that the prior -- not the squashed
    correction -- is where signal pays off):
        prior = b * persistence + (1-b) * spatial_IDW      (b = sigmoid(beta), learned)
        pred  = prior + GNN_correction
    persistence = the node's last KNOWN value carried forward through the gap. Long
    gaps make persistence stale -> that's exactly where temp/AOD should help.

LEAK SAFETY: known_mask = observed AND NOT (test gap OR train gap). All lag/
    persistence/IDW inputs are built from known_z (gap cells replaced by carry-
    forward), so neither a test-gap nor a train-gap target's true value ever feeds
    a feature or the prior. Carry-forward is computed ONCE per seed (gaps fixed).

USAGE
    PYTHONPATH=. .venv/bin/python scripts/eval_temporal.py --city pittsburgh \
        --wind hrrr --gap-len 24 --n-gaps 6 --temporal --seeds 0,1,2,3
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

from src.graph import build_graph2 as bg
from src.model import train as tr
from src.model.model import GraPhyNet
from scripts.eval_inductive import metrics, UNKNOWN

warnings.filterwarnings("ignore")


def carry_forward(z, known):
    """cf[t,n] = last KNOWN z strictly before t (0 in z-space = field mean if none).
    known: [T,N] bool. Vectorised per column would need a scan; T*N here is fine."""
    T, N = z.shape
    cf = np.zeros_like(z)
    last = np.zeros(N)
    seen = np.zeros(N, dtype=bool)
    for t in range(T):
        cf[t] = np.where(seen, last, 0.0)          # strictly-before value
        upd = known[t]
        last = np.where(upd, z[t], last)
        seen = seen | upd
    return cf


def make_gaps(obs, N, T, gap_len, n_gaps, rng):
    """Pick n_gaps non-overlapping fully-observed windows of length gap_len per
    sensor; return a [T,N] boolean mask of the gap (target) cells."""
    mask = np.zeros((T, N), dtype=bool)
    for n in range(N):
        col = obs[:, n]
        placed = 0
        tries = 0
        # candidate starts where the whole window is observed and not already gapped
        while placed < n_gaps and tries < n_gaps * 50:
            tries += 1
            s = int(rng.integers(0, max(1, T - gap_len)))
            w = slice(s, s + gap_len)
            if col[w].all() and not mask[w, n].any():
                mask[w, n] = True
                placed += 1
    return mask


def kfold_gaps(obs, N, T, gap_len, K):
    """Proper K-fold TEST partition over fixed-length gap windows.
    Tile each sensor's timeline into consecutive non-overlapping fully-observed
    windows of length gap_len; assign the j-th clean window of a sensor to fold
    (j % K). Returns K mutually-disjoint [T,N] boolean test masks whose union is
    every clean window -> every held-out cell is tested exactly once across folds.
    Sensor-interleaved (j % K) so each fold draws from every sensor & time region."""
    folds = [np.zeros((T, N), dtype=bool) for _ in range(K)]
    for n in range(N):
        col = obs[:, n]
        s = 0
        j = 0
        while s + gap_len <= T:
            if col[s:s + gap_len].all():
                folds[j % K][s:s + gap_len, n] = True
                s += gap_len
                j += 1
            else:
                s += 1                      # slide to the next clean window
    return folds


def run_seed(seed, graph, args, masks=None):
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

    # ---- gaps: disjoint TEST and TRAIN target windows ---------------------------
    # K-fold mode passes a fixed TEST partition mask; train gaps stay MC-style
    # (random subset of the non-test region) so the KNOWN set stays dense.
    if masks is not None:
        test_gap = masks
    else:
        test_gap = make_gaps(obs, N, T, args.gap_len, args.n_gaps, rng)
    train_gap = make_gaps(obs & ~test_gap, N, T, args.gap_len, args.n_gaps * 3, rng)
    known = obs & ~test_gap & ~train_gap          # what any feature/prior may see

    # z-score on KNOWN cells only (no leakage from either gap set)
    mu, sigma = tv[known].mean(), tv[known].std() + 1e-8
    z = (tv - mu) / sigma
    known_z = np.where(known, z, 0.0)             # gap cells -> 0, fixed below by cf
    cf = carry_forward(z, known)                  # persistence (last known before t)
    known_z = np.where(known, z, cf)              # best estimate available per cell

    def to_ug(zv):
        return inv(zv * sigma + mu)

    z_t = torch.tensor(z, dtype=torch.float)
    knownz_t = torch.tensor(known_z, dtype=torch.float)
    cf_t = torch.tensor(cf, dtype=torch.float)
    known_t = torch.tensor(known)
    test_t = torch.tensor(test_gap)
    train_t = torch.tensor(train_gap)

    # lag features from known_z (leak-free): lag1, lag24, (+lag48, lag168 weekly)
    lag1 = torch.zeros_like(knownz_t); lag1[1:] = knownz_t[:-1]
    lag24 = torch.zeros_like(knownz_t); lag24[24:] = knownz_t[:-24]
    lag48 = torch.zeros_like(knownz_t); lag48[48:] = knownz_t[:-48]
    lag168 = torch.zeros_like(knownz_t); lag168[168:] = knownz_t[:-168]

    # ---- SPATIAL IDW prior kernel (optionally terrain-aware, from eval_inductive) --
    dmat = np.hypot(x_m[:, None] - x_m[None, :], y_m[:, None] - y_m[None, :])
    w = 1.0 / (dmat + 1.0)
    if args.elev_kernel:
        e = np.asarray(elev, dtype=np.float64)
        w = w * np.exp(-np.abs(e[:, None] - e[None, :]) / max(args.elev_kernel_h, 1.0))
    np.fill_diagonal(w, 0.0)
    Wmat = torch.tensor(w, dtype=torch.float)

    def spatial_idw(t, vis_bool):
        vis = vis_bool.float()
        num = Wmat @ (z_t[t] * vis)
        den = (Wmat @ vis).clamp(min=1e-9)
        return num / den

    # RK-elev spatial prior (--rk-spatial): OLS elevation-drift trend over the
    # visible KNOWN nodes + IDW of the residual. The terrain-mean win (beats plain
    # IDW ~10% inductively) brought into the temporal task's spatial component.
    elev_tt = torch.tensor(np.asarray(elev, dtype=np.float64), dtype=torch.float)

    def spatial_rk(t, vis_bool):
        idx = vis_bool.nonzero().squeeze(1)
        if idx.numel() < 3:
            return spatial_idw(t, vis_bool)
        zc = z_t[t][idx]
        A = torch.stack([torch.ones_like(elev_tt[idx]), elev_tt[idx]], dim=1)
        beta_rk = torch.linalg.lstsq(A, zc.unsqueeze(1)).solution.squeeze(1)
        resid = zc - A @ beta_rk
        r_full = torch.zeros(N).scatter(0, idx, resid)
        r_idw = (Wmat @ r_full) / (Wmat @ vis_bool.float()).clamp(min=1e-9)
        return (beta_rk[0] + beta_rk[1] * elev_tt) + r_idw

    # ---- temperature gate input (per-node standardized temp) --------------------
    temp_z_t = None
    if args.temp_gate:
        if not has_temp:
            raise SystemExit(f"--temp-gate needs temperature; none for {args.city}")
        twn = temp_wide.to_numpy(dtype=np.float64)
        fin = np.isfinite(twn)
        ts = (twn - twn[fin].mean()) / (twn[fin].std() + 1e-6)
        ts[~fin] = 0.0
        temp_z_t = torch.tensor(ts, dtype=torch.float)

    # ---- AOD feature ------------------------------------------------------------
    aod_z_t = None
    if args.aod_feature:
        aod_wide, has_aod = bg.load_aod(bg.GROUP_CONFIG[args.sensor_set].get("aod_csv"),
                                        list(ids), pm.index)
        if not has_aod:
            raise SystemExit(f"--aod-feature needs AOD; none for {args.city}")
        aw = aod_wide.to_numpy(dtype=np.float64)
        fin = np.isfinite(aw)
        az = (aw - aw[fin].mean()) / (aw[fin].std() + 1e-6)
        az[~fin] = 0.0
        aod_z_t = torch.tensor(az, dtype=torch.float)

    # node features: [pm(masked), (persistence, lag1, lag24 if temporal), (aod)]
    use_temporal = args.temporal
    long_lags = use_temporal and args.long_lags
    node_in = 1 + (3 if use_temporal else 0) + (2 if long_lags else 0) + int(args.aod_feature)

    def node_features(t, hide_mask):
        x = knownz_t[t].reshape(-1, 1).clone()   # start from best-known estimate
        x[hide_mask, 0] = UNKNOWN                 # targets fed as unknown
        cols = [x]
        if use_temporal:
            cols += [cf_t[t].reshape(-1, 1), lag1[t].reshape(-1, 1), lag24[t].reshape(-1, 1)]
        if long_lags:
            cols += [lag48[t].reshape(-1, 1), lag168[t].reshape(-1, 1)]
        if aod_z_t is not None:
            cols.append(aod_z_t[t].reshape(-1, 1))
        return torch.cat(cols, dim=1)

    delev = edge_delev if args.elev_gate else torch.zeros_like(edge_delev)
    model = GraPhyNet(node_in=node_in, edge_in=edge_attr_t.shape[-1], hidden=args.hidden,
                      layers=args.layers, use_convection=not args.no_convection,
                      use_local=not args.no_local)
    torch.nn.init.zeros_(model.head.weight); torch.nn.init.zeros_(model.head.bias)
    # learnable spatio-temporal prior blend b = sigmoid(beta); init 0 -> 50/50
    beta = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam(list(model.parameters()) + [beta], lr=args.lr)
    loss_fn = (torch.nn.SmoothL1Loss(beta=args.huber_beta) if args.loss == "huber"
               else torch.nn.MSELoss())

    def prior(t, vis_bool):
        sp = spatial_rk(t, vis_bool) if args.rk_spatial else spatial_idw(t, vis_bool)
        if not use_temporal:
            return sp
        b = torch.sigmoid(beta)
        return b * cf_t[t] + (1.0 - b) * sp

    # hours that actually contain train-gap targets (skip empty ones)
    train_hours = np.where(train_gap.any(axis=1))[0]

    for epoch in range(args.epochs):
        model.train()
        batch = rng.choice(train_hours, size=min(args.steps, len(train_hours)),
                           replace=False)
        opt.zero_grad(); losses = []
        for t in batch:
            targets = np.where(train_gap[t])[0]
            if len(targets) == 0:
                continue
            vis = known_t[t].clone()                       # only KNOWN cells visible
            x = node_features(int(t), targets)
            out = model(x, edge_index, edge_weight, edge_attr_t[t], delev,
                        None if temp_z_t is None else temp_z_t[t])[:, 0]
            pred = prior(int(t), vis) + out
            sl = loss_fn(pred[targets], z_t[t][targets])
            if args.corr_reg > 0:
                sl = sl + args.corr_reg * out[targets].pow(2).mean()
            losses.append(sl)
        if not losses:
            continue
        torch.stack(losses).mean().backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

    # -------- eval on TEST gap cells --------------------------------------------
    model.eval(); trues, preds, persist_p, stk_p = [], [], [], []
    test_hours = np.where(test_gap.any(axis=1))[0]
    with torch.no_grad():
        for t in test_hours:
            targets = np.where(test_gap[t])[0]
            vis = known_t[t].clone()
            x = node_features(int(t), targets)
            out = model(x, edge_index, edge_weight, edge_attr_t[t], delev,
                        None if temp_z_t is None else temp_z_t[t])[:, 0]
            pr = prior(int(t), vis)                             # ST-kriging (no GNN)
            pred = pr + out
            for node in targets:
                trues.append(to_ug(z[t, node]))
                preds.append(to_ug(pred[node].item()))
                persist_p.append(to_ug(cf[t, node]))           # persistence floor
                stk_p.append(to_ug(pr[node].item()))           # spatiotemporal-kriging
    trues, preds, persist_p, stk_p = map(np.array, (trues, preds, persist_p, stk_p))
    m = metrics(trues, preds)
    m_persist = metrics(trues, persist_p)
    m_stk = metrics(trues, stk_p)                              # the HONEST baseline
    return m, m_persist, float(torch.sigmoid(beta)), m_stk


def main():
    ap = argparse.ArgumentParser(description="Long-gap temporal imputation eval.")
    ap.add_argument("--city", default="pittsburgh")
    ap.add_argument("--sensor-set", default="urban")
    ap.add_argument("--wind", choices=["era5", "hrrr", "zero"], default="hrrr")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--kfold", type=int, default=0,
                    help="if >0, run proper K-fold CV: partition the fixed-length gap "
                         "windows into K disjoint TEST folds (every clean window held "
                         "out exactly once) instead of Monte-Carlo random gaps. "
                         "Overrides --seeds/--n-gaps for TEST selection.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--gap-len", type=int, default=24, help="length of each test gap (hours)")
    ap.add_argument("--n-gaps", type=int, default=6, help="test gaps per sensor")
    ap.add_argument("--transform", choices=["log", "linear", "sqrt"], default="sqrt")
    ap.add_argument("--loss", choices=["mse", "huber"], default="huber")
    ap.add_argument("--huber-beta", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--corr-reg", type=float, default=0.6)
    ap.add_argument("--no-convection", action="store_true")
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--temporal", action="store_true",
                    help="add temporal memory: persistence prior blend + lag features")
    ap.add_argument("--long-lags", action="store_true",
                    help="add lag48 + lag168 (weekly) node features on top of --temporal. "
                         "Targets long-gap regimes where carry-forward persistence goes "
                         "stale (esp. sparse terrain / SLC).")
    ap.add_argument("--temp-gate", action="store_true")
    ap.add_argument("--aod-feature", action="store_true")
    ap.add_argument("--elev-gate", action="store_true")
    ap.add_argument("--elev-kernel", action="store_true")
    ap.add_argument("--elev-kernel-h", type=float, default=150.0)
    ap.add_argument("--rk-spatial", action="store_true",
                    help="use the RK-elev (elevation-drift regression-kriging) spatial "
                         "prior instead of plain spatial IDW. Combines the terrain-mean "
                         "win with temporal memory; the prior-only prediction is reported "
                         "as the ST-kriging baseline (the honest space+time floor to beat).")
    args = ap.parse_args()

    bg.use_city(args.city); bg.SENSOR_SET = args.sensor_set; bg.EPA_CORRECT = True
    tr.WIND_SOURCE = args.wind; tr.STRICT_INPUTS = (args.wind != "zero"); tr.USE_CACHE = False
    print(f"[temporal] city={args.city} gap_len={args.gap_len} n_gaps={args.n_gaps} "
          f"temporal={args.temporal} temp_gate={args.temp_gate} aod={args.aod_feature}")
    graph = tr.build_static_graph()

    ours, pers, blends, stks = [], [], [], []
    if args.kfold and args.kfold > 1:
        (ids, pm, observed, *_rest) = graph
        obs = observed.to_numpy()
        N, T = len(ids), len(pm)
        folds = kfold_gaps(obs, N, T, args.gap_len, args.kfold)
        units = "folds"
        labels = list(range(args.kfold))
        for k in labels:
            ncells = int(folds[k].sum())
            m, mp, b, mstk = run_seed(k, graph, args, masks=folds[k])
            ours.append(m); pers.append(mp); blends.append(b); stks.append(mstk)
            print(f"  fold {k}/{args.kfold}: OURS mae={m['mae']:.3f}  ST-krig mae={mstk['mae']:.3f}  "
                  f"persist mae={mp['mae']:.3f}  (n={m['n']}, test_cells={ncells})  b={b:.2f}")
    else:
        labels = [int(s) for s in args.seeds.split(",")]
        units = "seeds"
        for s in labels:
            m, mp, b, mstk = run_seed(s, graph, args)
            ours.append(m); pers.append(mp); blends.append(b); stks.append(mstk)
            print(f"  seed {s}: OURS mae={m['mae']:.3f}  ST-krig mae={mstk['mae']:.3f}  "
                  f"persist mae={mp['mae']:.3f}  (n={m['n']})  prior_blend b={b:.2f}")

    def agg(rows, k):
        v = np.array([r[k] for r in rows]); return v.mean(), v.std()
    cv = f"{args.kfold}-fold CV" if (args.kfold and args.kfold > 1) else f"MC {units}={labels}"
    print("\n" + "=" * 68)
    print(f"TEMPORAL GAP-FILL  city={args.city}  gap={args.gap_len}h  "
          f"rk_spatial={args.rk_spatial}  [{cv}]")
    print("=" * 68)
    for name, rows in [("OURS (GraPhyNet)", ours),
                       ("ST-kriging baseline", stks),
                       ("Persistence floor", pers)]:
        mae_m, mae_s = agg(rows, "mae"); r2_m, _ = agg(rows, "r2")
        print(f"{name:22s} MAE={mae_m:.3f}±{mae_s:.3f}  R2={r2_m:.3f}")
    om, _ = agg(ours, "mae"); sm, _ = agg(stks, "mae")
    print(f"OURS vs ST-kriging: {(om/sm - 1)*100:+.1f}%   "
          f"(<0 = GNN beats the strong space+time baseline)")
    print(f"mean learned prior blend b (persistence weight) = {np.mean(blends):.2f}")


if __name__ == "__main__":
    main()
