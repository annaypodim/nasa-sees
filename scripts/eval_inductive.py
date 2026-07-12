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
from src.graph import preprocessing as pp
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
    # train-only sensors: always in the train pool, never held out as val/test targets.
    # Use for lower-confidence nodes (e.g. single-channel-recovered sensors) that add
    # value as interpolation neighbours but are too noisy to score fairly as targets --
    # a denser reference network, which is what real imputation deployments have.
    forced_train = np.array(
        [i for i, sid in enumerate(ids) if int(sid) in getattr(args, "_train_only", set())],
        dtype=int,
    )
    if forced_train.size:
        pool = np.array([i for i in range(N) if i not in set(forced_train.tolist())], dtype=int)
        perm = rng.permutation(pool.size)
        n_test = max(1, round(N * 9 / 41))
        n_val = max(1, round(N * 4 / 41))
        test_idx = np.sort(pool[perm[:n_test]])
        val_idx = np.sort(pool[perm[n_test:n_test + n_val]])
        train_idx = np.sort(np.concatenate([pool[perm[n_test + n_val:]], forced_train]))
    else:
        train_idx, val_idx, test_idx = split_nodes(N, n_val, n_test, rng)
    hidden_input = np.concatenate([val_idx, test_idx])  # never fed their PM

    values = pm.to_numpy(dtype=np.float64)  # [T, N] ug/m3, NaN already filled to 0
    obs = observed.to_numpy()

    # target transform, then z-score, using ONLY observed TRAIN-sensor cells (no
    # leakage). The transform choice is decisive for LINEAR MAE:
    #   log    -> optimizes geometric error, UNDERPREDICTS peaks -> inflates MAE.
    #   linear -> optimizes the reported metric BUT heavy-tailed PM + MSE explodes
    #             (seeds diverge). Needs Huber + grad-clip to be stable.
    #   sqrt   -> the middle ground: compresses the tail enough to train stably in
    #             a near-linear space, so it optimizes ~MAE without diverging.
    base = np.clip(values, 0, None)
    if args.transform == "linear":
        tv = base
        inv = lambda r: np.clip(r, 0, None)
    elif args.transform == "sqrt":
        tv = np.sqrt(base)
        inv = lambda r: np.clip(r, 0, None) ** 2
    else:  # log
        tv = np.log1p(base)
        inv = lambda r: np.clip(np.expm1(r), 0, None)
    train_cells = tv[:, train_idx][obs[:, train_idx]]
    mu, sigma = train_cells.mean(), train_cells.std() + 1e-8
    z = (tv - mu) / sigma
    z_t = torch.tensor(z, dtype=torch.float)
    obs_t = torch.tensor(obs)

    # ---- TEMPERATURE GATE input (per-node standardized surface temp) ------------
    # Surface temp is a never-masked covariate (knowable everywhere, like elevation),
    # so it feeds the per-node gate at ALL nodes with NO target leakage. Standardize
    # over the whole window with a single mean/std (temperature.py's spec: s carries
    # seasonal+diurnal+spatial variation, centred at gate==1). Missing-sensor NaNs ->
    # 0 -> gate==1 (inert) for that node. temp_wide is per-SENSOR here (Pittsburgh),
    # so s genuinely varies in space -> the gate can discriminate locations.
    temp_z_t = None
    if args.temp_gate:
        if not has_temp:
            raise SystemExit("--temp-gate needs temperature data but has_temp=False "
                             f"for city={args.city}")
        tw = temp_wide.to_numpy(dtype=np.float64)          # [T, N] °C
        finite = np.isfinite(tw)
        tmu, tsd = tw[finite].mean(), tw[finite].std() + 1e-6
        ts = (tw - tmu) / tsd
        ts[~finite] = 0.0                                  # missing -> neutral gate
        temp_z_t = torch.tensor(ts, dtype=torch.float)

    def to_ug(zval):  # invert z-score + transform back to ug/m3
        return inv(zval * sigma + mu)

    # ---- IDW RESIDUAL PRIOR (added 2026-07-09) ----------------------------------
    # The GNN loses to plain inverse-distance-weighting because its diffusion core is
    # a Laplacian (difference) operator, not an averaging one. Fix: compute the IDW
    # interpolation in the SAME z-space we predict, and make the model predict a
    # CORRECTION on top of it: pred = idw_prior + model(). If the correction is 0 we
    # recover IDW exactly, so the model structurally cannot do worse than IDW.
    # Wmat[i,j] = 1/(dist+1) with zero diagonal; per-timestep IDW uses only the nodes
    # that are OBSERVED AND VISIBLE (not masked) that hour -> no target leakage.
    Wmat = None
    dmat_t = offdiag = log_bw = None
    cov_t = None          # kriging covariance matrix exp(-d/range)  [N,N]
    den_full = None       # per-node total IDW mass (all nodes visible) -> conf norm
    need_prior = (args.idw_prior or args.idw_feature or args.kriging_prior
                  or args.idw_geom_features)
    if need_prior:
        dmat = np.hypot(x_m[:, None] - x_m[None, :], y_m[:, None] - y_m[None, :])
        # KRIGING PRIOR (Tier-1 #2): a fitted-kernel simple-kriging BLUP replaces
        # the 1/d IDW as the prior workhorse. In z-space the field mean is ~0 so
        # simple kriging (no unbiasedness Lagrange term) applies:
        #   base = C_av (C_vv + nugget I)^-1 z_vis
        # with an exponential covariance C = exp(-d/range). Unlike IDW this
        # accounts for the spatial COVARIANCE structure and DOWNWEIGHTS clustered
        # (redundant) sensors, the classic reason kriging beats IDW. Solved
        # per-timestep on the visible set (N tiny -> cheap). Range/nugget are
        # hyperparameters (variogram scale); nugget also regularizes the solve.
        if args.kriging_prior:
            cov_t = torch.tensor(np.exp(-dmat / max(args.kriging_range, 1.0)),
                                 dtype=torch.float)
        if args.learn_bw:
            # LEARNABLE kernel exp(-d/bw): bw (metres) is a trained parameter so the
            # prior adapts to the network's real spatial correlation length instead
            # of a hard-coded 1/d. Init bw ~2 km. Added to the optimizer below.
            dmat_t = torch.tensor(dmat, dtype=torch.float)
            offdiag = 1.0 - torch.eye(len(dmat), dtype=torch.float)
            log_bw = torch.nn.Parameter(torch.tensor(np.log(2000.0), dtype=torch.float))
        else:
            w = 1.0 / (dmat + 1.0)
            if args.elev_kernel:
                # TERRAIN-AWARE PRIOR: downweight cross-terrain sensor pairs by their
                # elevation separation, so the IDW prior interpolates preferentially
                # WITHIN an airmass/valley rather than across a ridge. Factor
                # exp(-|Δelev|/h), h = vertical decay length (m). Flat ground ->
                # Δelev≈0 -> factor≈1 (inert), so it self-disables off terrain just
                # like the elevation gate -- but it reshapes the PRIOR (which does most
                # of the work) instead of only the small learned correction.
                e = np.asarray(elev, dtype=np.float64)
                de = np.abs(e[:, None] - e[None, :])
                w = w * np.exp(-de / max(args.elev_kernel_h, 1.0))
            np.fill_diagonal(w, 0.0)
            Wmat = torch.tensor(w, dtype=torch.float)
        if args.idw_geom_features and Wmat is not None:
            den_full = Wmat.sum(1)                    # total IDW mass per node

    use_prior = args.idw_prior or args.kriging_prior  # add the prior base to output

    def _W():
        """Current IDW weight matrix (constant unless the bandwidth is learned)."""
        if log_bw is not None:
            return torch.exp(-dmat_t / torch.exp(log_bw).clamp(min=1.0)) * offdiag
        return Wmat

    def kriging_base(t, visible_bool):
        """Simple-kriging BLUP in z-space over the visible set: base = C_av C_vv^-1 z."""
        vis_i = torch.where(visible_bool)[0]
        if len(vis_i) == 0:
            return torch.zeros(len(elev))
        Cvv = cov_t[vis_i][:, vis_i].clone()
        Cvv.diagonal().add_(args.kriging_nugget)      # nugget: noise + regularizer
        Cav = cov_t[:, vis_i]                         # [N, n_vis]
        wv = torch.linalg.solve(Cvv, z_t[t][vis_i])   # C_vv^-1 z_vis
        return Cav @ wv

    def prior_stats(t, visible_bool):
        """Return (base, geom) for the visible set, both in z-space.
          base -- prior interpolation (kriging if --kriging-prior, else IDW).
          geom -- None, or [N,2] = (conf, disp) IDW-confidence features:
                  conf = visible IDW mass / total mass  (low -> extrapolating into
                         a sparse region where the prior is least trustworthy);
                  disp = weighted std of the visible neighbour values (local
                         heterogeneity -> where a smooth prior is most likely wrong).
                  These tell the learned correction WHERE the prior needs fixing."""
        vis = visible_bool.float()
        W = _W()
        idw_val = geom = None
        if W is not None:
            num = W @ (z_t[t] * vis)
            den = W @ vis
            den_c = den.clamp(min=1e-9)
            idw_val = num / den_c
            if args.idw_geom_features:
                conf = den / den_full.clamp(min=1e-9)
                ex2 = (W @ (z_t[t].pow(2) * vis)) / den_c
                disp = (ex2 - idw_val.pow(2)).clamp(min=0).sqrt()
                geom = torch.stack([conf, disp], dim=1)          # [N,2]
        base = kriging_base(t, visible_bool) if args.kriging_prior else idw_val
        return base, geom

    # never-masked node feature channels. col 0 = PM2.5 (maskable). Optional extras:
    #   elev  -- DEM elevation (spatially varying, knowable anywhere)
    #   idw   -- the IDW prior value itself (added 2026-07-09): lets the learned
    #            correction CONDITION on the base it is correcting, instead of
    #            inferring it. Appended per-step in the loops (it depends on masking).
    elev_col = None
    if args.elev_feature:
        e = np.asarray(elev, dtype=np.float64)
        elev_col = torch.tensor((e - e.mean()) / (e.std() + 1e-6),
                                dtype=torch.float).reshape(-1, 1)

    # ---- AOD FEATURE (satellite column aerosol, never-masked) -------------------
    # MAIAC 1 km AOD sampled per sensor: a GENUINELY SPATIAL covariate (spatial std
    # ~0.021 on a 0.17 mean, ~12%), unlike common-mode temp/ERA5. Knowable at ALL
    # nodes (satellite), so no leakage feeding it at held-out test nodes. It's daily
    # + cloud-gappy (~33% coverage) so it's a FEATURE (not a gate): standardize over
    # finite cells, missing -> 0 (field-mean/neutral). Time-varying -> appended per t.
    aod_z_t = None
    if args.aod_feature:
        aod_csv = bg.GROUP_CONFIG[args.sensor_set].get("aod_csv")
        aod_wide, has_aod = bg.load_aod(aod_csv, list(ids), pm.index)
        if not has_aod:
            raise SystemExit(f"--aod-feature needs AOD data but none for city={args.city}")
        aw = aod_wide.to_numpy(dtype=np.float64)          # [T, N]
        fin = np.isfinite(aw)
        if args.aod_anomaly:
            # SPATIAL ANOMALY: subtract each hour's city-mean (over present sensors)
            # so the feature is "dustier/cleaner than the metro right now" -- pure
            # discriminative spatial signal, dropping the common-mode daily level
            # that carries no WHERE information for interpolation.
            row = np.where(fin, aw, np.nan)
            rowmean = np.nanmean(row, axis=1, keepdims=True)
            aw = aw - rowmean
        amu, asd = aw[fin].mean(), aw[fin].std() + 1e-6
        az = (aw - amu) / asd
        az[~fin] = 0.0                                     # missing -> neutral
        aod_z_t = torch.tensor(az, dtype=torch.float)

    node_in = (1 + int(args.elev_feature) + int(args.aod_feature)
               + int(args.idw_feature) + 2 * int(args.idw_geom_features))

    def node_features(t):
        x = z_t[t].reshape(-1, 1).clone()
        x[hidden_input, 0] = UNKNOWN          # val+test never reveal their PM
        cols = [x]
        if elev_col is not None:
            cols.append(elev_col)
        if aod_z_t is not None:
            cols.append(aod_z_t[t].reshape(-1, 1))
        return torch.cat(cols, dim=1) if len(cols) > 1 else x

    delev = edge_delev if args.elev_gate else torch.zeros_like(edge_delev)

    model = GraPhyNet(node_in=node_in, edge_in=edge_attr_t.shape[-1],
                      hidden=args.hidden, layers=args.layers,
                      use_convection=not args.no_convection,
                      use_local=not args.no_local)
    # RESIDUAL-KRIGING STABILITY (added 2026-07-09): with the IDW prior the head
    # outputs a CORRECTION added to IDW. If it starts nonzero / grows unchecked it
    # can drag held-out predictions AWAY from IDW and lose (seeds diverged this way).
    # Zero-init the head so training starts EXACTLY at IDW, and (below) penalize the
    # correction's magnitude so it stays a genuine small residual -> robustly >= IDW.
    if use_prior:
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
    params = list(model.parameters())
    if log_bw is not None:
        params.append(log_bw)              # train the prior bandwidth jointly
    opt = torch.optim.Adam(params, lr=args.lr)
    # Huber (smooth-L1) bounds the gradient from peak outliers -> stable in near-
    # linear target space where MSE diverges; beta is in z-units (~1 std).
    # loss on the (transformed, z-scored) target. l1 = MEDIAN regression: the
    # MAE-optimal predictor is the conditional median, so L1 targets the reported
    # metric directly (huber/mse target a mean-ish quantity). Now that the prior
    # is a stable base and the correction is regularized small, L1 no longer
    # diverges the way raw-linear MSE did.
    loss_fn = {"huber": torch.nn.SmoothL1Loss(beta=args.huber_beta),
               "l1": torch.nn.L1Loss(),
               "mse": torch.nn.MSELoss()}[args.loss]

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
            # IGNNK-style implicit augmentation: instead of always hiding the same
            # fraction, sample it per step over [mask_frac, mask_frac_hi]. Harder
            # (higher-fraction) steps force predict-many-from-few extrapolation that
            # matches a sparse held-out sensor; easier steps keep dense supervision.
            # --mask-frac-hi unset -> mf == mask_frac (original fixed behavior).
            mf = (args.mask_frac if args.mask_frac_hi is None
                  else rng.uniform(args.mask_frac, args.mask_frac_hi))
            n_hide = max(1, int(len(known_train) * mf))
            targets = rng.choice(known_train, size=n_hide, replace=False)
            x = node_features(int(t))
            x[targets, 0] = UNKNOWN
            # visible = observed AND not masked (targets + always-hidden val/test).
            # The IDW prior is built from exactly what the model can see -> no leakage.
            visible = obs_t[t].clone()
            visible[hidden_input] = False
            visible[targets] = False
            base, geom = prior_stats(int(t), visible)
            if args.idw_feature:
                x = torch.cat([x, base.reshape(-1, 1)], dim=1)   # base as a feature
            if geom is not None:
                x = torch.cat([x, geom], dim=1)                  # IDW-confidence feats
            out = model(x, edge_index, edge_weight, edge_attr_t[t], delev,
                        None if temp_z_t is None else temp_z_t[t])[:, 0]
            pred = (base if use_prior else 0.0) + out
            step_loss = loss_fn(pred[targets], z_t[t][targets])
            # keep the correction small so we can't stray far below the prior floor
            if use_prior and args.corr_reg > 0:
                step_loss = step_loss + args.corr_reg * out[targets].pow(2).mean()
            losses.append(step_loss)
        if not losses:
            continue
        torch.stack(losses).mean().backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

    # -------- inductive eval: predict TEST sensors over every hour ----------
    # One forward pass per hour also yields VAL-sensor predictions (val is held
    # out identically to test), which --recal uses to fit a monotonic predicted->
    # true map. That corrects the transform's inversion bias (predicting in sqrt/
    # log z-space then squaring back is convex -> biased) WITHOUT touching test
    # truth: val sensors never enter training and are disjoint from test.
    model.eval()
    trues, preds = [], []
    val_trues, val_preds = [], []
    with torch.no_grad():
        for t in range(T):
            if obs[t, test_idx].sum() == 0 and obs[t, val_idx].sum() == 0:
                continue
            x = node_features(t)                  # test+val PM already masked
            visible = obs_t[t].clone()
            visible[hidden_input] = False          # only train sensors are visible
            base, geom = prior_stats(t, visible)
            if args.idw_feature:
                x = torch.cat([x, base.reshape(-1, 1)], dim=1)
            if geom is not None:
                x = torch.cat([x, geom], dim=1)
            out = model(x, edge_index, edge_weight, edge_attr_t[t], delev,
                        None if temp_z_t is None else temp_z_t[t])[:, 0]
            pred_z = (base if use_prior else 0.0) + out
            for node in test_idx:
                if obs[t, node]:
                    trues.append(to_ug(z[t, node]))
                    preds.append(to_ug(pred_z[node].item()))
            if args.recal:
                for node in val_idx:
                    if obs[t, node]:
                        val_trues.append(to_ug(z[t, node]))
                        val_preds.append(to_ug(pred_z[node].item()))
    trues, preds = np.array(trues), np.array(preds)
    if args.recal and len(val_preds) > 10:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.array(val_preds), np.array(val_trues))
        preds = iso.predict(preds)
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
    ap.add_argument("--mask-frac-hi", type=float, default=None,
                    help="IGNNK-style random masking: sample the per-step hidden "
                         "fraction uniformly in [mask-frac, mask-frac-hi] as implicit "
                         "augmentation. Unset -> fixed mask-frac (original behavior).")
    ap.add_argument("--transform", choices=["log", "linear", "sqrt"], default="sqrt",
                    help="target transform before z-scoring. sqrt (default) trains "
                         "stably near-linear; log underpredicts peaks; linear needs "
                         "huber+clip or it diverges.")
    ap.add_argument("--loss", choices=["mse", "huber", "l1"], default="huber",
                    help="huber (default) bounds peak-outlier gradients -> stable; "
                         "l1 = median regression, targets MAE directly")
    ap.add_argument("--huber-beta", type=float, default=1.0,
                    help="Huber transition point in z-units (~std). default 1.0")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="max grad-norm (0=off). default 1.0 for stability")
    ap.add_argument("--knn", type=int, default=None,
                    help="override K nearest-neighbour edges per node (bg.K, default 5). "
                         "IDW weighs ALL sensors; a denser graph lets the GNN see more.")
    ap.add_argument("--no-convection", action="store_true",
                    help="ablate the convection module (diffusion[+local] only)")
    ap.add_argument("--no-local", action="store_true",
                    help="ablate the local module (diffusion[+convection] only)")
    ap.add_argument("--idw-prior", action="store_true",
                    help="predict IDW interpolation + a learned GNN correction "
                         "(residual kriging) -> cannot do worse than IDW by construction")
    ap.add_argument("--epa-correct", action="store_true",
                    help="apply the EPA/Barkjohn PurpleAir correction (needs data "
                         "fetched with --extra-fields); closes the absolute gap to 2.38")
    ap.add_argument("--corr-reg", type=float, default=0.1,
                    help="L2 penalty on the IDW-prior correction magnitude (z-units). "
                         "Keeps the residual small so we stay >= IDW. 0=off.")
    ap.add_argument("--idw-feature", action="store_true",
                    help="also feed the IDW prior value as a never-masked node feature "
                         "so the correction can condition on the base it corrects")
    ap.add_argument("--idw-geom-features", action="store_true",
                    help="Tier1 #1: add 2 IDW-confidence node features (visible-mass "
                         "fraction + neighbour dispersion) telling the correction WHERE "
                         "the prior is uncertain (sparse/heterogeneous regions)")
    ap.add_argument("--kriging-prior", action="store_true",
                    help="Tier1 #2: replace the 1/d IDW prior with a simple-kriging "
                         "BLUP (exp covariance) that downweights clustered sensors")
    ap.add_argument("--kriging-range", type=float, default=3000.0,
                    help="exponential covariance range (m) for --kriging-prior")
    ap.add_argument("--kriging-nugget", type=float, default=0.1,
                    help="kriging nugget (obs noise / solve regularizer), z-units^2")
    ap.add_argument("--recal", action="store_true",
                    help="Tier1 #3b: fit a monotonic (isotonic) predicted->true map on "
                         "the held-out VAL sensors and apply it to test -> removes the "
                         "transform-inversion bias. Leak-safe (val disjoint from test)")
    ap.add_argument("--despike", action="store_true",
                    help="temporal MAD despike of the raw PM series in preprocessing "
                         "(isolated single-hour spikes -> not observed). Data QA.")
    ap.add_argument("--flatline", action="store_true",
                    help="mask stuck-sensor runs (>=24h identical nonzero) -> not "
                         "observed. Data QA (catches jammed lasers despike misses).")
    ap.add_argument("--spatial-qa", action="store_true",
                    help="mask per-hour cross-sensor gross outliers (robust-z vs the "
                         "network median) -> not observed. Data QA (multi-hour faults).")
    ap.add_argument("--learn-bw", action="store_true",
                    help="learnable Gaussian prior kernel exp(-d/bw) instead of fixed "
                         "1/d; bw (spatial correlation length) trained jointly")
    ap.add_argument("--elev-feature", action="store_true",
                    help="add never-masked DEM elevation channel (ablation; GraPhy base=off)")
    ap.add_argument("--elev-gate", action="store_true",
                    help="enable the elevation gate (ablation; GraPhy base=off)")
    ap.add_argument("--elev-kernel", action="store_true",
                    help="terrain-aware IDW PRIOR: multiply the 1/d kernel by "
                         "exp(-|Δelev|/h) so the prior weights within-airmass pairs")
    ap.add_argument("--elev-kernel-h", type=float, default=150.0,
                    help="vertical decay length (m) for --elev-kernel (default 150)")
    ap.add_argument("--temp-gate", action="store_true",
                    help="enable the per-node temperature gate (needs per-sensor temp; "
                         "Pittsburgh only so far). Ablation; GraPhy base=off")
    ap.add_argument("--aod-feature", action="store_true",
                    help="add never-masked satellite AOD channel (spatially varying "
                         "covariate; needs aod_csv, Pittsburgh only). Ablation")
    ap.add_argument("--aod-anomaly", action="store_true",
                    help="feed AOD as a per-hour SPATIAL ANOMALY (minus city-mean) "
                         "instead of the raw level -> pure discriminative signal")
    ap.add_argument("--train-only-ids", default="",
                    help="comma-list of sensor IDs to keep ALWAYS in the train pool "
                         "(never val/test targets) -- lower-confidence neighbours, e.g. "
                         "single-channel-recovered sensors")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="DENSITY SWEEP: randomly keep only this many sensors (edges "
                         "rebuilt on the subset) to trace MAE vs density. None = all.")
    ap.add_argument("--subsample-seed", type=int, default=0,
                    help="seed for the --max-nodes sensor subset (vary to average over "
                         "different random subsets at a fixed density)")
    args = ap.parse_args()
    args._train_only = {int(s) for s in args.train_only_ids.split(",") if s.strip()}

    bg.use_city(args.city)
    tr.SUBSAMPLE_N = args.max_nodes
    tr.SUBSAMPLE_SEED = args.subsample_seed
    bg.SENSOR_SET = args.sensor_set
    bg.EPA_CORRECT = args.epa_correct
    pp.DESPIKE = args.despike
    pp.FLATLINE = args.flatline
    pp.SPATIAL_QA = args.spatial_qa
    if args.knn is not None:
        bg.K = args.knn
    tr.WIND_SOURCE = args.wind
    tr.STRICT_INPUTS = (args.wind != "zero")
    tr.USE_CACHE = False  # Fresno has no processed cache; build fresh from raw

    print(f"[repro] city={args.city} wind={args.wind} hidden={args.hidden} "
          f"layers={args.layers} epochs={args.epochs} elev_feat={args.elev_feature} "
          f"elev_gate={args.elev_gate} temp_gate={args.temp_gate}")
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
