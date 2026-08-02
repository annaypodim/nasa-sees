"""
eval_graphy_faithful.py -- train + inductively evaluate the FAITHFUL GraPhy
rebuild (src/model/graphy_faithful.py) under the SAME kriging protocol as
scripts/eval_inductive.py, so its MAE is directly comparable to both the repo's
current best (~3.05 on fresno_dense_abc) and GraPhy's reported 2.38.

WHAT IS FAITHFUL HERE (vs eval_inductive.py)
  * MODEL: the published 3-module + dynamic-softmax-fusion architecture, with NO
    IDW prior / correction-regularisation / kriging / elevation-temp gates.
  * MASKING: exactly ONE train sensor masked per training example (paper's spec),
    NOT the IGNNK random-fraction masking.
  * LOSS: plain MSE in raw ug/m3 (linear z-score only for input conditioning,
    inverted linearly before the loss -- NO log/sqrt transform, NO Huber, NO clip).
  * OPTIM: Adam lr 1e-4, betas (0.9,0.999), batch 32.
  * CAPACITY: --config large (hidden 512, 5 layers) or small (hidden 128, 3 layers).

REUSED FROM THE HARNESS (data plumbing only): train.build_static_graph (graph +
real HRRR wind edge features [dist, w_A=cos, w_v]), and eval_inductive's
split_nodes / idw_baseline / metrics.

USAGE
  .venv/bin/python scripts/eval_graphy_faithful.py --city fresno_dense_abc \
      --wind hrrr --despike --spatial-qa --config large --seeds 0,1,2,3,4,5,6,7
"""
from __future__ import annotations

import argparse
import copy
import time
import warnings

warnings.filterwarnings("ignore")

import glob
from pathlib import Path

import numpy as np
import torch

from src.graph import build_graph2 as bg
from src.graph import preprocessing as pp
from src.model import train as tr
from src.model.graphy_faithful import GraPhyFaithful, build_L_D
from scripts.eval_inductive import split_nodes, idw_baseline, metrics
from scripts.aod_loss import aod_gradient_loss

UNKNOWN = 0.0  # z-space placeholder fed for masked (unmonitored) nodes

CONFIGS = {
    "large": dict(hidden=512, layers=5),   # GraPhy "large"
    "small": dict(hidden=128, layers=3),   # GraPhy "small" (~30x smaller, ~10% worse)
}


def rk_elev_baseline(values, obs, x_m, y_m, elev, train_idx, test_idx, power=2.0):
    """Regression-kriging with ELEVATION as external drift (universal-kriging style).

    The literature terrain-aware baseline (universal/regression kriging with a DEM
    covariate). Per hour: OLS-fit v ~ 1 + elev on the OBSERVED train sensors to get a
    linear elevation TREND, then IDW-interpolate the REGRESSION RESIDUAL to each test
    node and add the trend back:  pred = (b0 + b1*elev_test) + IDW(residual).
    Reduces to plain IDW when b1->0, so it isolates the elevation-covariate lever.
    Identical split/scoring to idw_baseline() -> directly comparable to that number.
    """
    elev = np.asarray(elev, dtype=np.float64)
    d = np.hypot(x_m[test_idx, None] - x_m[None, train_idx],
                 y_m[test_idx, None] - y_m[None, train_idx])   # [n_test, n_train]
    w_all = 1.0 / np.maximum(d, 1.0) ** power
    preds, trues = [], []
    for t in range(values.shape[0]):
        tr_obs = obs[t, train_idx]
        if tr_obs.sum() < 3:                     # need >=3 pts for a stable trend
            continue
        tr_nodes = train_idx[tr_obs]
        v = values[t, tr_nodes]
        ez = elev[tr_nodes]
        # OLS elevation drift: [1, elev] -> (b0, b1); lstsq is stable if elev varies
        A = np.column_stack([np.ones_like(ez), ez])
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        resid = v - A @ beta                     # regression residual on train
        w = w_all[:, tr_obs]
        r_idw = (w * resid[None, :]).sum(1) / w.sum(1)          # [n_test]
        trend_test = beta[0] + beta[1] * elev[test_idx]
        est = np.maximum(trend_test + r_idw, 0.0)              # [n_test], PM>=0
        for k, node in enumerate(test_idx):
            if obs[t, node]:
                preds.append(est[k]); trues.append(values[t, node])
    return np.array(trues), np.array(preds)


def run_seed(seed, graph, args):
    (ids, pm, observed, edge_index, edge_weight, edge_attr_t, edge_delev, elev,
     x_m, y_m, has_wind, temp_wide, has_temp) = graph
    N, T = len(ids), len(pm)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # AOD spatial-gradient loss (SPIN): satellite AOD as a training constraint, not
    # an input. Tensors aligned to (T, N) by main(); inert when weight=0 / no data.
    aod_w = getattr(args, "aod_weight", 0.0)
    aod_t = getattr(args, "_aod_t", None)
    aod_mask_t = getattr(args, "_aod_mask_t", None)

    # ---- inductive split (identical to eval_inductive): 28:4:9 scaled to N -----
    n_test = max(1, round(N * 9 / 41))
    n_val = max(1, round(N * 4 / 41))
    train_idx, val_idx, test_idx = split_nodes(N, n_val, n_test, rng)
    hidden_input = np.concatenate([val_idx, test_idx])   # val+test PM never fed

    values = np.clip(pm.to_numpy(dtype=np.float64), 0, None)  # [T, N] ug/m3, >=0
    obs = observed.to_numpy()

    # ---- FAST PATH: baselines only (IDW + elevation regression-kriging), no train --
    if getattr(args, "baselines_only", False):
        it, ip = idw_baseline(values, obs, x_m, y_m, train_idx, test_idx)
        rt, rp = rk_elev_baseline(values, obs, x_m, y_m, elev, train_idx, test_idx)
        m_idw, m_rk = metrics(it, ip), metrics(rt, rp)
        m_none = dict(mae=float("nan"), rmse=float("nan"), r2=float("nan"),
                      corr=float("nan"), n=0, best_step=0, best_val=float("nan"))
        return (m_none, m_idw, (len(train_idx), len(val_idx), len(test_idx)),
                np.zeros(3), False, [], m_rk)

    # ---- LINEAR standardisation ONLY (mean/std over observed TRAIN cells) -------
    # This is a linear rescale, so MSE in z-space == MSE in ug/m3 up to a constant;
    # we still invert it and compute the loss in ug/m3 to be literal. Crucially it
    # is NOT the log/sqrt target transform (those optimise a different error).
    train_cells = values[:, train_idx][obs[:, train_idx]]
    mu, sigma = train_cells.mean(), train_cells.std() + 1e-8
    z = (values - mu) / sigma
    z_t = torch.tensor(z, dtype=torch.float)
    obs_t = torch.tensor(obs)

    # rescaled normalised Laplacian: static adjacency -> computed ONCE per graph.
    L_D = build_L_D(edge_index, edge_weight, N)

    # ---- HYBRID IDW PRIOR (Track 2): pred = idw_prior + faithful correction ------
    # z-space inverse-distance interpolation over the VISIBLE observed nodes each
    # hour; the faithful model then predicts a residual on top. If the residual is 0
    # we recover IDW exactly, so the hybrid structurally cannot underperform IDW.
    # Optional terrain-aware kernel exp(-|Delta elev|/h) (inert on flat ground).
    # --rk-prior selects the ELEVATION-DRIFT regression-kriging prior instead of IDW
    # (residual IDW uses a plain distance kernel; the trend carries the terrain).
    use_prior = args.idw_prior or args.rk_prior
    elev_t = torch.tensor(np.asarray(elev, dtype=np.float64), dtype=torch.float)
    Wmat = None
    if use_prior:
        dmat = np.hypot(x_m[:, None] - x_m[None, :], y_m[:, None] - y_m[None, :])
        # RK prior matches the standalone RK-elev baseline kernel (power-2, max(d,1))
        # so the corrector actually stands on the 3.65 floor; IDW prior keeps 1/(d+1).
        w = (1.0 / np.maximum(dmat, 1.0) ** 2) if args.rk_prior else (1.0 / (dmat + 1.0))
        if args.idw_prior_elev and not args.rk_prior:
            e = np.asarray(elev, dtype=np.float64)
            # SIGNED vertical decay. W[i,j] = weight of SOURCE j for TARGET i, so
            # the physically-relevant gap is de = z_target - z_source = e_i - e_j.
            # In a valley inversion, pollutant pools DOWNHILL: a target should draw
            # readily from sources at/below it (de>=0, gentle h_down) and steeply
            # discount sources ABOVE it (de<0, steep h_up). h_up/h_down default to
            # --idw-prior-h, so the unset case reproduces the old symmetric kernel.
            de = e[:, None] - e[None, :]                 # signed [N,N]
            h_up = max(args.idw_h_up if args.idw_h_up else args.idw_prior_h, 1.0)
            h_down = max(args.idw_h_down if args.idw_h_down else args.idw_prior_h, 1.0)
            h = np.where(de >= 0.0, h_down, h_up)
            w = w * np.exp(-np.abs(de) / h)
            # INVERSION CAP: sensors on the SAME side of the cap z* share an air
            # mass and stay coupled; a straddling pair (one below, one above the
            # inversion) is in decoupled air and gets cut. Smooth same-side term
            # via two sigmoids of softness --idw-cap-w. Off unless --idw-cap set.
            if args.idw_cap is not None:
                zc = float(args.idw_cap)
                cw = max(float(args.idw_cap_w), 1e-6)
                zmax = np.maximum(e[:, None], e[None, :])
                zmin = np.minimum(e[:, None], e[None, :])
                sig = lambda t: 1.0 / (1.0 + np.exp(-t))
                same_side = sig((zc - zmax) / cw) + sig((zmin - zc) / cw)
                w = w * np.clip(same_side, 0.0, 1.0)
        np.fill_diagonal(w, 0.0)
        Wmat = torch.tensor(w, dtype=torch.float)

    def idw_prior_z(t, hidden_bool):
        """z-space IDW over nodes observed at t AND visible (not in hidden_bool)."""
        vis = (obs_t[t] & ~hidden_bool).float()          # [N]
        num = Wmat @ (z_t[t] * vis)
        den = (Wmat @ vis).clamp(min=1e-9)
        return num / den                                 # [N] in z-space

    def rk_prior_z(t, hidden_bool):
        """z-space REGRESSION-KRIGING prior: OLS elevation-drift trend over the
        visible observed nodes + distance-IDW of the residual. Beats IDW on terrain
        (SLC RK-elev 3.65 vs IDW 4.06). Falls back to IDW if <3 visible obs."""
        vis = (obs_t[t] & ~hidden_bool)                  # bool [N]
        idx = vis.nonzero().squeeze(1)
        if idx.numel() < 3:
            return idw_prior_z(t, hidden_bool)
        zc = z_t[t][idx]                                 # [k] observed z
        A = torch.stack([torch.ones_like(elev_t[idx]), elev_t[idx]], dim=1)  # [k,2]
        beta = torch.linalg.lstsq(A, zc.unsqueeze(1)).solution.squeeze(1)    # [2]
        resid = zc - A @ beta                            # [k] residual (z-units)
        r_full = torch.zeros(N).scatter(0, idx, resid)   # [N], 0 off the visible set
        visf = vis.float()
        r_idw = (Wmat @ r_full) / (Wmat @ visf).clamp(min=1e-9)  # [N]
        trend = beta[0] + beta[1] * elev_t               # [N] elevation drift
        return trend + r_idw                             # [N] z-space RK-elev prior

    prior_z = rk_prior_z if args.rk_prior else idw_prior_z

    # val+test are never-visible inputs -> the base hidden set for the IDW prior.
    hidden_base_bool = torch.zeros(N, dtype=torch.bool)
    hidden_base_bool[hidden_input] = True

    cfg = CONFIGS[args.config]
    hidden = args.hidden if args.hidden else cfg["hidden"]   # capacity overrides
    layers = args.layers if args.layers else cfg["layers"]
    # gate mode: terrain (new learned gates) > elev (old two-scalar gate) > none.
    gate_mode = "terrain" if args.terrain_gate else ("elev" if args.elev_gate else "none")
    use_gate = gate_mode != "none"
    model = GraPhyFaithful(node_in=1, edge_in=edge_attr_t.shape[-1],
                           hidden=hidden, layers=layers,
                           gate_mode=gate_mode, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()   # plain MSE, in ug/m3 (see below)

    # terrain-mode drainage gate reads per-edge wind alignment w_A = edge_attr col 1
    # (cos(wind_dir - bearing)); only meaningful with real wind. When off, the gate
    # degrades to a Δelev-only drainage gate (wind_align=None).
    WIND_ALIGN_COL = 1
    directional = (gate_mode == "terrain" and has_wind
                   and edge_attr_t.shape[-1] > WIND_ALIGN_COL)

    # elevation-gate context: single-graph structures the gated layers rebuild L_D
    # from, plus the batch's per-edge wind alignment for the drainage gate. `ef` is
    # the current (possibly batched) edge-feature tensor; None -> gate off.
    def gctx(B, ef=None):
        if not use_gate:
            return None
        wa = ef[:, WIND_ALIGN_COL] if (directional and ef is not None) else None
        return (edge_index, edge_weight, edge_delev, wa, N, B)

    def to_ug(zval):  # linear inverse of the standardisation
        return zval * sigma + mu

    def node_input(t, extra_mask=None):
        """z-scored PM per node with val/test (+ optional extra) set to UNKNOWN."""
        x = z_t[t].reshape(-1, 1).clone()
        x[hidden_input, 0] = UNKNOWN
        if extra_mask is not None:
            x[extra_mask, 0] = UNKNOWN
        return x

    # ---- TRAINING: one-sensor-at-a-time masking, MSE in ug/m3 ------------------
    # Each training EXAMPLE = (timestep t, one masked train sensor). A batch is
    # `--batch` such examples; we accumulate their ug/m3 squared errors and step.
    train_ts = np.arange(T)
    # precompute, per timestep, which train sensors are observed (valid mask targets)
    obs_train = {int(t): train_idx[obs[t, train_idx]] for t in train_ts}
    usable_ts = np.array([t for t in train_ts if len(obs_train[t]) >= 3])

    # BLOCK-DIAGONAL BATCHING: the graph topology is identical across the `batch`
    # examples (only node PM, the masked target, and the hour's wind edge features
    # differ), so we stack B independent copies into ONE disconnected graph of B*N
    # nodes and do a single forward. block-diag L_D + offset edge_index keep the
    # examples from talking to each other, giving the EXACT same per-example
    # semantics as a Python loop but ~B-fold faster. This is a pure speed change.
    E = edge_index.shape[1]
    ei_np = edge_index.numpy()

    # ---- VAL EARLY STOPPING (uses the reserved val sensors; leak-safe) ---------
    # The train masking task converges/overfits by ~2k steps while held-out test
    # MAE degrades, so we select the model by MAE on the VAL sensors (disjoint from
    # test) over a fixed random subset of hours, keep the best state, and stop if
    # val hasn't improved for `patience` checks. This is standard train/val/test
    # model selection -- more faithful to how GraPhy trains than a fixed step count.
    val_hours_pool = np.array([t for t in range(T) if obs[t, val_idx].sum() > 0])
    val_ts = (rng.choice(val_hours_pool, size=min(args.val_hours, len(val_hours_pool)),
                         replace=False) if len(val_hours_pool) else np.array([], int))

    def val_mae():
        model.eval()
        err = []
        with torch.no_grad():
            for t in val_ts:
                t = int(t)
                out = model(node_input(t), None if use_gate else L_D, edge_index,
                            edge_weight, edge_attr_t[t],
                            gate_ctx=gctx(1, edge_attr_t[t]))[:, 0]
                if use_prior:
                    out = out + prior_z(t, hidden_base_bool)
                for node in val_idx:
                    if obs[t, node]:
                        err.append(abs(to_ug(out[node].item()) - values[t, node]))
        model.train()
        return float(np.mean(err)) if err else float("inf")

    best_val, best_state, best_step, since_improve = float("inf"), None, -1, 0
    losses_hist = []
    val_hist = []           # (step, val_mae, running_best) for convergence plots
    diverged = False
    B = args.batch
    ew_b = edge_weight.repeat(B)
    # static block-diag L_D for the ungated path (constant -> build once). When the
    # elevation gate is on, each gated layer rebuilds L_D per step from gate params.
    L_D_b_static = None if use_gate else torch.block_diag(*([L_D] * B))
    for step in range(args.steps):
        model.train()
        opt.zero_grad()
        ts_batch = rng.choice(usable_ts, size=B, replace=True)
        xb = torch.zeros(B * N, 1)
        eib = np.empty((2, B * E), dtype=np.int64)
        efb = torch.empty(B * E, edge_attr_t.shape[-1])
        tgt_nodes = np.empty(B, dtype=np.int64)
        trues_ug = np.empty(B, dtype=np.float64)
        prior_b = torch.zeros(B * N) if use_prior else None
        for b, t in enumerate(ts_batch):
            t = int(t)
            target = int(rng.choice(obs_train[t]))    # mask exactly ONE train sensor
            xb[b * N:(b + 1) * N] = node_input(t, extra_mask=[target])
            eib[:, b * E:(b + 1) * E] = ei_np + b * N
            efb[b * E:(b + 1) * E] = edge_attr_t[t]
            tgt_nodes[b] = b * N + target
            trues_ug[b] = values[t, target]
            if use_prior:
                # the masked target joins val/test in the hidden set for THIS example
                hb = hidden_base_bool.clone(); hb[target] = True
                prior_b[b * N:(b + 1) * N] = prior_z(t, hb)
        corr = model(xb, L_D_b_static, torch.from_numpy(eib), ew_b, efb,
                     gate_ctx=gctx(B, efb))[:, 0]     # raw GNN correction (z-space)
        out = corr + prior_b if use_prior else corr   # residual on the IDW/RK floor
        pred = to_ug(out[torch.from_numpy(tgt_nodes)])
        true = torch.tensor(trues_ug, dtype=torch.float)
        loss = loss_fn(pred, true)                    # MSE in raw ug/m3
        # CORRECTION ANCHOR: penalise the GNN correction magnitude at the scored
        # nodes so the hybrid stays close to (>=) the prior floor. Small correction
        # over a strong prior is where signal pays; unregularised it wandered and
        # HURT (RK-prior 3.92 > RK-elev 3.65). Only meaningful with a prior.
        if use_prior and args.corr_reg > 0.0:
            loss = loss + args.corr_reg * (corr[torch.from_numpy(tgt_nodes)] ** 2).mean()
        # SPIN AOD spatial-gradient constraint: on each example's full node field,
        # the predicted PM gradient should match the satellite AOD gradient where a
        # retrieval exists. Scale-free -> shapes the field without touching PM level.
        if aod_w > 0.0 and aod_t is not None:
            al = out.new_zeros(())
            for b in range(B):
                tt = int(ts_batch[b])
                al = al + aod_gradient_loss(out[b * N:(b + 1) * N],
                                            aod_t[tt], aod_mask_t[tt], edge_index)
            loss = loss + aod_w * (al / B)
        if not torch.isfinite(loss):
            diverged = True
            print(f"    [seed {seed}] non-finite loss at step {step} -> DIVERGED")
            break
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            losses_hist.append((step, float(loss.detach())))
        # val model selection + early stop
        if len(val_ts) and (step % args.val_every == 0 or step == args.steps - 1):
            vm = val_mae()
            if vm < best_val - 1e-4:
                best_val, best_step, since_improve = vm, step, 0
                best_state = copy.deepcopy(model.state_dict())
                val_hist.append((step, vm, best_val))
            else:
                val_hist.append((step, vm, best_val))
                since_improve += 1
                if args.patience and since_improve >= args.patience:
                    break
            if getattr(args, "curve_out", None):
                star = " *" if vm <= best_val + 1e-4 else ""
                print(f"    [seed {seed}] step {step:6d}  val_mae={vm:.3f}  "
                      f"best={best_val:.3f}@{best_step}{star}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)   # restore best-val checkpoint

    # ---- INDUCTIVE EVAL: predict TEST sensors over every hour ------------------
    model.eval()
    trues, preds = [], []
    w_sum = np.zeros(3); w_cnt = 0        # mean fusion weight (w_D, w_C, w_L)
    with torch.no_grad():
        for t in range(T):
            if obs[t, test_idx].sum() == 0:
                continue
            x = node_input(t)             # only test/val masked; train all visible
            out, ws = model(x, None if use_gate else L_D, edge_index, edge_weight,
                            edge_attr_t[t], return_weights=True,
                            gate_ctx=gctx(1, edge_attr_t[t]))
            pred_z = out[:, 0]
            if use_prior:
                pred_z = pred_z + prior_z(t, hidden_base_bool)
            for node in test_idx:
                if obs[t, node]:
                    trues.append(values[t, node])
                    preds.append(to_ug(pred_z[node].item()))
            # average fusion weights over layers, at the TEST nodes we score
            for w in ws:
                w_sum += w[test_idx].mean(0).numpy(); w_cnt += 1
    trues, preds = np.array(trues), np.array(preds)
    m = metrics(trues, preds)
    it, ip = idw_baseline(values, obs, x_m, y_m, train_idx, test_idx)
    m_idw = metrics(it, ip)
    rt, rp = rk_elev_baseline(values, obs, x_m, y_m, elev, train_idx, test_idx)
    m_rk = metrics(rt, rp)
    fusion_w = (w_sum / max(w_cnt, 1))
    m["best_step"] = best_step; m["best_val"] = best_val
    if getattr(args, "curve_out", None) and val_hist:
        import os
        os.makedirs(args.curve_out, exist_ok=True)
        p = os.path.join(args.curve_out, f"val_curve_seed{seed}.csv")
        with open(p, "w") as fh:
            fh.write("step,val_mae,running_best\n")
            for s, v, b in val_hist:
                fh.write(f"{s},{v:.5f},{b:.5f}\n")
        print(f"    [seed {seed}] wrote {len(val_hist)} val points -> {p}", flush=True)
    return m, m_idw, (len(train_idx), len(val_idx), len(test_idx)), fusion_w, diverged, losses_hist, m_rk


def main():
    ap = argparse.ArgumentParser(description="Faithful GraPhy inductive kriging eval.")
    ap.add_argument("--city", default="fresno_dense_abc")
    ap.add_argument("--sensor-set", default="urban")
    ap.add_argument("--wind", choices=["era5", "hrrr", "zero"], default="hrrr")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--config", choices=list(CONFIGS), default="large")
    ap.add_argument("--steps", type=int, default=3000, help="optimizer steps")
    ap.add_argument("--curve-out", default=None,
                    help="if set, write per-seed val-MAE-vs-step CSV here (for convergence plots)")
    ap.add_argument("--batch", type=int, default=32, help="examples per step (paper: 32)")
    ap.add_argument("--lr", type=float, default=1e-4, help="Adam lr (paper: 1e-4)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam L2 weight decay. Combats the overfitting/collapse the pure "
                         "faithful model shows on small sparse nets (few train sensors) -> "
                         "the fix for it failing to beat IDW on Fresno. Try 1e-4..1e-3.")
    ap.add_argument("--hidden", type=int, default=None,
                    help="override the config hidden width (fewer params = less overfit)")
    ap.add_argument("--layers", type=int, default=None,
                    help="override the config layer count")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout inside the module MLPs (regularizer; 0=off)")
    ap.add_argument("--val-every", type=int, default=200, help="steps between val checks")
    ap.add_argument("--val-hours", type=int, default=400, help="hours sampled for val MAE")
    ap.add_argument("--patience", type=int, default=12,
                    help="stop if val MAE has not improved for this many checks (0=off)")
    ap.add_argument("--knn", type=int, default=None, help="override K nearest-neighbour edges")
    ap.add_argument("--despike", action="store_true", help="temporal MAD despike (data QA)")
    ap.add_argument("--spatial-qa", action="store_true", help="cross-sensor outlier mask (data QA)")
    ap.add_argument("--flatline", action="store_true", help="mask stuck-sensor runs (data QA)")
    ap.add_argument("--epa-correct", action="store_true",
                    help="apply the EPA/Barkjohn PurpleAir correction (needs cf_1+humidity "
                         "columns) -> tests the absolute-scale gap to GraPhy's 2.38")
    ap.add_argument("--elev-gate", action="store_true",
                    help="add the elevation gate from the IDW+corr model (per-layer, gates "
                         "diffusion+convection by signed Delev). NOT faithful GraPhy; only "
                         "meaningful on terrain (SLC). Needs coords with real altitude.")
    ap.add_argument("--terrain-gate", action="store_true",
                    help="LEARNED terrain gating (per-layer): a TerrainGate with a Delev-"
                         "dependent decay rate on diffusion + a directional DrainageGate on "
                         "convection (couples Delev sign with wind alignment). Supersedes "
                         "--elev-gate when both set. Terrain cities (SLC); needs real altitude.")
    ap.add_argument("--idw-prior", action="store_true",
                    help="HYBRID: add a z-space IDW interpolation prior to the model output "
                         "(pred = idw_prior + faithful_GraPhy_correction). The physics modules "
                         "become a residual corrector over the interpolation floor -> structurally "
                         "cannot do worse than IDW, and aims to beat both IDW and vanilla GraPhy.")
    ap.add_argument("--rk-prior", action="store_true",
                    help="HYBRID over the ELEVATION regression-kriging prior instead of "
                         "IDW (pred = RK-elev_prior + faithful correction). RK-elev beats "
                         "IDW ~10%% on terrain (SLC 3.65 vs 4.06), so this lowers the floor "
                         "the corrector builds on. Mutually exclusive with --idw-prior.")
    ap.add_argument("--idw-prior-elev", action="store_true",
                    help="make the IDW prior terrain-aware: weight *= exp(-|Delta elev|/h) so it "
                         "interpolates within a valley/airmass, not across ridges (SLC). Inert on flat.")
    ap.add_argument("--idw-prior-h", type=float, default=200.0,
                    help="vertical decay length h (m) for --idw-prior-elev (default 200)")
    ap.add_argument("--idw-h-up", type=float, default=None,
                    help="SIGNED kernel: decay length (m) for sources ABOVE the target "
                         "(steep in a valley inversion). Defaults to --idw-prior-h.")
    ap.add_argument("--idw-h-down", type=float, default=None,
                    help="SIGNED kernel: decay length (m) for sources AT/BELOW the target "
                         "(gentle: pollutant pools downhill). Defaults to --idw-prior-h.")
    ap.add_argument("--idw-cap", type=float, default=None, metavar="Z_STAR",
                    help="INVERSION CAP altitude z* (m): same-side pairs stay coupled, a "
                         "straddling (below/above z*) pair is cut. Off unless set. Requires "
                         "--idw-prior-elev.")
    ap.add_argument("--idw-cap-w", type=float, default=50.0,
                    help="softness (m) of the --idw-cap transition (default 50)")
    ap.add_argument("--baselines-only", action="store_true",
                    help="skip model training; report only the IDW and elevation "
                         "regression-kriging (RK-elev) baselines over all seeds. Fast "
                         "literature terrain-aware baseline on the identical splits.")
    ap.add_argument("--corr-reg", type=float, default=0.0,
                    help="L2 penalty (z-units^2) on the GNN correction magnitude at the "
                         "scored nodes, keeping the hybrid close to (>=) its prior floor. "
                         "Fixes the unregularised RK-prior wandering off 3.65 and HURTING. "
                         "Only meaningful with --idw-prior/--rk-prior. Try 0.05..0.5.")
    ap.add_argument("--aod-weight", type=float, default=0.0,
                    help="weight of the SPIN masked AOD spatial-gradient loss (0=off). "
                         "Uses satellite AOD as a training constraint (pred gradient should "
                         "match AOD gradient where retrieved), NOT as an input feature. "
                         "Needs data/<city>/aod/*.csv (scripts/fetch_aod.py). Try 0.1..1.0.")
    ap.add_argument("--track", action="store_true", help="print the training-loss trajectory")
    args = ap.parse_args()

    bg.use_city(args.city)
    bg.SENSOR_SET = args.sensor_set
    bg.EPA_CORRECT = args.epa_correct
    pp.DESPIKE = args.despike
    pp.SPATIAL_QA = args.spatial_qa
    pp.FLATLINE = args.flatline
    if args.knn is not None:
        bg.K = args.knn
    tr.WIND_SOURCE = args.wind
    tr.STRICT_INPUTS = (args.wind != "zero")
    tr.USE_CACHE = False

    cfg = CONFIGS[args.config]
    _cfg = CONFIGS[args.config]
    _h = args.hidden if args.hidden else _cfg["hidden"]
    _l = args.layers if args.layers else _cfg["layers"]
    print(f"[faithful] city={args.city} wind={args.wind} config={args.config} "
          f"(hidden={_h} layers={_l}) steps={args.steps} batch={args.batch} "
          f"lr={args.lr} wd={args.weight_decay} dropout={args.dropout}")
    graph = tr.build_static_graph()

    # AOD spatial-gradient loss: load satellite AOD, align to the graph (ids, hours),
    # stash (T,N) value+mask tensors on args for run_seed. Inert unless --aod-weight>0.
    if args.aod_weight > 0.0:
        ids0, pm0 = graph[0], graph[1]
        aod_files = sorted(glob.glob(str(Path(bg.DATA_DIR) / bg.CITY / "aod" / "*.csv")))
        if aod_files:
            aod_wide, has_aod = bg.load_aod(Path(aod_files[-1]),
                                            list(map(str, ids0)), pm0.index)
            anp = aod_wide.reindex(columns=ids0).to_numpy(dtype=np.float64)
            args._aod_t = torch.tensor(np.nan_to_num(anp), dtype=torch.float)
            args._aod_mask_t = torch.tensor(~np.isnan(anp))
            n_hr = int((~np.isnan(anp)).sum(1) .__ge__(2).sum())
            print(f"[aod] gradient loss ACTIVE  weight={args.aod_weight}  "
                  f"has_aod={has_aod}  hours with >=2 retrievals={n_hr}  "
                  f"file={Path(aod_files[-1]).name}")
        else:
            print(f"[aod] --aod-weight set but no {bg.CITY}/aod/*.csv -> loss inert")

    seeds = [int(s) for s in args.seeds.split(",")]
    ours, idws, rks, fusion_ws, diverged_any = [], [], [], [], []
    split = None
    t0 = time.time()
    for s in seeds:
        m, m_idw, split, fw, div, hist, m_rk = run_seed(s, graph, args)
        ours.append(m); idws.append(m_idw); rks.append(m_rk); fusion_ws.append(fw)
        diverged_any.append(div)
        tag = "  DIVERGED" if div else ""
        print(f"  seed {s}: OURS mae={m['mae']:.3f}  IDW mae={m_idw['mae']:.3f}  "
              f"RK(elev) mae={m_rk['mae']:.3f}   w(D,C,L)="
              f"({fw[0]:.2f},{fw[1]:.2f},{fw[2]:.2f}){tag}")
        if args.track and hist:
            print("    train-MSE(ug^2) curve: "
                  + "  ".join(f"{st}:{ls:.1f}" for st, ls in hist))

    def agg(rows, k):
        v = np.array([r[k] for r in rows])
        return v.mean(), v.std()

    fw_mean = np.mean(fusion_ws, axis=0)
    print("\n" + "=" * 72)
    print(f"FAITHFUL GraPhy  city={args.city}  config={args.config}  "
          f"split(tr/val/te)={split}  seeds={seeds}  ({time.time()-t0:.0f}s)")
    print("=" * 72)
    for name, rows in [("FAITHFUL GraPhy", ours), ("IDW baseline", idws),
                       ("RK-elev baseline", rks)]:
        mae_m, mae_s = agg(rows, "mae")
        rmse_m, _ = agg(rows, "rmse")
        r2_m, r2_s = agg(rows, "r2")
        corr_m, _ = agg(rows, "corr")
        print(f"{name:18s}  MAE={mae_m:.3f}+-{mae_s:.3f}  RMSE={rmse_m:.3f}  "
              f"R2={r2_m:.3f}+-{r2_s:.3f}  corr={corr_m:.3f}")
    print(f"{'repo best (ref)':18s}  MAE=3.054+-0.29 (IDW-residual scaffold, wind zero)")
    print(f"{'GraPhy (paper)':18s}  MAE=2.380   (Fresno, 41 sensors, Oct23-Jan24)")
    print(f"mean fusion weight  w_D(diffusion)={fw_mean[0]:.3f}  "
          f"w_C(convection)={fw_mean[1]:.3f}  w_L(local)={fw_mean[2]:.3f}")
    print(f"diverged seeds: {sum(diverged_any)}/{len(seeds)}"
          + ("  (plain MSE stable)" if not any(diverged_any) else "  <-- see above"))
    print("=" * 72)


if __name__ == "__main__":
    main()
