"""
Masked AOD spatial-gradient loss  (SPIN's mechanism, arXiv 2511.16013)
=============================================================================
Satellite AOD as a TRAINING CONSTRAINT, not an input feature. Our prior finding
was "AOD-as-node-feature is neutral/redundant" -- SPIN found the same and instead
uses the satellite AOD *spatial gradient* to shape the prediction field: where AOD
is available, the model's spatial gradient of predicted PM should agree (in
direction and relative magnitude) with the AOD gradient. That single constraint
took SPIN 25% below IGNNK on complex terrain.

We implement it as a masked, scale-free gradient-matching penalty over graph edges:
for every edge (i,j) whose BOTH endpoints have a valid AOD retrieval this hour,
standardise the pred-gradient and the AOD-gradient across the valid edges and
penalise their mismatch. Standardising makes it scale-free -> it constrains the
SHAPE of the field (spatial correlation with AOD), never the absolute PM level,
so it composes cleanly with the MSE/hybrid-prior objective.

    L_aod(t) = mean_over_valid_edges ( zscore(dpred) - zscore(daod) )^2

Inactive (returns 0) when fewer than `min_edges` edges have both endpoints valid,
which is most hours (night / cloud / no overpass). AOD is a daily covariate, so the
loss fires only on days with a cloud-free retrieval -- expected to be SPARSE on the
SLC winter stress case (snow + cloud confound MAIAC).

self-test:  .venv/bin/python scripts/aod_loss.py
"""
from __future__ import annotations
import torch


def _zscore(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (v - v.mean()) / (v.std() + eps)


def aod_gradient_loss(pred_z: torch.Tensor,
                      aod_vec: torch.Tensor,
                      aod_mask: torch.Tensor,
                      edge_index: torch.Tensor,
                      min_edges: int = 4) -> torch.Tensor:
    """Masked, scale-free AOD spatial-gradient matching penalty.

    pred_z     [N]     model prediction this hour (any linear PM units; z-space fine)
    aod_vec    [N]     AOD@550 per node this hour (NaN/0 where absent; masked out)
    aod_mask   [N]bool True where this node has a valid AOD retrieval this hour
    edge_index [2,E]   graph edges (same node indexing as pred_z)
    -> scalar loss (0.0 if too few valid edges; keeps gradient graph on pred_z)
    """
    i, j = edge_index[0], edge_index[1]
    both = aod_mask[i] & aod_mask[j]
    if int(both.sum()) < min_edges:
        return pred_z.sum() * 0.0                      # 0, but keeps autograd happy
    dpred = pred_z[i][both] - pred_z[j][both]          # [Ev] pred spatial gradient
    daod = aod_vec[i][both] - aod_vec[j][both]         # [Ev] AOD spatial gradient
    if daod.std() < 1e-6:                              # AOD flat here -> no signal
        return pred_z.sum() * 0.0
    return ((_zscore(dpred) - _zscore(daod)) ** 2).mean()


# ---------------------------------------------------------------------------
# self-test: a field that follows AOD gets ~0 loss; an anti-correlated field is
# maximally penalised; a masked-out region contributes nothing.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 12
    ei = torch.tensor([[k for k in range(N - 1)] + [k + 1 for k in range(N - 1)],
                       [k + 1 for k in range(N - 1)] + [k for k in range(N - 1)]])
    aod = torch.linspace(0.1, 0.9, N)                  # smooth spatial AOD ramp
    mask = torch.ones(N, dtype=torch.bool)

    pred_follow = 5.0 + 30.0 * aod + 0.01 * torch.randn(N)   # tracks AOD (diff scale)
    pred_anti = 5.0 - 30.0 * aod                             # opposes AOD
    pred_noise = torch.randn(N)                              # unrelated

    lf = aod_gradient_loss(pred_follow, aod, mask, ei).item()
    la = aod_gradient_loss(pred_anti, aod, mask, ei).item()
    ln = aod_gradient_loss(pred_noise, aod, mask, ei).item()
    print(f"follows AOD  (diff scale): loss={lf:.4f}   (want ~0 -> scale-free)")
    print(f"anti-correlated with AOD : loss={la:.4f}   (want ~4 -> max mismatch)")
    print(f"unrelated noise          : loss={ln:.4f}   (want ~2 -> chance)")

    # masking: hide half the nodes -> loss still defined on the visible half only
    half = mask.clone(); half[N // 2:] = False
    lh = aod_gradient_loss(pred_anti, aod, half, ei).item()
    print(f"anti, half masked        : loss={lh:.4f}   (still ~4 on visible edges)")

    # gradient actually flows to pred (so it can train)
    p = pred_noise.clone().requires_grad_(True)
    aod_gradient_loss(p, aod, mask, ei).backward()
    print(f"grad flows to pred: |grad|={p.grad.abs().sum().item():.4f}  (want > 0)")
    assert lf < 0.2 and la > 3.0 and p.grad.abs().sum() > 0, "self-test FAILED"
    print("self-test PASSED")
