"""
graphy_faithful.py -- a FAITHFUL, standalone reimplementation of GraPhy
(arXiv 2506.06917, "Graph-Based Physics-Guided Urban PM2.5 Air Quality
Imputation with Constrained Monitoring Data").

WHY THIS FILE EXISTS (vs src/model/model.py)
    The repo's existing GraPhyNet (model.py) is NOT the published architecture:
    it was refactored into `IDW_estimate + small_regularized_correction`, plus
    elevation/temperature gates and an outer `h + fused` residual. Good numbers,
    but it never tested GraPhy AS PUBLISHED. This file rebuilds the paper's
    architecture exactly -- three parallel physics GNN modules combined by a
    dynamic softmax fusion, and nothing else -- so we can measure whether the
    real design reaches GraPhy's reported 2.38 ug/m3 on Fresno.

    One GraPhy LAYER (paper Eq. block):
        x_D = GNN_D(x)          diffusion   (Laplacian high-pass)
        x_C = GNN_C(x, e)       convection  (wind message passing)
        x_L = GNN_L(x)          local       (source/sink)
        (w_D, w_C, w_L) = Fusion(x_D, x_C, x_L)     per-node softmax, sums to 1
        x_out = w_D*x_D + w_C*x_C + w_L*x_L          <-- fusion IS the update.
    NOTE: the paper's layer update is the fused sum, with NO extra outer residual
    (that `h + fused` residual is a model.py addition; we do not carry it here).

DISAMBIGUATIONS (paper is silent at the sub-module level; choices flagged for
review, echoed in the final report):
  * sigma in diffusion  -> sigmoid (paper's stated reasonable default).
  * lambda_max           -> computed ONCE per graph from the static Laplacian
                            (edge weights are distance-only and time-invariant),
                            cached as a buffer; not re-estimated per forward.
  * diffusion "MLP first" -> a 2-layer node MLP producing the node representation,
                            then the paper's l (.) sigma(L_D X W). W is a bias-free
                            Linear; l is a learned per-OUTPUT-feature vector.
  * local "standard graph convolution" -> degree-NORMALISED mean aggregation
                            (GCN convention). With 1/d weights in METRES the raw
                            edge weights are ~1e-3, so an UN-normalised conv would
                            make the local branch ~1e3x smaller than the others
                            and numerically vanish under fusion; normalising the
                            aggregation keeps F at O(node-feature) scale. The
                            paper's M = I + D scaling is then applied on top of
                            that F, exactly as written (X = M.F).
  * convection ResMessage / Update MLP widths -> hidden == layer width; the edge
                            feature is ADDED (residually) to the neighbour node
                            feature before the message MLP (paper Fig. 6 wording
                            "edge feature is added to node features"), and the
                            update MLP sees [message, node+message] per the text.
  * per-module internal MLP hidden size -> the layer width (`dim`).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.elevation import ElevationGate   # reused verbatim from the IDW+corr model


def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    """2-layer perceptron -- the shared building block of the message modules."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# 1. DIFFUSION  -- GNN_D(X) = l (.) sigma(L_D . NodeMLP(X) . W)
#    A high-pass (difference) operator on the rescaled normalised Laplacian.
#    L_D depends only on the static 1/d adjacency, so it is precomputed once
#    (see GraPhyFaithful.__init__) and passed in as `L_D`.
# ---------------------------------------------------------------------------
class DiffusionModule(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.node_mlp = _mlp(in_dim, dim, dim)          # node representation first
        self.W = nn.Linear(dim, dim, bias=False)        # the paper's weight matrix W
        self.l = nn.Parameter(torch.ones(dim))          # per-feature diffusion coeff

    def forward(self, x, L_D):
        h = self.node_mlp(x)                            # [N, dim]
        return self.l * torch.sigmoid(L_D @ self.W(h))  # l (.) sigma(L_D h W)


# ---------------------------------------------------------------------------
# 2. CONVECTION -- wind transport by learned message passing (paper Sec 3.2 /
#    Fig. 6). Edge input e_{j,i} = [dist, w_A=cos(wind_dir - bearing), w_v=speed]
#    fed as THREE SEPARATE numbers (built upstream in train.build_static_graph);
#    the edge MLP's output becomes the NEXT layer's edge feature.
# ---------------------------------------------------------------------------
class ConvectionModule(nn.Module):
    def __init__(self, in_dim: int, edge_in: int, dim: int):
        super().__init__()
        self.node_mlp = _mlp(in_dim, dim, dim)          # transform x_i, x_j
        self.edge_mlp = _mlp(edge_in, dim, dim)         # raw [dist,w_A,w_v] -> dim
        self.msg_mlp  = _mlp(dim, dim, dim)             # ResMessage
        self.upd_mlp  = _mlp(2 * dim, dim, dim)         # Update

    def forward(self, x, edge_index, edge_feat, edge_gate=None):
        src, dst = edge_index                           # message flows src(j) -> dst(i)
        h = self.node_mlp(x)                            # [N, dim]
        e = self.edge_mlp(edge_feat)                    # [E, dim]  -> next-layer edge feat
        # ResMessage: edge feature ADDED (residually) to the neighbour node feature,
        # then through the message MLP (paper: "edge feature is added to node features").
        m = self.msg_mlp(h[src] + e)                    # [E, dim]
        # optional elevation gate: dampen the wind message across height (model.py parity)
        if edge_gate is not None:
            m = m * edge_gate[:, None]
        agg = torch.zeros_like(h).index_add_(0, dst, m) # [N, dim]  sum incoming messages
        # Update: concat the message feature with (node feature + message feature).
        out = self.upd_mlp(torch.cat([agg, h + agg], dim=1))
        return out, e                                   # e = evolved edge feature


# ---------------------------------------------------------------------------
# 3. LOCAL  -- source/sink term. NodeMLP -> graph conv -> normalise by M=I+D.
#    X = M . F,  M_{ii} = 1 + sum_j e_{j,i}  (incoming edge-weight degree + self).
# ---------------------------------------------------------------------------
class LocalModule(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.node_mlp = _mlp(in_dim, dim, dim)          # local transform in isolation
        self.conv_mlp = _mlp(dim, dim, dim)             # what each node hands neighbours

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index
        h = self.node_mlp(x)                            # [N, dim]
        # incoming edge-weight degree D_ii = sum_j e_{j,i}
        deg = torch.zeros(x.size(0), device=x.device).index_add_(0, dst, edge_weight)
        # degree-normalised graph conv (GCN convention) -> F at O(h) scale
        msg = self.conv_mlp(h)[src] * edge_weight[:, None]          # [E, dim]
        f = torch.zeros_like(h).index_add_(0, dst, msg)            # [N, dim]
        f = f / deg.clamp(min=1e-12)[:, None]
        M = 1.0 + deg                                              # M_ii = 1 + sum_j e_{j,i}
        return M[:, None] * f                                      # X = M . F


# ---------------------------------------------------------------------------
# 4. DYNAMIC SOFTMAX FUSION -- per-node learned softmax over the three module
#    outputs. Recomputed every forward (that is what "dynamic" means).
# ---------------------------------------------------------------------------
class Fusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = _mlp(3 * dim, 3, dim)               # -> 3 raw scores per node

    def forward(self, x_D, x_C, x_L):
        z = torch.cat([x_D, x_C, x_L], dim=1)           # [N, 3*dim]
        w = torch.softmax(self.gate(z), dim=1)          # [N, 3], rows sum to 1
        stacked = torch.stack([x_D, x_C, x_L], dim=1)   # [N, 3, dim]
        out = (w[:, :, None] * stacked).sum(dim=1)      # [N, dim]
        return out, w


# ---------------------------------------------------------------------------
# ONE GraPhy LAYER
# ---------------------------------------------------------------------------
class GraPhyLayer(nn.Module):
    def __init__(self, in_dim: int, edge_in: int, dim: int, use_elev_gate: bool = False):
        super().__init__()
        self.diffusion  = DiffusionModule(in_dim, dim)
        self.convection = ConvectionModule(in_dim, edge_in, dim)
        self.local      = LocalModule(in_dim, dim)
        self.fusion     = Fusion(dim)
        # OPTIONAL elevation gate (NOT part of faithful GraPhy; imported from the
        # IDW+corr model). Per-layer, own learned uphill/downhill height scales.
        # Gates diffusion (via a rebuilt gated Laplacian) + convection (message);
        # local + the fusion update stay ungated -- same wiring as src/model/model.py.
        self.elev_gate = ElevationGate() if use_elev_gate else None

    def forward(self, x, L_D, edge_index, edge_weight, edge_feat, gate_ctx=None):
        edge_gate = None
        if self.elev_gate is not None and gate_ctx is not None:
            ei_s, ew_s, delev_s, N, B = gate_ctx           # single-graph structures
            g = self.elev_gate(delev_s)                    # [E] per-edge gate in (0,1]
            L_D_single = build_L_D(ei_s, ew_s * g, N)      # diffusion reads the gate here
            L_D = torch.block_diag(*([L_D_single] * B))    # override the passed L_D
            edge_gate = g.repeat(B)                        # convection reads it as a message gate
        x_D = self.diffusion(x, L_D)
        x_C, e_next = self.convection(x, edge_index, edge_feat, edge_gate)
        x_L = self.local(x, edge_index, edge_weight)       # local: ungated
        fused, w = self.fusion(x_D, x_C, x_L)              # fusion IS the layer update
        return fused, e_next, w


# ---------------------------------------------------------------------------
# THE MODEL: stack K layers, read out the final node feature -> one PM2.5 value.
# x^(0) = raw (masked) sensor measurement per node; layer 1 modules do the
# encoding (no separate encoder, matching the paper's x^(k-1) -> modules flow).
# ---------------------------------------------------------------------------
class GraPhyFaithful(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 512, layers: int = 5,
                 use_elev_gate: bool = False):
        super().__init__()
        dims_in = [node_in] + [hidden] * (layers - 1)   # layer 1 takes node_in, rest hidden
        edges_in = [edge_in] + [hidden] * (layers - 1)  # convection edge feat evolves to `hidden`
        self.layers = nn.ModuleList(
            GraPhyLayer(dims_in[k], edges_in[k], hidden, use_elev_gate) for k in range(layers)
        )
        self.head = nn.Linear(hidden, 1)                # final node feature -> PM2.5

    def forward(self, x, L_D, edge_index, edge_weight, edge_feat, return_weights=False,
                gate_ctx=None):
        # gate_ctx=(edge_index_single, edge_weight_single, edge_delev, N, B) enables the
        # per-layer elevation gate (rebuilds L_D from gated weights). None -> ungated,
        # and the passed static L_D is used verbatim (identical to the base faithful model).
        e = edge_feat
        ws = []
        for layer in self.layers:
            x, e, w = layer(x, L_D, edge_index, edge_weight, e, gate_ctx)
            ws.append(w)
        out = self.head(x)                              # [N, 1]
        if return_weights:
            return out, ws
        return out


# ---------------------------------------------------------------------------
# Rescaled normalised Laplacian L_D = 2L/lambda_max - I, L = I - D^-1/2 A D^-1/2.
# Static (distance-only adjacency) -> computed ONCE per graph.
# ---------------------------------------------------------------------------
def build_L_D(edge_index: torch.Tensor, edge_weight: torch.Tensor, N: int) -> torch.Tensor:
    eye = torch.eye(N)
    A = torch.zeros(N, N)
    A[edge_index[0], edge_index[1]] = edge_weight
    A = torch.maximum(A, A.t())                         # symmetrise (undirected diffusion)
    d = A.sum(1).clamp(min=1e-12).pow(-0.5)
    L = eye - d[:, None] * A * d[None, :]               # normalised Laplacian, eigs in [0,2]
    lam = torch.linalg.eigvalsh(L).max()
    return 2.0 * L / lam - eye
