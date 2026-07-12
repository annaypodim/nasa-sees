"""
full model  (fusion + message passing)
=============================================================================
this is the last step: glue the three physics modules into one predictor and
run them over SEVERAL layers so information hops sensor -> sensor -> sensor.

one layer does this:
    d = diffusion(h)      how PM2.5 spreads high->low concentration
    c = convection(h)     how the wind carries it
    l = local(h)          what's created / removed right at the node
    fused = softmax-weighted mix of d, c, l   (FUSION, see below)
    h_out = h + fused                          (RESIDUAL, see below)

MESSAGE PASSING  (why we stack layers):
    one layer = a node only hears its direct neighbours. stack L layers and
    news travels L hops. we use FEWER layers than GraPhy's 5: with so few
    sensors there's little real signal diversity, so stacking too deep makes
    every node's embedding collapse to the same blur (oversmoothing). the
    `h = h + fused` RESIDUAL fights that -- each layer can only *nudge* a
    node, never erase what it already knew.

run the demo:   .venv/bin/python model.py
"""
from pathlib import Path

import torch
import torch.nn as nn

from src.model.diffusion import DiffusionModule, inverse_distance_weights
from src.model.convection import ConvectionModule
from src.model.local import LocalModule
from src.model.fusion import Fusion
from src.model.elevation import ElevationGate
from src.model.temperature import TemperatureGate


class GraPhyLayer(nn.Module):
    """one message-passing layer: three modules -> fuse -> residual.

    two MODULATORS, not extra modules:
      * ElevationGate  -- per-EDGE scalar from signed Δheight; dampens BOTH
        transports via diffusion's edge weight and convection's message.
      * TemperatureGate -- per-NODE scalar from surface temperature; scales BOTH
        transports' contributions at the destination node (cold/inversion < 1
        traps, hot/convective > 1 enhances).
    local (a node source/sink) and the outer residual stay ungated. see
    elevation.py and temperature.py.
    """
    def __init__(self, dim: int, edge_in: int, use_convection: bool = True,
                 use_local: bool = True):
        super().__init__()
        # MODULE TOGGLES (added 2026-07-09 for ablations): diffusion is always on
        # (it is the interpolation core); convection/local can be switched off to
        # isolate what actually helps vs the IDW baseline. Disabled modules are not
        # even constructed, so no dead parameters. Fusion sizes to the active count.
        self.use_convection = use_convection
        self.use_local = use_local
        self.diffusion  = DiffusionModule(dim, dim)
        self.convection = ConvectionModule(dim, edge_in, dim) if use_convection else None
        self.local      = LocalModule(dim, dim) if use_local else None
        n_modules = 1 + int(use_convection) + int(use_local)
        self.fusion     = Fusion(dim, n_modules=n_modules)
        self.elev_gate  = ElevationGate()   # shared height scale for both transport terms
        self.temp_gate  = TemperatureGate()  # per-node mixing scale for both transports
        # EDGE-GATE ANALOGS of the elevation gate (per-EDGE scalar from the signed
        # pairwise difference of a covariate; dampens BOTH transports when the pair
        # differs). Same exp form as ElevationGate; h_init set to each covariate's scale
        # (learnable). temp Δ ~ a few °C, AOD Δ ~ tenths. Fed the sensor-pair analog of
        # Δelevation: Δtemp[dst,src] / Δaod[dst,src], time-varying (recomputed per hour).
        self.temp_edge_gate = ElevationGate(h_init=5.0)
        self.aod_edge_gate  = ElevationGate(h_init=0.2, h_floor=0.02)

    def forward(self, h, edge_index, edge_weight, edge_attr, edge_delev=None,
                node_temp=None, edge_dtemp=None, edge_daod=None):
        # combine any active per-edge gates multiplicatively (each in (0,1]); an
        # independent ablation passes exactly one of Δelev / Δtemp / Δaod.
        gate = None
        if edge_delev is not None:
            gate = self.elev_gate(edge_delev)                      # [E]
        if edge_dtemp is not None:
            g = self.temp_edge_gate(edge_dtemp)
            gate = g if gate is None else gate * g
        if edge_daod is not None:
            g = self.aod_edge_gate(edge_daod)
            gate = g if gate is None else gate * g
        # diffusion reads the gate through the edge weight; convection through the message
        diff_weight = edge_weight if gate is None else edge_weight * gate
        d = self.diffusion(h, edge_index, diff_weight)
        # TEMPERATURE gate: per-node, scales each transport's contribution at the
        # destination node (None -> inert). local + the outer residual are ungated.
        tg = None if node_temp is None else self.temp_gate(node_temp)[:, None]  # [N, 1]
        if tg is not None:
            d = d * tg
        # active-module list, fixed order [diffusion, (convection), (local)]
        active = [d]
        if self.convection is not None:
            c = self.convection(h, edge_index, edge_attr, edge_gate=gate)
            if tg is not None:
                c = c * tg
            active.append(c)
        if self.local is not None:
            active.append(self.local(h, edge_index, edge_weight))  # local: ungated
        fused, _ = self.fusion(*active)   # returns (blend, weights); we want the blend
        return h + fused   # RESIDUAL: nudge, don't overwrite -> no oversmoothing


class GraPhyNet(nn.Module):
    """encode PM2.5 -> stack a few layers -> predict PM2.5 back."""
    def __init__(self, node_in: int, edge_in: int, hidden: int = 8, layers: int = 3,
                 use_convection: bool = True, use_local: bool = True):
        super().__init__()
        self.encoder = nn.Linear(node_in, hidden)         # raw PM2.5 -> width H
        self.layers  = nn.ModuleList(
            GraPhyLayer(hidden, edge_in, use_convection, use_local)
            for _ in range(layers)
        )
        self.head = nn.Linear(hidden, 1)                  # width H -> one PM2.5 value

    def forward(self, x, edge_index, edge_weight, edge_attr, edge_delev=None,
                node_temp=None, edge_dtemp=None, edge_daod=None):
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight, edge_attr, edge_delev, node_temp,
                      edge_dtemp, edge_daod)
        return self.head(h)   # [N, 1] predicted PM2.5 at every node


# ---------------------------------------------------------------------------
# demo  (same graph setup the module demos use)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from src.graph import build_graph as bg
    from src.model.convection import wind_edge_features, edge_bearings

    long_df = bg.load_sensor_data()
    ids = sorted(long_df["station_id"].unique())
    coords = bg.get_coordinates(ids)
    ids = coords["station_id"].tolist()
    x_m, y_m = bg.project(coords["lat"].to_numpy(), coords["lon"].to_numpy())

    pm = (long_df.pivot_table(index="timestamp", columns="station_id", values="pm25")
          .reindex(columns=ids).sort_index())

    edge_index = bg.knn_edges(x_m, y_m, bg.K)
    dist = bg.distance_matrix(x_m, y_m)
    edge_dist = np.array([dist[i, j] for i, j in edge_index.t()])

    # diffusion & local want a scalar inverse-distance weight per edge
    edge_weight = inverse_distance_weights(torch.tensor(edge_dist, dtype=torch.float))
    # convection wants [distance, wind_along, wind_speed] per edge
    bearing = edge_bearings(x_m, y_m, edge_index.numpy())
    WIND_DIR, WIND_SPEED = np.deg2rad(270.0), 3.0   # one city-wide wind (placeholder)
    edge_attr = wind_edge_features(edge_dist, bearing, WIND_DIR, WIND_SPEED)

    # signed Δelevation per edge (dst − src) feeds the elevation gate. with the
    # constant placeholder elevation this is all zeros -> gate ≡ 1 (inert no-op).
    elev = coords["elevation"].to_numpy()
    src, dst = edge_index.numpy()
    edge_delev = torch.tensor(elev[dst] - elev[src], dtype=torch.float)

    # node features X: PM2.5 at the first hour, one scalar per node (NaN -> 0)
    x = torch.tensor(pm.iloc[0].fillna(0).to_numpy().reshape(-1, 1), dtype=torch.float)

    model = GraPhyNet(node_in=1, edge_in=edge_attr.shape[1], hidden=8, layers=3)
    pred = model(x, edge_index, edge_weight, edge_attr, edge_delev)

    print(f"nodes={x.shape[0]}  edges={edge_index.shape[1]}  layers={len(model.layers)}")
    print(f"X in : {tuple(x.shape)}   (PM2.5 per node)")
    print(f"pred : {tuple(pred.shape)}   (predicted PM2.5 per node)")
    print(f"first 5 predictions: {pred[:5, 0].detach().numpy().round(2)}")

    # --- save results: one timestamped folder per run, so nothing overwrites ---
    import pandas as pd
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")          # e.g. 20260705_2210_03
    run_dir = Path(__file__).resolve().parents[2] / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({
        "station_id": ids,
        "pm25_in":    x[:, 0].numpy(),
        "pm25_pred":  pred[:, 0].detach().numpy(),
    })
    out_path = run_dir / "predictions.csv"
    results.to_csv(out_path, index_label="node")
    print(f"[saved] predictions -> {out_path}")
