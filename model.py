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

from diffusion import DiffusionModule, inverse_distance_weights
from convection import ConvectionModule
from local import LocalModule
from fusion import Fusion


class GraPhyLayer(nn.Module):
    """one message-passing layer: three modules -> fuse -> residual."""
    def __init__(self, dim: int, edge_in: int):
        super().__init__()
        self.diffusion  = DiffusionModule(dim, dim)
        self.convection = ConvectionModule(dim, edge_in, dim)
        self.local      = LocalModule(dim, dim)
        self.fusion     = Fusion(dim)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        d = self.diffusion(h, edge_index, edge_weight)
        c = self.convection(h, edge_index, edge_attr)
        l = self.local(h, edge_index, edge_weight)
        fused, _ = self.fusion(l, d, c)   # returns (blend, weights); we want the blend
        return h + fused   # RESIDUAL: nudge, don't overwrite -> no oversmoothing


class GraPhyNet(nn.Module):
    """encode PM2.5 -> stack a few layers -> predict PM2.5 back."""
    def __init__(self, node_in: int, edge_in: int, hidden: int = 8, layers: int = 3):
        super().__init__()
        self.encoder = nn.Linear(node_in, hidden)         # raw PM2.5 -> width H
        self.layers  = nn.ModuleList(
            GraPhyLayer(hidden, edge_in) for _ in range(layers)
        )
        self.head = nn.Linear(hidden, 1)                  # width H -> one PM2.5 value

    def forward(self, x, edge_index, edge_weight, edge_attr):
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight, edge_attr)
        return self.head(h)   # [N, 1] predicted PM2.5 at every node


# ---------------------------------------------------------------------------
# demo  (same graph setup the module demos use)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import build_graph as bg
    from convection import wind_edge_features, edge_bearings

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

    # node features X: PM2.5 at the first hour, one scalar per node (NaN -> 0)
    x = torch.tensor(pm.iloc[0].fillna(0).to_numpy().reshape(-1, 1), dtype=torch.float)

    model = GraPhyNet(node_in=1, edge_in=edge_attr.shape[1], hidden=8, layers=3)
    pred = model(x, edge_index, edge_weight, edge_attr)

    print(f"nodes={x.shape[0]}  edges={edge_index.shape[1]}  layers={len(model.layers)}")
    print(f"X in : {tuple(x.shape)}   (PM2.5 per node)")
    print(f"pred : {tuple(pred.shape)}   (predicted PM2.5 per node)")
    print(f"first 5 predictions: {pred[:5, 0].detach().numpy().round(2)}")

    # --- save results: one timestamped folder per run, so nothing overwrites ---
    import pandas as pd
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")          # e.g. 20260705_2210_03
    run_dir = Path(__file__).resolve().parent / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({
        "station_id": ids,
        "pm25_in":    x[:, 0].numpy(),
        "pm25_pred":  pred[:, 0].detach().numpy(),
    })
    out_path = run_dir / "predictions.csv"
    results.to_csv(out_path, index_label="node")
    print(f"[saved] predictions -> {out_path}")
