"""
fusion 
=============================================================================
three modules each hand back a per-node feature vector for the SAME nodes:

    local       -> pollution added/removed right at the node
    diffusion   -> pollution spreading high -> low along the graph
    convection  -> pollution carried by the wind from node to node

fusion picks how much to trust each one, PER NODE and PER HOUR:

    z   = [ local | diffusion | convection ]      concatenate the three vectors
    g   = GateMLP(z)                               -> 3 raw scores
    w   = softmax(g)                               -> 3 weights that sum to 1
    out = w0·local + w1·diffusion + w2·convection  the blended feature

    softmax forces w0 + w1 + w2 = 1, so the three modules genuinely compete for
    trust rather than all being turned up at once.

why no elevation input here: elevation is already baked into diffusion and
convection (through the graph geometry / wind projection), so it is not a 4th
thing to fuse -- only the three module outputs are.

run the demo:   .venv/bin/python fusion.py
"""
import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    """small 2-layer perceptron -- the same building block the modules use."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Fusion(nn.Module):
    def __init__(self, dim: int, hidden: int = 8, n_modules: int = 3):
        super().__init__()
        # in: the n active module outputs concatenated (n*dim). out: one score each.
        # n_modules is now configurable so we can ablate modules (e.g. diffusion-only
        # -> n_modules=1) without a degenerate softmax over dead inputs. See model.py.
        self.n_modules = n_modules
        self.gate_mlp = mlp(n_modules * dim, n_modules, hidden)

    def forward(self, *outs):
        """Blend a variable number of [N, dim] module outputs by a per-node softmax.
        With one module this is a learned pass-through; with three it is the original
        [local, diffusion, convection] competition."""
        assert len(outs) == self.n_modules, \
            f"Fusion expected {self.n_modules} module outputs, got {len(outs)}"
        z = torch.cat(outs, dim=1)                       # [N, n*dim]
        w = torch.softmax(self.gate_mlp(z), dim=1)       # [N, n], each row sums to 1
        stacked = torch.stack(outs, dim=1)               # [N, n, dim]
        out = (w[:, :, None] * stacked).sum(dim=1)       # [N, dim]
        return out, w


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from src.graph import build_graph as bg
    from src.model.diffusion import DiffusionModule, inverse_distance_weights
    from src.model.local import LocalModule
    from src.model.convection import ConvectionModule, edge_bearings, wind_edge_features

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
    edge_weight = inverse_distance_weights(torch.tensor(edge_dist, dtype=torch.float))

    # convection needs the wind projected onto each edge (see convection.py)
    bearing = edge_bearings(x_m, y_m, edge_index.numpy())
    WIND_DIR = np.deg2rad(270.0)   # placeholder city-wide wind (blowing east)
    WIND_SPEED = 3.0
    edge_attr = wind_edge_features(edge_dist, bearing, WIND_DIR, WIND_SPEED)

    # node features X: PM2.5 at the first hour, one scalar per node (NaN -> 0)
    x = torch.tensor(pm.iloc[0].fillna(0).to_numpy().reshape(-1, 1), dtype=torch.float)

    DIM = 8
    local = LocalModule(node_in=1, out_dim=DIM)
    diffusion = DiffusionModule(in_dim=1, out_dim=DIM)
    convection = ConvectionModule(node_in=1, edge_in=edge_attr.shape[1], out_dim=DIM)
    fusion = Fusion(dim=DIM)

    local_out = local(x, edge_index, edge_weight)
    diffusion_out = diffusion(x, edge_index, edge_weight)
    convection_out = convection(x, edge_index, edge_attr)
    out, w = fusion(local_out, diffusion_out, convection_out)

    print(f"nodes={x.shape[0]}  edges={edge_index.shape[1]}")
    print(f"each module out : {tuple(local_out.shape)}   (local / diffusion / convection)")
    print(f"fused out       : {tuple(out.shape)}   (weighted blend, {DIM} feature dims)")
    print(f"weights w       : {tuple(w.shape)}   (per node: [local, diffusion, convection], sum=1)")
    mean_w = w.mean(0).detach().numpy().round(3)
    print(f"mean weight across nodes: local={mean_w[0]}  diffusion={mean_w[1]}  convection={mean_w[2]}")
