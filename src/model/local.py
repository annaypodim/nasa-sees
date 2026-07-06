"""
local module
=============================================================================
local = pollution ADDED or REMOVED right at a node like a fire, a factory, rain
washing particles out; it has nothing to do with transport from elsewhere.

    PATH 1 -- edge features -> a normalization only:
        M = diag( sum of each node's incoming edge weights )   the node's degree
        (a fixed correction from how connected a node is; NOT learned)

    PATH 2 -- node features (PM2.5):
        h   = NodeMLP(x)                     transform each node in isolation
        f   = graph-convolve h to neighbours  (intermediate feature)
        out = M^{-1} · f                      rescale by the node's own degree

    the two paths meet when they rescale; path 2's propagated feature is divided
    by path 1's degree matrix, so a node's local term is normalized by how many
    neighbours it has and redefines that node's feature.

slight tension: a module meant to be "node-local" folding in
edge features at all is arguable -- if it's truly local, neighbours shouldn't
matter. GraPhy does it anyway (the normalization above), so we follow it, but
the edge role here is deliberately minimal: just the degree, nothing learned.

run the demo:   .venv/bin/python local.py
"""
import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    """small 2-layer perceptron -- same building block the other modules use."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class LocalModule(nn.Module):
    def __init__(self, node_in: int, out_dim: int, hidden: int = 8):
        super().__init__()
        self.node_mlp = mlp(node_in, out_dim, hidden)   # transform each node in isolation
        self.conv_mlp = mlp(out_dim, out_dim, hidden)    # what a node hands to its neighbours

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index                    # message flows src -> dst
        h = self.node_mlp(x)                     # [N, out]  node features, in isolation
        # PATH 1: degree = sum of each node's incoming edge weights (the diag of M)
        deg = torch.zeros(x.size(0), device=x.device).index_add_(0, dst, edge_weight)
        # PATH 2: graph-convolve -- weight each neighbour's message by its edge weight
        msg = self.conv_mlp(h)[src] * edge_weight[:, None]   # [E, out]
        f = torch.zeros_like(h).index_add_(0, dst, msg)      # aggregate at dst -> intermediate f
        # the two paths MEET: rescale the propagated feature by the node's own degree (M^{-1})
        return f / deg.clamp(min=1e-12)[:, None]


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.graph import build_graph as bg
    from src.model.diffusion import inverse_distance_weights

    long_df = bg.load_sensor_data()
    ids = sorted(long_df["station_id"].unique())
    coords = bg.get_coordinates(ids)
    ids = coords["station_id"].tolist()
    x_m, y_m = bg.project(coords["lat"].to_numpy(), coords["lon"].to_numpy())

    pm = (long_df.pivot_table(index="timestamp", columns="station_id", values="pm25")
          .reindex(columns=ids).sort_index())

    edge_index = bg.knn_edges(x_m, y_m, bg.K)
    dist = bg.distance_matrix(x_m, y_m)
    edge_dist = torch.tensor([dist[i, j] for i, j in edge_index.t()], dtype=torch.float)
    edge_weight = inverse_distance_weights(edge_dist)   # same weight diffusion uses

    # node features X: PM2.5 at the first hour, one scalar per node (NaN -> 0)
    x = torch.tensor(pm.iloc[0].fillna(0).to_numpy().reshape(-1, 1), dtype=torch.float)

    model = LocalModule(node_in=1, out_dim=8)
    out = model(x, edge_index, edge_weight)

    print(f"nodes={x.shape[0]}  edges={edge_index.shape[1]}")
    print(f"X in  : {tuple(x.shape)}   (PM2.5 per node)")
    print(f"out   : {tuple(out.shape)}   (local source/sink term, 8 feature dims)")
