"""
diffusion module 
=============================================================================
diffusion = pollution spreads from high concentration to low, along the graph.

    out = l ⊙ σ( L_D · X · W )      with   L_D = 2L/λmax − I

    X  = node features going in            (PM2.5 history)
    W  = learnable weight   -> what gets spread
    L  = graph Laplacian    -> "how different is a node from its neighbours"
    L_D= rescaled Laplacian -> keeps eigenvalues in [-1, 1] so the math is stable
    σ  = sigmoid squash
    l  = learnable PER-FEATURE diffusion strength (GraPhy's actual contribution:
         a vector, not one global rate, so each feature diffuses at its own speed)

run the demo:   .venv/bin/python diffusion.py
"""
import torch
import torch.nn as nn


class DiffusionModule(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)   # what gets spread
        self.l = nn.Parameter(torch.ones(out_dim))        # per-feature diffusion rate

    def forward(self, x, edge_index, edge_weight):
        N = x.size(0)
        eye = torch.eye(N, device=x.device)
        # 1. adjacency A: put each edge's inverse-distance weight at [i, j]
        A = torch.zeros(N, N, device=x.device)
        A[edge_index[0], edge_index[1]] = edge_weight
        # 2. degree D (how strongly a node is connected overall) -> D^(-1/2)
        d = A.sum(1).clamp(min=1e-12).pow(-0.5)
        # 3. normalized graph Laplacian L = I − D^(-1/2) A D^(-1/2)  (eigenvalues in [0, 2])
        L = eye - d[:, None] * A * d[None, :]
        # 4. rescale so eigenvalues land in [-1, 1] (spectral GCN needs this to be stable)
        lam = torch.linalg.eigvalsh(L).max()
        L_D = 2 * L / lam - eye
        # 5. spread (L_D·X·W), squash (σ), scale each feature by its own rate (l)
        return self.l * torch.sigmoid(L_D @ self.W(x))


def inverse_distance_weights(dist_m, eps: float = 1.0):
    """Raw distance (metres) -> edge weight. Closer sensors diffuse more."""
    return 1.0 / (dist_m + eps)


# --------------------------------------------------------------------------- 
# demo
# --------------------------------------------------------------------------- 
if __name__ == "__main__":
    from src.graph import build_graph as bg

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
    edge_weight = inverse_distance_weights(edge_dist)

    # node features X: PM2.5 at the first hour, one scalar per node (NaN -> 0)
    x = torch.tensor(pm.iloc[0].fillna(0).to_numpy().reshape(-1, 1), dtype=torch.float)

    model = DiffusionModule(in_dim=1, out_dim=8)
    out = model(x, edge_index, edge_weight)

    print(f"nodes={x.shape[0]}  edges={edge_index.shape[1]}")
    print(f"X in  : {tuple(x.shape)}   (PM2.5 per node)")
    print(f"out   : {tuple(out.shape)}   (diffused, 8 feature dims)")
    print(f"learnable l (per-feature diffusion rate): {model.l.detach().numpy().round(3)}")
