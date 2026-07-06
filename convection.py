"""
convection module
=============================================================================
convection = pollution gets carried by the wind from one sensor toward another.

diffusion had a closed-form operator (the Laplacian). convection does NOT --
the whole wind -> PM2.5 relationship is *learned* through small MLPs. there is
no physics matrix here, just message passing:

    e   = EdgeMLP(edge_feat)            edge_feat = [dist, wind_along, wind_speed]
    h   = NodeMLP(x)                    each node's features, reshaped to width H
    m   = MessageMLP( [h[src], e] )     what neighbour src tells dst, given the wind
    agg = sum of m over each node's incoming edges
    agg = agg + h                       RESIDUAL: keep the node's own identity
    out = UpdateMLP( [h + agg, agg] )   sees combined state AND the raw message

        wind_along = wind_speed * cos(wind_dir - edge_bearing)

    +1 = wind blows straight from src to dst, -1 = straight against, 0 = crosswind.
    that geometric relationship can only live on the edge, never on a node.

run the demo:   .venv/bin/python convection.py
"""
import numpy as np
import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    """small 2-layer perceptron -- the reusable building block of this module."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class ConvectionModule(nn.Module):
    def __init__(self, node_in: int, edge_in: int, out_dim: int, hidden: int = 8):
        super().__init__()
        self.node_mlp = mlp(node_in, hidden, hidden)   # reshape node features -> width H
        self.edge_mlp = mlp(edge_in, hidden, hidden)   # raw wind/dist -> learned edge embedding
        self.msg_mlp  = mlp(2 * hidden, hidden, hidden)  # [neighbour, edge] -> one message
        self.upd_mlp  = mlp(2 * hidden, out_dim, hidden)  # [combined, message] -> node output

    def forward(self, x, edge_index, edge_attr, edge_gate=None):
        src, dst = edge_index                       # message flows src -> dst
        h = self.node_mlp(x)                        # [N, H]
        e = self.edge_mlp(edge_attr)                # [E, H]
        # one message per edge: what neighbour `src` tells `dst`, shaped by the wind
        m = self.msg_mlp(torch.cat([h[src], e], dim=1))   # [E, H]
        # ELEVATION GATE: dampen the message when the pair differs in height. one
        # shared scalar per edge (see elevation.py); None -> ungated (gate==1).
        if edge_gate is not None:
            m = m * edge_gate[:, None]
        # aggregate: sum each node's incoming messages
        agg = torch.zeros_like(h).index_add_(0, dst, m)   # [N, H]
        # RESIDUAL: fold the node's own features back so neighbours don't swamp it
        agg = agg + h
        # update sees the combined state (h+agg) AND the isolated message (agg-h==msg sum)
        return self.upd_mlp(torch.cat([h + agg, agg], dim=1))


def wind_edge_features(dist_m, bearing_rad, wind_dir_rad, wind_speed):
    """Raw city wind -> per-edge features [distance, wind_along, wind_speed].

    wind_dir_rad / wind_speed are one city-wide value (scalars or per-edge arrays).
    wind_along = wind_speed * cos(wind_dir - edge_bearing): the wind projected onto
    each sensor pair, which is the only part of the wind that varies edge to edge.
    """
    wind_along = wind_speed * np.cos(wind_dir_rad - bearing_rad)
    speed = np.broadcast_to(wind_speed, bearing_rad.shape)
    return torch.tensor(np.column_stack([dist_m, wind_along, speed]), dtype=torch.float)


def edge_bearings(x_m, y_m, edge_index):
    """Compass angle (radians) of each edge src->dst, from UTM x/y positions."""
    src, dst = edge_index
    dx = x_m[dst] - x_m[src]
    dy = y_m[dst] - y_m[src]
    return np.arctan2(dy, dx)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import build_graph as bg

    long_df = bg.load_sensor_data()
    ids = sorted(long_df["station_id"].unique())
    coords = bg.get_coordinates(ids)
    ids = coords["station_id"].tolist()
    x_m, y_m = bg.project(coords["lat"].to_numpy(), coords["lon"].to_numpy())

    pm = (long_df.pivot_table(index="timestamp", columns="station_id", values="pm25")
          .reindex(columns=ids).sort_index())

    edge_index = bg.knn_edges(x_m, y_m, bg.K)

    # per-edge geometry
    dist = bg.distance_matrix(x_m, y_m)
    edge_dist = np.array([dist[i, j] for i, j in edge_index.t()])
    bearing = edge_bearings(x_m, y_m, edge_index.numpy())

    # ONE city-wide wind for the demo (wind is area-wide in Boulder for now).
    # placeholder until the .nc wind is wired into edge_attr -- angle is what matters.
    WIND_DIR = np.deg2rad(270.0)   # blowing toward the east
    WIND_SPEED = 3.0               # m/s, same everywhere
    edge_attr = wind_edge_features(edge_dist, bearing, WIND_DIR, WIND_SPEED)

    # node features X: PM2.5 at the first hour, one scalar per node (NaN -> 0)
    x = torch.tensor(pm.iloc[0].fillna(0).to_numpy().reshape(-1, 1), dtype=torch.float)

    model = ConvectionModule(node_in=1, edge_in=edge_attr.shape[1], out_dim=8)
    out = model(x, edge_index, edge_attr)

    print(f"nodes={x.shape[0]}  edges={edge_index.shape[1]}")
    print(f"X in      : {tuple(x.shape)}   (PM2.5 per node)")
    print(f"edge_attr : {tuple(edge_attr.shape)}   [distance, wind_along, wind_speed]")
    print(f"out       : {tuple(out.shape)}   (convected, 8 feature dims)")
    walong = edge_attr[:, 1]
    print(f"wind_along range: {walong.min():.2f} .. {walong.max():.2f}  "
          f"(+1*speed = with the wind, -1*speed = against)")
