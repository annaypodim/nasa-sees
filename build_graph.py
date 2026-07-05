"""
Build PM2.5 sensor graphs for Boulder, CO  ->  PyTorch Geometric Data objects.
=============================================================================

    1. SETTINGS        - the few knobs you might change
    2. LOAD            - raw PurpleAir CSVs         -> tidy table
    3. COORDINATES     - attach lat/lon/elevation   (placeholder for now)
    4. PROJECT         - lat/lon (degrees)          -> x/y (metres, UTM 13N)
    5. EDGES           - k-nearest-neighbours       -> edge_index [2, E]
    6. EDGE FEATURES   - distance / wind / dElev / correlation -> edge_attr [E, 5]
    7. GRAPHS          - one PyG Data object per timestep
    8. SHOW + CHECK    - print every matrix, save CSVs, sanity checks, plot

running:   .venv/bin/python build_graph.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save a PNG instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyproj import Transformer
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


# =========================================================================== 
# 1. settings
# =========================================================================== 
# define data directories
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MATRIX_DIR = OUT_DIR / "matrices"

# choose urban group: "u" for urban (26 sensors) or "r" for rural (14 sensors)
SENSOR_SET = "u"
SENSOR_DIR = DATA_DIR / f"{SENSOR_SET}purple_air_denver_boulder"

K = 5                       # each node connects to its K nearest neighbors
LATLON_CRS = "EPSG:4326"    # world geodetic system for lat/lon coordinates (input)
UTM_CRS = "EPSG:32613"      # UTM zone 13N -> metres (covers Boulder, CO)

# filler data for now
COORDS_FILE = DATA_DIR / "sensor_coords.csv"   # columns: station_id,lat,lon,elevation

# determinisitic placeholders which will get replaced
PLACEHOLDER_BOX = dict(lat=(39.95, 40.10), lon=(-105.30, -105.15))  # Boulder-ish
PLACEHOLDER_ELEV_M = 1655.0   # constant -> elevation-difference edges are 0
SEED = 1337

# edge feature columns to get stored in edge_attr matrix: edge_attr[:, c] is column c below.
EDGE_COLS = ["distance_m", "wind_angle", "wind_speed", "delev_m", "pm25_corr"]


# =========================================================================== 
# 2. LOAD - csvs into one table 
#       [station_id, timestamp, pm25]
# =========================================================================== 
def load_sensor_data() -> pd.DataFrame:
    """Read every non-empty PurpleAir CSV. The station id is the filename."""
    rows = []
    for csv_path in sorted(SENSOR_DIR.glob("*.csv")):
        station_id = re.match(r"^(\d+)", csv_path.name).group(1)
        df = pd.read_csv(csv_path)
        if df.empty or "pm2.5_atm" not in df.columns:
            continue  # several sensors returned no data in range
        rows.append(pd.DataFrame({
            "station_id": station_id,
            # tz-aware timestamp -> UTC so all sensors share one timeline
            "timestamp": pd.to_datetime(df["time_stamp"], utc=True),
            "pm25": pd.to_numeric(df["pm2.5_atm"], errors="coerce"),
        }))
    return pd.concat(rows, ignore_index=True)


# =========================================================================== 
# 3. COORDINATES - from csv
#       function exists because node identity is structurally tied to coordinates; this way coords comes from one place
#       this is unlike the other attributes, which are measurements and are model inputs; sensor -> sensor id + location
# =========================================================================== 
def get_coordinates(station_ids: list[str]) -> pd.DataFrame:
    """Return [station_id, lat, lon, elevation]. Real file if present, else fake."""
    if COORDS_FILE.exists():
        print(f"[coords] using REAL coordinates from {COORDS_FILE.name}")
        c = pd.read_csv(COORDS_FILE, dtype={"station_id": str})
        if "elevation" not in c:
            c["elevation"] = PLACEHOLDER_ELEV_M
        c["elevation"] = c["elevation"].fillna(PLACEHOLDER_ELEV_M)
        c["placeholder"] = False
    else:
        print(f"[coords] {COORDS_FILE.name} not found -> PLACEHOLDER coords (TODO)")
        rng = np.random.default_rng(SEED)  # deterministic: same map every run
        ids = sorted(station_ids)
        c = pd.DataFrame({
            "station_id": ids,
            "lat": rng.uniform(*PLACEHOLDER_BOX["lat"], len(ids)),
            "lon": rng.uniform(*PLACEHOLDER_BOX["lon"], len(ids)),
            "elevation": PLACEHOLDER_ELEV_M,   # TODO: real DEM -> dElev != 0
            "placeholder": True,
        })
    return c[c["station_id"].isin(station_ids)].sort_values("station_id")


# =========================================================================== 
# 4. PROJECT - lat/lon (degrees) become x/y (metres) so distances are physical
# =========================================================================== 
def project(lat: np.ndarray, lon: np.ndarray):
    tf = Transformer.from_crs(LATLON_CRS, UTM_CRS, always_xy=True)
    x, y = tf.transform(lon, lat)          # always_xy: pass lon,lat -> east,north
    return np.asarray(x), np.asarray(y)


# =========================================================================== 
# 5. EDGES - creating the graph skeleton by getting the connecting line
#       
#       deciding that the nearest K neighbors of a node (k-nearest-neighbors over x/y) should be connected w/ an edge
#       create edge_index [2, E], undirected
# =========================================================================== 
def knn_edges(x: np.ndarray, y: np.ndarray, k: int) -> torch.Tensor:
    """Connect each node to its k nearest neighbours; make it undirected."""
    pts = np.column_stack([x, y])
    k = min(k, len(pts) - 1)
    # +1 because the nearest point to any node is itself (column 0, dropped).
    _, nbr = NearestNeighbors(n_neighbors=k + 1).fit(pts).kneighbors(pts)

    edges = set()
    for i in range(len(pts)):
        for j in nbr[i, 1:]:
            edges.add((i, int(j)))
            edges.add((int(j), i))   # add reverse -> undirected, no self-loops
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


# =========================================================================== #
# 6. EDGE FEATURES - build full NxN matrices, then read off each edge
#       in matrix, each column is a sensor, so each cell holds the into of a sensor pair
#       matrix[i,j] holds a vector with the relationship between sensor i and j
# 
#       create a distance matrix - meters from each other  
#       correlation matrix - pm2.5 history similarities
#       edge attribute tensor - edge features, one row per edge with 5 feature columns  
#           (distance_m, wind_angle, wind_speed, delev_m, pm25_corr))
# =========================================================================== #
def distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NxN straight-line distance in metres between every pair of sensors."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.hypot(dx, dy)


def correlation_matrix(pm_wide: pd.DataFrame) -> np.ndarray:
    """NxN Pearson correlation of each sensor's full PM2.5 time series."""
    # pandas .corr() is pairwise-complete: uses the hours two sensors share.
    return pm_wide.corr(method="pearson", min_periods=24).to_numpy()


def build_edge_attr(edge_index, dist, corr, elevation) -> torch.Tensor:
    """edge_attr[e] = [distance, wind_angle, wind_speed, dElev(signed), corr]."""
    E = edge_index.shape[1]
    attr = np.zeros((E, len(EDGE_COLS)))
    for e in range(E):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        attr[e, 0] = dist[i, j]                     # distance (m)
        attr[e, 1] = 0.0                            # wind angle  TODO: from .nc
        attr[e, 2] = 0.0                            # wind speed  TODO: from .nc
        attr[e, 3] = elevation[j] - elevation[i]    # dElev, SIGNED (keep sign)
        attr[e, 4] = corr[i, j]                     # historical PM2.5 corr
    return torch.tensor(attr, dtype=torch.float)


# =========================================================================== #
# 7. GRAPHS - one PyG Data object per timestep (hourly)
#    we assume graph structure (edges, edge_attr, positions) is the same every
#    hour without hourly wind data, only the node PM2.5 signal x (and target y) change
# =========================================================================== #
def build_graphs(pm_wide, edge_index, edge_attr, lat, lon, x_m, y_m,
                 station_ids, max_timesteps=None):
    pos = torch.tensor(np.column_stack([lat, lon]), dtype=torch.float)      # lat/lon
    pos_utm = torch.tensor(np.column_stack([x_m, y_m]), dtype=torch.float)  # metres

    timestamps = list(pm_wide.index)
    if max_timesteps:
        timestamps = timestamps[:max_timesteps]

    graphs = []
    for ts in timestamps:
        # x: PM2.5 at this hour, one scalar per node -> [num_nodes, 1].
        # NaN where a sensor didn't report -> that's the imputation target.
        x = torch.tensor(pm_wide.loc[ts].to_numpy().reshape(-1, 1), dtype=torch.float)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                 pos=pos, y=x.clone())   # y == ground-truth PM2.5 for now
        g.pos_utm = pos_utm
        g.timestamp = ts
        g.station_ids = station_ids
        # --- LEAVE-ONE-SENSOR-OUT MASK GOES HERE (not built tonight) --------
        # e.g. g.eval_mask = one-hot over nodes; hide that node's y, predict it.
        graphs.append(g)
    return graphs


# =========================================================================== #
# 8. SHOW + CHECK  -- print every matrix, save CSVs, run asserts, draw the graph
# =========================================================================== #
def show_and_check(graphs, station_ids, lat, lon, x_m, y_m, elevation,
                   dist, corr, edge_index, edge_attr, pm_wide, placeholder):
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    ids = station_ids
    np.set_printoptions(precision=1, suppress=True, linewidth=200)
    pd.set_option("display.width", 200, "display.max_columns", 40)

    print(f"\n{'='*70}\nGRAPH: set={SENSOR_SET!r}  nodes={len(ids)}  "
          f"edges={edge_index.shape[1]}  timesteps={len(pm_wide)}  "
          f"placeholder_coords={placeholder}\n{'='*70}")

    # ---- node table -------------------------------------------------------
    nodes = pd.DataFrame({"station_id": ids, "lat": lat, "lon": lon,
                          "x_utm_m": x_m, "y_utm_m": y_m, "elev_m": elevation})
    print("\n--- NODE TABLE (index = node number) ---")
    print(nodes.round(2))
    nodes.to_csv(MATRIX_DIR / "node_table.csv", index_label="node")

    # ---- distance matrix --------------------------------------------------
    dist_df = pd.DataFrame(dist, index=ids, columns=ids)
    print("\n--- DISTANCE MATRIX (metres, NxN) [top-left 6x6] ---")
    print(dist_df.iloc[:6, :6].round(0))
    dist_df.to_csv(MATRIX_DIR / "distance_matrix_m.csv")

    # ---- correlation matrix ----------------------------------------------
    corr_df = pd.DataFrame(corr, index=ids, columns=ids)
    print("\n--- PM2.5 CORRELATION MATRIX (NxN) [top-left 6x6] ---")
    print(corr_df.iloc[:6, :6].round(2))
    corr_df.to_csv(MATRIX_DIR / "correlation_matrix.csv")

    # ---- edge_index -------------------------------------------------------
    print(f"\n--- EDGE_INDEX  shape={tuple(edge_index.shape)} [2, E] (first 10) ---")
    print(edge_index[:, :10].numpy())

    # ---- edge_attr (with human-readable endpoints) -----------------------
    edge_df = pd.DataFrame(edge_attr.numpy(), columns=EDGE_COLS)
    edge_df.insert(0, "dst_station", [ids[int(j)] for j in edge_index[1]])
    edge_df.insert(0, "src_station", [ids[int(i)] for i in edge_index[0]])
    print(f"\n--- EDGE_ATTR  shape={tuple(edge_attr.shape)} [E, {len(EDGE_COLS)}] "
          f"(first 8 edges) ---")
    print(edge_df.head(8).round(2).to_string(index=False))
    edge_df.to_csv(MATRIX_DIR / "edge_attr.csv", index_label="edge")

    # ---- node features x for the first timestep --------------------------
    g0 = graphs[0]
    xdf = pd.DataFrame({"station_id": ids, "pm25": g0.x.squeeze().numpy()})
    print(f"\n--- NODE FEATURES x  shape={tuple(g0.x.shape)}  (t={g0.timestamp}) ---")
    print(xdf.round(1).to_string(index=False))
    xdf.to_csv(MATRIX_DIR / "node_features_first_timestep.csv", index_label="node")

    print(f"\n[saved] full matrices as CSV -> {MATRIX_DIR}")

    # ---- SANITY CHECKS (real asserts) ------------------------------------
    print("\n--- SANITY CHECKS ---")
    src, dst = edge_index
    assert (src != dst).all(), "self-loop found!"
    print("[OK] no self-loops")
    assert edge_attr.shape[0] == edge_index.shape[1], "edge_attr/edge_index mismatch"
    print(f"[OK] edge_attr rows ({edge_attr.shape[0]}) == edge_index cols "
          f"({edge_index.shape[1]})")
    eset = {(int(a), int(b)) for a, b in edge_index.t()}
    assert all((b, a) in eset for a, b in eset), "graph not symmetric"
    print("[OK] undirected (every edge has its reverse)")
    assert not torch.isnan(edge_attr).any(), "NaN in edge_attr"
    print("[OK] no NaNs in edge_attr")
    nan_nodes = torch.isnan(g0.x.squeeze()).nonzero().squeeze(-1).tolist()
    print(f"[info] x NaNs at t0: {len(nan_nodes)}/{len(ids)} nodes -> "
          f"{[ids[i] for i in nan_nodes]}  (missing hours = imputation targets)")

    # ---- PLOT -------------------------------------------------------------
    draw_graph(graphs[0], lat, lon, station_ids, edge_index, placeholder)


def draw_graph(g0, lat, lon, ids, edge_index, placeholder):
    fig, ax = plt.subplots(figsize=(9, 8))
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="0.7", lw=0.8, zorder=1)
    pm = g0.x.squeeze().numpy()
    ok = ~np.isnan(pm)
    sc = ax.scatter(lon[ok], lat[ok], c=pm[ok], s=90, cmap="viridis",
                    edgecolors="k", zorder=3)
    ax.scatter(lon[~ok], lat[~ok], marker="x", c="red", s=90, zorder=3,
               label="missing (NaN)")
    fig.colorbar(sc, ax=ax, label="PM2.5 (ug/m3)")
    for idx, sid in enumerate(ids):
        ax.annotate(sid, (lon[idx], lat[idx]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    tag = " [PLACEHOLDER COORDS - not physical]" if placeholder else ""
    ax.set(title=f"k-NN sensor graph (k={K}), t={g0.timestamp}{tag}",
           xlabel="longitude", ylabel="latitude")
    ax.legend(loc="best")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "knn_graph.png", dpi=130)
    plt.close(fig)
    print(f"[saved] plot -> {OUT_DIR / 'knn_graph.png'}")


# =========================================================================== #
# MAIN  -- wire the steps together
# =========================================================================== #
def main():
    long_df = load_sensor_data()                                   # step 2
    station_ids = sorted(long_df["station_id"].unique())

    coords = get_coordinates(station_ids)                          # step 3
    station_ids = coords["station_id"].tolist()   # node order = sorted ids
    lat = coords["lat"].to_numpy()
    lon = coords["lon"].to_numpy()
    elevation = coords["elevation"].to_numpy()
    placeholder = bool(coords["placeholder"].iloc[0])

    x_m, y_m = project(lat, lon)                                   # step 4

    # PM2.5 as a wide table: rows = timestamps, columns = sensors (node order).
    pm_wide = (long_df.pivot_table(index="timestamp", columns="station_id",
                                   values="pm25")
               .reindex(columns=station_ids).sort_index())

    edge_index = knn_edges(x_m, y_m, K)                            # step 5
    dist = distance_matrix(x_m, y_m)                              # step 6
    corr = correlation_matrix(pm_wide)
    edge_attr = build_edge_attr(edge_index, dist, corr, elevation)

    graphs = build_graphs(pm_wide, edge_index, edge_attr, lat, lon,   # step 7
                          x_m, y_m, station_ids, max_timesteps=48)

    show_and_check(graphs, station_ids, lat, lon, x_m, y_m, elevation,  # step 8
                   dist, corr, edge_index, edge_attr, pm_wide, placeholder)

    print("\nFirst Data object:\n", graphs[0])


if __name__ == "__main__":
    main()
