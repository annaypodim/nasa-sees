"""
Build PM2.5 sensor graphs for Boulder, CO  ->  PyTorch Geometric Data objects.
=============================================================================
Data layout (from https://github.com/annaypodim/nasa-sees, all under ./data):

  1. COORDINATES  -  data/sensor_lat_long_alt
     Not a CSV. A text file with a header line, then named group blocks, each
     a Python-literal list of [sensor_id, lat, lon, altitude_ft]:

         Node Values (Sensor Index, Latitude, Longitude, Altitude)

         Urban
         [
           [283804, 39.98126, -105.12931, 5328],
           ...
         ]

         Rural Sensors
         [ ... ]

     Altitude is in FEET (values ~5300-8800 ft match Boulder-area elevation,
     ~1600-2700 m) -> converted to metres for use alongside UTM distances.

  2. WIND  -  data/group_a.zip (urban) / data/group_b.zip (rural)
     Each zip extracts to a folder of one CSV per sensor, named
     sensor_<id>.csv, columns time,u10,v10,u100,v100, sampled every 2 HOURS.
     group_a's ids match the "Urban" list exactly; group_b's match "Rural
     Sensors" exactly -- so which zip/folder to read is determined by
     SENSOR_SET below, not fixed to one group.

  3. AIR QUALITY (PM2.5)  -  data/upurple_air_denver_boulder/ (urban) or
     data/rpurple_air_denver_boulder/ (rural)
     One raw PurpleAir CSV per sensor, named like
     "<id> 2023-01-01 2023-12-31 60-Minute Average.csv", columns
     time_stamp,pm2.5_atm, HOURLY. The station id is the leading digits of
     the filename. A few sensors' files are header-only (no data in range)
     and are skipped. Each purple-air folder also has one or two ids with no
     entry in sensor_lat_long_alt (and a stray Progress_Logs.log) -- these
     fall out naturally since we only keep ids present in both coords and
     air-quality data.

  4. TIME ALIGNMENT
     Air quality is hourly -> that's the graph's timestep grid. Wind is only
     every 2 hours, so each sensor's wind series is time-interpolated onto
     the hourly AQ grid. Distance / elevation-difference / historical AQ
     correlation are static per edge (computed once); wind angle & speed are
     recomputed for every hourly graph from the interpolated wind at that
     timestep.

Pipeline stages:
    1. SETTINGS
    2. COORDINATES     - parse sensor_lat_long_alt -> station_id/lat/lon/elev
    3. AIR QUALITY      - hourly PurpleAir CSVs      -> tidy table -> wide
    4. WIND             - 2-hourly per-sensor CSVs   -> interpolated to hourly
    5. PROJECT          - lat/lon (degrees)           -> x/y (metres, UTM 13N)
    6. EDGES            - k-nearest-neighbours        -> edge_index [2, E]
    7. EDGE FEATURES    - static part (distance / dElev / corr) + per-step wind
    8. GRAPHS           - one PyG Data object per hourly timestep
    9. SHOW + CHECK     - print every matrix, save CSVs, sanity checks, plot

running:   .venv/bin/python build_graph.py
"""
from __future__ import annotations

import ast
import re
import zipfile
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
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MATRIX_DIR = OUT_DIR / "matrices"

# choose which named group in sensor_lat_long_alt to build a graph for.
SENSOR_SET = "urban"   # "urban" or "rural"

COORDS_FILE = DATA_DIR / "sensor_lat_long_alt"

# per-group file locations: urban <-> group_a / upurple_air_denver_boulder,
# rural <-> group_b / rpurple_air_denver_boulder.
GROUP_CONFIG = {
    "urban": dict(
        purple_air_dir=DATA_DIR / "upurple_air_denver_boulder",
        wind_zip=DATA_DIR / "group_a.zip",
        wind_dir=DATA_DIR / "group_a",
    ),
    "rural": dict(
        purple_air_dir=DATA_DIR / "rpurple_air_denver_boulder",
        wind_zip=DATA_DIR / "group_b.zip",
        wind_dir=DATA_DIR / "group_b",
    ),
}

K = 5                       # each node connects to its K nearest neighbors
LATLON_CRS = "EPSG:4326"    # world geodetic system for lat/lon coordinates (input)
UTM_CRS = "EPSG:32613"      # UTM zone 13N -> metres (covers Boulder, CO)
FEET_TO_M = 0.3048          # sensor_lat_long_alt altitude is in feet

# edge feature columns stored in edge_attr: edge_attr[:, c] is column c below.
# distance / delev / corr are static across timesteps; wind_angle/wind_speed
# are recomputed for every hourly graph from the interpolated wind field.
EDGE_COLS = ["distance_m", "wind_angle", "wind_speed", "delev_m", "pm25_corr"]


# ===========================================================================
# 2. COORDINATES - parse the custom sensor_lat_long_alt text format
#       function exists because node identity is structurally tied to
#       coordinates; this is unlike the other attributes (wind, PM2.5), which
#       are time-varying measurements rather than what defines a node.
# ===========================================================================
def parse_sensor_coords(path: Path) -> pd.DataFrame:
    """Parse `sensor_lat_long_alt` into [station_id, lat, lon, elevation_m, group].

    File format: a header line, then repeated blocks of
        <group name>
        [ [id, lat, lon, altitude_ft], ... ]
    Group names and list lengths aren't fixed, so we scan line-by-line,
    treating any non-bracket, non-empty line as a pending group name, then
    bracket-count from the next "[" to its matching "]" to grab that group's
    full literal list (ast.literal_eval handles the actual parsing).
    """
    lines = path.read_text().splitlines()
    rows = []
    pending_name = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.lower().startswith("node values"):
            i += 1
            continue
        if line.startswith("["):
            depth, block, j = 0, [], i
            while j < len(lines):
                block.append(lines[j])
                depth += lines[j].count("[") - lines[j].count("]")
                j += 1
                if depth == 0:
                    break
            records = ast.literal_eval("\n".join(block))
            group = "urban" if "urban" in (pending_name or "").lower() else "rural"
            for sensor_id, lat, lon, alt_ft in records:
                rows.append({
                    "station_id": str(int(sensor_id)),
                    "lat": float(lat),
                    "lon": float(lon),
                    "elevation": alt_ft * FEET_TO_M,   # -> metres
                    "group": group,
                })
            i = j
        else:
            pending_name = line
            i += 1
    return pd.DataFrame(rows)


# ===========================================================================
# 3. AIR QUALITY - hourly PurpleAir CSVs -> tidy table -> wide [time x node]
# ===========================================================================
def load_air_quality(purple_air_dir: Path, station_ids: list[str]) -> pd.DataFrame:
    """Read every non-empty PurpleAir CSV. The station id is the filename's
    leading digits (filenames look like "120859 2023-01-01 ... .csv"), so
    non-sensor files in the same folder (e.g. Progress_Logs.log) are ignored
    by the *.csv glob, and any csv id not requested is skipped too.
    """
    rows = []
    wanted = set(station_ids)
    for csv_path in sorted(purple_air_dir.glob("*.csv")):
        m = re.match(r"^(\d+)", csv_path.name)
        if not m or m.group(1) not in wanted:
            continue
        station_id = m.group(1)
        df = pd.read_csv(csv_path)
        if df.empty or "pm2.5_atm" not in df.columns:
            continue  # several sensors returned no data in range
        rows.append(pd.DataFrame({
            "station_id": station_id,
            # tz-aware timestamp -> UTC so all sensors share one timeline
            "timestamp": pd.to_datetime(df["time_stamp"], utc=True),
            "pm25": pd.to_numeric(df["pm2.5_atm"], errors="coerce"),
        }))
    if not rows:
        raise FileNotFoundError(
            f"No usable PurpleAir CSVs found under {purple_air_dir} for the "
            f"requested station ids. Check that the nasa-sees `data/` folder "
            f"is next to this script."
        )
    return pd.concat(rows, ignore_index=True)


# ===========================================================================
# 4. WIND - 2-hourly per-sensor CSVs -> interpolated onto the hourly AQ grid
# ===========================================================================
def load_wind(wind_zip: Path, wind_dir: Path, station_ids: list[str],
              target_index: pd.DatetimeIndex):
    """Return (u10_wide, v10_wide, has_wind): wide DataFrames [time x station].

    Each sensor's native 2-hourly series is time-interpolated onto
    `target_index` (the hourly AQ timeline). Sensors with no wind file get
    all-NaN (later zero-filled + flagged) rather than crashing.
    """
    if not wind_dir.exists() and wind_zip.exists():
        print(f"[wind] extracting {wind_zip.name} -> {wind_dir}")
        with zipfile.ZipFile(wind_zip) as zf:
            zf.extractall(DATA_DIR)

    u10_cols, v10_cols = {}, {}
    have_wind = []
    for sid in station_ids:
        csv_path = wind_dir / f"sensor_{sid}.csv"
        if not csv_path.exists():
            u10_cols[sid] = pd.Series(dtype=float)
            v10_cols[sid] = pd.Series(dtype=float)
            continue
        df = pd.read_csv(csv_path, parse_dates=["time"])
        df = df.set_index(pd.DatetimeIndex(df["time"]).tz_localize("UTC")).sort_index()
        have_wind.append(sid)
        # union index so time-based interpolation has native points to work from,
        # then reindex down to just the hourly timestamps we actually need.
        full_index = df.index.union(target_index)
        interp = df.reindex(full_index)[["u10", "v10"]].interpolate(method="time")
        u10_cols[sid] = interp["u10"].reindex(target_index)
        v10_cols[sid] = interp["v10"].reindex(target_index)

    if not have_wind:
        print(f"[wind] no wind files found under {wind_dir} -> wind edge "
              f"features will be zero")
    elif len(have_wind) < len(station_ids):
        missing = sorted(set(station_ids) - set(have_wind))
        print(f"[wind] {len(missing)}/{len(station_ids)} sensors missing wind "
              f"files -> zero wind at those nodes: {missing}")

    u10 = pd.DataFrame(u10_cols, index=target_index)[station_ids]
    v10 = pd.DataFrame(v10_cols, index=target_index)[station_ids]
    return u10.fillna(0.0), v10.fillna(0.0), bool(have_wind)


# ===========================================================================
# 5. PROJECT - lat/lon (degrees) become x/y (metres) so distances are physical
# ===========================================================================
def project(lat: np.ndarray, lon: np.ndarray):
    tf = Transformer.from_crs(LATLON_CRS, UTM_CRS, always_xy=True)
    x, y = tf.transform(lon, lat)          # always_xy: pass lon,lat -> east,north
    return np.asarray(x), np.asarray(y)


# ===========================================================================
# 6. EDGES - k-nearest-neighbours over x/y -> edge_index [2, E], undirected
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


# ===========================================================================
# 7. EDGE FEATURES
#       static part: distance (m), dElev (signed, m), PM2.5-history correlation
#       dynamic part: wind angle/speed, recomputed per hourly timestep from
#       the mean wind vector of an edge's two endpoint sensors
# ===========================================================================
def distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NxN straight-line distance in metres between every pair of sensors."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.hypot(dx, dy)


def correlation_matrix(pm_wide: pd.DataFrame) -> np.ndarray:
    """NxN Pearson correlation of each sensor's full PM2.5 time series."""
    # pandas .corr() is pairwise-complete: uses the hours two sensors share.
    return pm_wide.corr(method="pearson", min_periods=24).to_numpy()


def build_static_edge_attr(edge_index, dist, corr, elevation) -> torch.Tensor:
    """Static edge_attr columns [distance, wind_angle=0, wind_speed=0, dElev, corr].
    Wind columns are placeholders here; build_graphs() overwrites them per timestep.
    """
    i, j = edge_index[0].numpy(), edge_index[1].numpy()
    E = edge_index.shape[1]
    attr = np.zeros((E, len(EDGE_COLS)))
    attr[:, 0] = dist[i, j]
    attr[:, 3] = elevation[j] - elevation[i]    # dElev, SIGNED (keep sign)
    attr[:, 4] = corr[i, j]                      # historical PM2.5 correlation
    return torch.tensor(attr, dtype=torch.float)


def wind_edge_features(edge_index, u_row: np.ndarray, v_row: np.ndarray):
    """Angle (degrees) and speed (m/s) of the mean wind vector across each edge."""
    i, j = edge_index[0].numpy(), edge_index[1].numpy()
    u_mean = (u_row[i] + u_row[j]) / 2.0
    v_mean = (v_row[i] + v_row[j]) / 2.0
    speed = np.hypot(u_mean, v_mean)
    angle = np.degrees(np.arctan2(v_mean, u_mean))
    return angle, speed


# ===========================================================================
# 8. GRAPHS - one PyG Data object per hourly timestep.
#    distance/dElev/corr are the same every graph (static); wind_angle and
#    wind_speed are recomputed each hour from the interpolated wind field,
#    and the node feature x (and target y) is that hour's PM2.5 reading.
# ===========================================================================
def build_graphs(pm_wide, u10_wide, v10_wide, static_attr, edge_index,
                 lat, lon, x_m, y_m, station_ids, max_timesteps=None):
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

        u_row = u10_wide.loc[ts].to_numpy()
        v_row = v10_wide.loc[ts].to_numpy()
        angle, speed = wind_edge_features(edge_index, u_row, v_row)
        edge_attr = static_attr.clone()
        edge_attr[:, 1] = torch.tensor(angle, dtype=torch.float)
        edge_attr[:, 2] = torch.tensor(speed, dtype=torch.float)

        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                 pos=pos, y=x.clone())   # y == ground-truth PM2.5 for now
        g.pos_utm = pos_utm
        g.timestamp = ts
        g.station_ids = station_ids
        # --- LEAVE-ONE-SENSOR-OUT MASK GOES HERE (not built tonight) --------
        # e.g. g.eval_mask = one-hot over nodes; hide that node's y, predict it.
        graphs.append(g)
    return graphs


# ===========================================================================
# 9. SHOW + CHECK  -- print every matrix, save CSVs, run asserts, draw the graph
# ===========================================================================
def show_and_check(graphs, station_ids, lat, lon, x_m, y_m, elevation,
                   dist, corr, edge_index, pm_wide, has_wind):
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    ids = station_ids
    np.set_printoptions(precision=1, suppress=True, linewidth=200)
    pd.set_option("display.width", 200, "display.max_columns", 40)

    g0 = graphs[0]
    print(f"\n{'='*70}\nGRAPH: set={SENSOR_SET!r}  nodes={len(ids)}  "
          f"edges={edge_index.shape[1]}  timesteps={len(graphs)}  "
          f"has_wind={has_wind}\n{'='*70}")

    # ---- node table -------------------------------------------------------
    nodes = pd.DataFrame({"station_id": ids, "lat": lat, "lon": lon,
                          "x_utm_m": x_m, "y_utm_m": y_m, "elev_m": elevation})
    print("\n--- NODE TABLE (index = node number) ---")
    print(nodes.round(2))
    nodes.to_csv(MATRIX_DIR / "node_table.csv", index_label="node")

    # ---- distance matrix ---------------------------------------------------
    dist_df = pd.DataFrame(dist, index=ids, columns=ids)
    print("\n--- DISTANCE MATRIX (metres, NxN) [top-left 6x6] ---")
    print(dist_df.iloc[:6, :6].round(0))
    dist_df.to_csv(MATRIX_DIR / "distance_matrix_m.csv")

    # ---- correlation matrix -------------------------------------------------
    corr_df = pd.DataFrame(corr, index=ids, columns=ids)
    print("\n--- PM2.5 CORRELATION MATRIX (NxN) [top-left 6x6] ---")
    print(corr_df.iloc[:6, :6].round(2))
    corr_df.to_csv(MATRIX_DIR / "correlation_matrix.csv")

    # ---- edge_index -------------------------------------------------------
    print(f"\n--- EDGE_INDEX  shape={tuple(edge_index.shape)} [2, E] (first 10) ---")
    print(edge_index[:, :10].numpy())

    # ---- edge_attr for the first timestep (with human-readable endpoints) --
    edge_df = pd.DataFrame(g0.edge_attr.numpy(), columns=EDGE_COLS)
    edge_df.insert(0, "dst_station", [ids[int(j)] for j in edge_index[1]])
    edge_df.insert(0, "src_station", [ids[int(i)] for i in edge_index[0]])
    print(f"\n--- EDGE_ATTR  shape={tuple(g0.edge_attr.shape)} [E, {len(EDGE_COLS)}] "
          f"t={g0.timestamp} (first 8 edges) ---")
    print(edge_df.head(8).round(2).to_string(index=False))
    edge_df.to_csv(MATRIX_DIR / "edge_attr_first_timestep.csv", index_label="edge")

    # ---- node features x for the first timestep ----------------------------
    xdf = pd.DataFrame({"station_id": ids, "pm25": g0.x.squeeze().numpy()})
    print(f"\n--- NODE FEATURES x  shape={tuple(g0.x.shape)}  (t={g0.timestamp}) ---")
    print(xdf.round(1).to_string(index=False))
    xdf.to_csv(MATRIX_DIR / "node_features_first_timestep.csv", index_label="node")

    print(f"\n[saved] full matrices as CSV -> {MATRIX_DIR}")

    # ---- SANITY CHECKS (real asserts) ---------------------------------------
    print("\n--- SANITY CHECKS ---")
    src, dst = edge_index
    assert (src != dst).all(), "self-loop found!"
    print("[OK] no self-loops")
    assert g0.edge_attr.shape[0] == edge_index.shape[1], "edge_attr/edge_index mismatch"
    print(f"[OK] edge_attr rows ({g0.edge_attr.shape[0]}) == edge_index cols "
          f"({edge_index.shape[1]})")
    eset = {(int(a), int(b)) for a, b in edge_index.t()}
    assert all((b, a) in eset for a, b in eset), "graph not symmetric"
    print("[OK] undirected (every edge has its reverse)")
    assert not torch.isnan(g0.edge_attr).any(), "NaN in edge_attr"
    print("[OK] no NaNs in edge_attr")
    nan_nodes = torch.isnan(g0.x.squeeze()).nonzero().squeeze(-1).tolist()
    print(f"[info] x NaNs at t0: {len(nan_nodes)}/{len(ids)} nodes -> "
          f"{[ids[i] for i in nan_nodes]}  (missing hours = imputation targets)")
    if len(graphs) > 1:
        moved = not torch.equal(graphs[0].edge_attr[:, 1:3], graphs[1].edge_attr[:, 1:3])
        print(f"[info] wind edge features change hour-to-hour: {moved} "
              f"(expected True when real wind data is present)")

    # ---- PLOT ----------------------------------------------------------------
    draw_graph(graphs[0], lat, lon, station_ids, edge_index)


def draw_graph(g0, lat, lon, ids, edge_index):
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
    ax.set(title=f"k-NN sensor graph (k={K}), t={g0.timestamp}",
           xlabel="longitude", ylabel="latitude")
    ax.legend(loc="best")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "knn_graph.png", dpi=130)
    plt.close(fig)
    print(f"[saved] plot -> {OUT_DIR / 'knn_graph.png'}")


# ===========================================================================
# MAIN  -- wire the steps together
# ===========================================================================
def main():
    cfg = GROUP_CONFIG[SENSOR_SET]

    coords = parse_sensor_coords(COORDS_FILE)                        # step 2
    coords = coords[coords["group"] == SENSOR_SET].sort_values("station_id")

    long_pm = load_air_quality(cfg["purple_air_dir"],                # step 3
                               coords["station_id"].tolist())
    # keep only ids that have BOTH coordinates and at least some PM2.5 data
    station_ids = sorted(set(coords["station_id"]) & set(long_pm["station_id"]))
    coords = coords[coords["station_id"].isin(station_ids)].sort_values("station_id")
    station_ids = coords["station_id"].tolist()   # node order = sorted ids
    lat = coords["lat"].to_numpy()
    lon = coords["lon"].to_numpy()
    elevation = coords["elevation"].to_numpy()

    pm_wide = (long_pm.pivot_table(index="timestamp", columns="station_id",
                                   values="pm25")
              .reindex(columns=station_ids).sort_index())

    u10_wide, v10_wide, has_wind = load_wind(cfg["wind_zip"], cfg["wind_dir"],  # step 4
                                             station_ids, pm_wide.index)

    x_m, y_m = project(lat, lon)                                    # step 5

    edge_index = knn_edges(x_m, y_m, K)                             # step 6
    dist = distance_matrix(x_m, y_m)                                # step 7
    corr = correlation_matrix(pm_wide)
    static_attr = build_static_edge_attr(edge_index, dist, corr, elevation)

    graphs = build_graphs(pm_wide, u10_wide, v10_wide, static_attr,  # step 8
                          edge_index, lat, lon, x_m, y_m, station_ids,
                          max_timesteps=48)

    show_and_check(graphs, station_ids, lat, lon, x_m, y_m, elevation,  # step 9
                   dist, corr, edge_index, pm_wide, has_wind)

    print("\nFirst Data object:\n", graphs[0])


if __name__ == "__main__":
    main()
