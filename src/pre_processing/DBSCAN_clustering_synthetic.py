import os
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import sys

# --- Add parent directory to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities.instance_generator import Trip
from utilities import haversine

import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import shutil

from models.eb_md_vsp_tw_solver_with_vi import load_instance_from_txt

# ------------------------------------------------------------
# Configuration (tune here, not in logic)
# ------------------------------------------------------------

SPATIAL_EPS_METERS = 5000.0
TEMPORAL_EPS_MIN   = 25.0
MIN_TRIPS_PER_CLUSTER = 1
MAX_TRIPS_PER_CLUSTER = 20

FIRST_TRIP_INDEX = 0
MAX_TRIPS_PER_ROUTE_PER_DIRECTION = 1000

TIME_WEIGHT  = 1.0
SPACE_WEIGHT = 0.5

def read_raw_instance_lines(txt_path: str) -> list[str]:
    """Read the instance txt file as raw lines (no parsing), preserving formatting."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]
    
def parse_header_KTF(header_line: str):
    parts = header_line.split()
    if len(parts) < 3:
        raise ValueError(f"Header row too short: {header_line}")
    K = int(float(parts[0]))
    T = int(float(parts[1]))
    F = int(float(parts[2]))
    return K, T, F, parts

def split_instance_blocks(lines: list[str]):
    if not lines:
        raise ValueError("Empty instance file")

    header_line = lines[0]
    K, T, F, header_parts = parse_header_KTF(header_line)

    depot_count = 2 * K

    expected_min = 1 + depot_count + T + F
    if len(lines) < expected_min:
        raise ValueError(
            f"File has {len(lines)} lines but expected at least {expected_min} "
            f"(1 header + {depot_count} depots + {T} trips + {F} chargers)"
        )

    depots = lines[1 : 1 + depot_count]
    trips  = lines[1 + depot_count : 1 + depot_count + T]
    chargers = lines[1 + depot_count + T : 1 + depot_count + T + F]

    tail = lines[1 + depot_count + T + F :]
    if tail:
        # If your format sometimes has extra stuff, you can decide to keep it too.
        # For strict formats, better to error.
        raise ValueError(f"Unexpected extra lines after chargers: {len(tail)}")

    return header_parts, depots, trips, chargers

def update_header_T(header_parts: list[str], new_T: int) -> str:
    parts = header_parts.copy()
    if len(parts) < 2:
        raise ValueError("Header missing T field")
    parts[1] = str(int(new_T))
    return "\t".join(parts)

def export_clusters_to_txt_instances_verbatim(
    clusters: dict,
    source_txt_path: str,
    output_root: str,
    out_filename: str = "instance.txt"
):
    os.makedirs(output_root, exist_ok=True)

    # 1) Read and split raw instance file
    lines = read_raw_instance_lines(source_txt_path)
    header_parts, depot_lines, _trip_lines_original, charger_lines = split_instance_blocks(lines)

    # 2) Write one file per cluster
    for cid, c_trips in clusters.items():
        cluster_dir = os.path.join(output_root, f"cluster_{cid}")
        os.makedirs(cluster_dir, exist_ok=True)

        cluster_trip_lines = build_trip_lines_for_cluster(c_trips)

        # Header with updated T only
        header_line = update_header_T(header_parts, len(cluster_trip_lines))

        out_path = os.path.join(cluster_dir, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header_line + "\n")
            for ln in depot_lines:
                f.write(ln + "\n")
            for ln in cluster_trip_lines:
                f.write(ln + "\n")
            for ln in charger_lines:
                f.write(ln + "\n")

def build_trip_lines_for_cluster(c_trips: list[Trip]) -> list[str]:
    lines = []
    for new_tid, t in enumerate(c_trips, start=1):
        row = [
            new_tid,
            float(t.start_point.y),
            float(t.start_point.x),
            float(t.end_point.y),
            float(t.end_point.x),
            float(t.start_time),
            float(t.end_time),
        ]
        lines.append("\t".join(fmt_num(v) for v in row))
    return lines

def fmt_num(x) -> str:
    """Print integers without .0, otherwise keep up to 3 decimals."""
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    return f"{xf:.3f}".rstrip("0").rstrip(".")

def write_row(f, row) -> None:
    f.write("\t".join(fmt_num(v) for v in row) + "\n")

def export_clusters_to_txt_instances(
    clusters: dict,
    original_instance,
    output_root: str,
    out_filename: str = "instance.txt"
):
    """
    Write each cluster as a standalone TXT instance readable by load_instance_from_txt.
    Preserves: header params, depots (2*K rows), chargers (F rows).
    Replaces: trips section with cluster trips only.
    """
    os.makedirs(output_root, exist_ok=True)

    # ---- Header parameters come from instance.meta (as used by your loader) ----
    meta = original_instance.meta

    # K = total vehicles across depots
    K = sum(d.vehicle_count for d in original_instance.depots)

    # F = number of charging stations/events
    F = len(original_instance.charging_stations)

    base_header = [
        int(K),
        0,  # T placeholder, set per cluster
        int(F),
        float(meta["lambda"]),
        float(meta["p_max"]),
        float(meta["p_min"]),
        float(meta["travel_cost"]),
        float(meta["charging_rate"]),
        float(meta["theta_factor"]),
    ]

    # ---- Build 2*K depot rows (origins then destinations), 7 columns like original ----
    # Format: depot_id, o_lat, o_lon, d_lat, d_lon, t_earliest, t_latest
    depot_rows = []
    for depot in original_instance.depots:
        for _ in range(depot.vehicle_count):
            lat = float(depot.location.y)
            lon = float(depot.location.x)

            # We do not have separate destination/time-window in the ProblemInstance.
            # To match the original TXT "look", we mirror origin->destination and use a wide window.
            depot_rows.append([int(depot.id), lat, lon, lat, lon, 0, 1440])

    depot_rows = depot_rows + depot_rows  # destinations identical

    # ---- Build charger rows (match your loader’s expectations: 7 columns) ----
    charger_rows = []
    for cs in original_instance.charging_stations:
        l, u = getattr(cs, "time_window", (0.0, 1e6))
        charger_rows.append([
            int(cs.id),
            float(cs.location.y),
            float(cs.location.x),
            0, 0,
            float(l),
            float(u)
        ])

    # ---- Write one TXT instance per cluster ----
    for cid, c_trips in clusters.items():
        cluster_dir = os.path.join(output_root, f"cluster_{cid}")
        os.makedirs(cluster_dir, exist_ok=True)

        trip_rows = []
        for new_tid, t in enumerate(c_trips, start=1):
            trip_rows.append([
                new_tid,
                float(t.start_point.y),
                float(t.start_point.x),
                float(t.end_point.y),
                float(t.end_point.x),
                float(t.start_time),
                float(t.end_time),
            ])

        header = base_header.copy()
        header[1] = int(len(trip_rows))  # T for this cluster

        out_path = os.path.join(cluster_dir, out_filename)

        with open(out_path, "w") as f:
            write_row(f, header)
            for r in depot_rows:
                write_row(f, r)
            for r in trip_rows:
                write_row(f, r)
            for r in charger_rows:
                write_row(f, r)

def minutes_to_hhmm(t_minutes: float) -> str:
    """Convert minutes-from-midnight to HHMM string used by GTFS trip_id parsing."""
    t = int(round(float(t_minutes)))
    t = max(0, min(t, 24 * 60 - 1))  # clamp into [0, 1439]
    hh = t // 60
    mm = t % 60
    return f"{hh:02d}{mm:02d}"


def build_trips_from_txt_instance(txt_path: str):
    instance = load_instance_from_txt(txt_path)

    trips = list(instance.trips)  # keep as-is, no GTFS metadata needed

    return instance, trips



def build_fake_gtfs_trips_txt(trips: list):
    """
    Build a GTFS-shaped trips.txt DataFrame containing ONLY the columns that
    export_clusters_to_gtfs() expects.
    """
    rows = []
    for t in trips:
        rows.append({
            "route_id": str(getattr(t, "route_id", 0)),
            "service_id": "TXT_SERVICE",
            "trip_id": str(t.gtfs_trip_id),
            "trip_headsign": "TXT_CLUSTER",
            "direction_id": int(getattr(t, "direction_id", 0)),
            "shape_id": int(getattr(t, "route_id", 0)),  # must be int for your exporter
        })

    return pd.DataFrame(rows)

def export_clusters_to_gtfs(clusters, trips_txt_df, output_root):
    """
    Export each cluster as a GTFS-compliant trips.txt file
    with EXACT column structure and ordering.
    """

    REQUIRED_COLUMNS = [
        "route_id",
        "service_id",
        "trip_id",
        "trip_headsign",
        "direction_id",
        "shape_id",
    ]

    os.makedirs(output_root, exist_ok=True)

    # --- Defensive copy ---
    trips_txt_df = trips_txt_df.copy()

    cluster_summaries = []

    # --- Ensure trip_id exists ---
    if "trip_id" not in trips_txt_df.columns:
        raise ValueError("GTFS trips.txt must contain 'trip_id' column")

    # --- Ensure required columns exist ---
    missing_cols = set(REQUIRED_COLUMNS) - set(trips_txt_df.columns)
    if missing_cols:
        raise ValueError(
            f"GTFS trips.txt missing required columns: {sorted(missing_cols)}"
        )

    # --- Normalize trip_id type ---
    trips_txt_df["trip_id"] = trips_txt_df["trip_id"].astype(str)

    # --- Index by trip_id for fast lookup ---
    trips_txt_df = trips_txt_df.set_index("trip_id", drop=False)

    for cid, c_trips in clusters.items():
        cluster_dir = os.path.join(output_root, f"cluster_{cid}")
        os.makedirs(cluster_dir, exist_ok=True)

        times = [t.start_time for t in c_trips]

        num_trips = len(c_trips)
        time_start = min(times)
        time_end = max(times)

        cluster_summaries.append(
                                    {
                                        "cluster_id": cid,
                                        "num_trips": num_trips,
                                        "time_start": time_start,
                                        "time_end": time_end,
                                    }
                                )

        # --- Collect GTFS trip_ids from cluster ---
        cluster_trip_ids = [str(t.gtfs_trip_id) for t in c_trips]

        # --- Validate existence ---
        missing = set(cluster_trip_ids) - set(trips_txt_df.index)
        if missing:
            raise ValueError(
                f"Cluster {cid}: trip_ids not found in GTFS trips.txt: "
                f"{list(missing)[:10]}"
            )

        # --- Subset rows ---
        cluster_df = trips_txt_df.loc[cluster_trip_ids]

        # --- Select and order columns EXACTLY ---
        cluster_df = cluster_df[REQUIRED_COLUMNS]

        # --- Force GTFS-compliant dtypes ---
        # shape_id must be integer (not float)
        cluster_df["shape_id"] = (
            cluster_df["shape_id"]
            .astype("Int64")      # pandas nullable int
            .astype(int)          # write as plain int
        )

        # --- Write GTFS trips.txt (NO index) ---
        cluster_df.to_csv(
            os.path.join(cluster_dir, "trips.txt"),
            index=False
        )

    summary_df = pd.DataFrame(cluster_summaries)

    summary_df.to_csv(
        os.path.join(output_root, "clusters_summary.csv"),
        index=False
    )

def split_large_cluster_temporal(trips, max_size=MAX_TRIPS_PER_CLUSTER):
    """
    Split a cluster into subclusters of size <= max_size,
    preserving temporal continuity.
    """
    trips_sorted = sorted(trips, key=lambda t: t.start_time)
    return [
        trips_sorted[i:i + max_size]
        for i in range(0, len(trips_sorted), max_size)
    ]

def plot_trip_clusters(trips, clusters):
    """
    Spatial visualization of spatio-temporal trip clusters.
    """

    trip_id_to_cluster = {}
    for cid, c_trips in clusters.items():
        for t in c_trips:
            trip_id_to_cluster[t.id] = cid

    # Prepare data
    lats = []
    lons = []
    times = []
    labels = []

    for trip in trips:
        lats.append(trip.start_point.y)
        lons.append(trip.start_point.x)
        times.append(trip.start_time)
        labels.append(trip_id_to_cluster.get(trip.id, -1))

    lats = np.array(lats)
    lons = np.array(lons)
    times = np.array(times)
    labels = np.array(labels)

    unique_labels = sorted(l for l in set(labels) if l != -1)
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(unique_labels))

    plt.figure(figsize=(10, 8))

    for idx, label in enumerate(unique_labels):
        mask = labels == label

        if label == -1:
            plt.scatter(
                lons[mask],
                lats[mask],
                c="black",
                marker="x",
                s=40,
                label="Noise"
            )
        else:
            plt.scatter(
                lons[mask],
                lats[mask],
                c=[cmap(idx)],
                s=40,
                label=f"Cluster {label}",
                alpha=0.75
            )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatio-Temporal Clustering of TXT Trips (Start Locations)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Helper: spatio-temporal distance
# ------------------------------------------------------------

def spatio_temporal_distance(a, b):
    """
    a, b:
    [start_lat, start_lon, end_lat, end_lon, start_time]
    """

    # Distance from end of one to start of the other (both directions)
    d1 = haversine.main(a[2], a[3], b[0], b[1])  # A.end -> B.start
    d2 = haversine.main(b[2], b[3], a[0], a[1])  # B.end -> A.start

    spatial = min(d1, d2) / SPATIAL_EPS_METERS
    
    gap_a_to_b = abs(b[4] - a[5])
    gap_b_to_a = abs(a[4] - b[5])

    temporal = min(gap_a_to_b, gap_b_to_a) / TEMPORAL_EPS_MIN

    return max(
        SPACE_WEIGHT * spatial,
        TIME_WEIGHT * temporal
    )

# ------------------------------------------------------------
# Build Trip objects (mirrors your loader logic)
# ------------------------------------------------------------

def build_trips_from_gtfs(
    route_id: int,
    day: str,
    gtfs_folder_path: str,
    trips_txt_df: pd.DataFrame,
    number_of_removed_stops: int = 0
):
    route_info = extract_gtfs_trips_data.process_gtfs_data(
        day=day,
        gtfs_folder_path=gtfs_folder_path,
        number_of_removed_stops=number_of_removed_stops
    )

    if route_id not in route_info:
        raise ValueError(f"Route {route_id} not found.")

    route_data = route_info[route_id]

    go_times = route_data[0]
    come_times = route_data[1]
    avg_go_time = route_data[2]
    avg_come_time = route_data[3]

    go_first = route_data[6]
    go_last = route_data[4]
    come_first = route_data[7]
    come_last = route_data[5]

    trips = []
    trip_id = 1

    # Filter GTFS trips for this route
    route_trips_df = trips_txt_df[trips_txt_df["route_id"].astype(str) == str(route_id)]

    if route_trips_df.empty:
        raise ValueError(f"No GTFS trips found for route {route_id}")

    route_trips_df = route_trips_df.sort_index()

    route_trip_ids = route_trips_df.index.tolist()

    needed = len(go_times) + len(come_times)
    if len(route_trip_ids) < needed:
        raise ValueError(
            f"Not enough trip_ids in trips.txt for route {route_id}. "
            f"Need {needed}, found {len(route_trip_ids)}."
        )

    # Split the route trip_ids into GO and COME groups
    go_trip_ids = route_trip_ids[:len(go_times)]
    come_trip_ids = route_trip_ids[len(go_times):len(go_times) + len(come_times)]

    def create_trips(times, first_coords, last_coords, avg_time, gtfs_trip_ids, direction_id):

        nonlocal trip_id, trips

        for i, start_time in enumerate(times):
            try:
                s_lat, s_lon = first_coords[i]
                e_lat, e_lon = last_coords[i]
            except IndexError:
                continue

            start = Point(s_lon, s_lat)
            end = Point(e_lon, e_lat)

            trip = Trip(
                id=trip_id,
                start_point=start,
                end_point=end,
                start_time=float(start_time),
                end_time=float(start_time + avg_time),
                trip_type="REGULAR"
            )

            trip.route_id = route_id
            trip.direction_id = direction_id
            trip.trip_length = avg_time
            gid = gtfs_trip_ids[i]

            if isinstance(gid, (list, tuple, np.ndarray, dict)):
                raise ValueError(
                    f"gtfs_trip_id must be scalar, got container type={type(gid)} at index {i}"
                )

            trip.gtfs_trip_id = str(gid)

            trips.append(trip)
            trip_id += 1

    create_trips(go_times, go_first, go_last, avg_go_time, go_trip_ids, direction_id=0)
    create_trips(come_times, come_first, come_last, avg_come_time, come_trip_ids, direction_id=1)

    return trips

def build_trips_for_routes(
    route_ids: list[int],
    day: str,
    gtfs_folder_path: str,
    trips_txt_df: pd.DataFrame,
    number_of_removed_stops: int = 0
):
    all_trips = []
    global_trip_id = 1

    for rid in route_ids:
        print(f"Loading trips for route {rid}...")

        trips = build_trips_from_gtfs(
            route_id=rid,
            day=day,
            gtfs_folder_path=gtfs_folder_path,
            trips_txt_df=trips_txt_df,
            number_of_removed_stops=number_of_removed_stops
        )

        # Ensure unique Trip.id across routes
        for t in trips:
            t.id = global_trip_id
            global_trip_id += 1

        all_trips.extend(trips)

    if not all_trips:
        raise ValueError("No trips built for any route.")

    return all_trips

# ------------------------------------------------------------
# Clustering logic
# ------------------------------------------------------------

def cluster_trips(trips):
    """
    Returns: dict {cluster_id: [Trip, ...]}
    """

    # Feature vector: [lat, lon, start_time]
    X = np.array([
        [
            trip.start_point.y,
            trip.start_point.x,
            trip.end_point.y,
            trip.end_point.x,
            trip.start_time,
            trip.end_time
        ]
        for trip in trips
    ])

    # Custom metric DBSCAN
    db = DBSCAN(
        eps=1.0,
        min_samples=MIN_TRIPS_PER_CLUSTER,
        metric=spatio_temporal_distance
    )

    labels = db.fit_predict(X)

    clusters = {}
    for label, trip in zip(labels, trips):
        clusters.setdefault(label, []).append(trip)

    final_clusters = {}
    new_id = 0

    for c_trips in clusters.values():
        if len(c_trips) <= MAX_TRIPS_PER_CLUSTER:
            final_clusters[new_id] = c_trips
            new_id += 1
        else:
            subclusters = split_large_cluster_temporal(c_trips)
            for sub in subclusters:
                final_clusters[new_id] = sub
                new_id += 1

    return final_clusters

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    synthetic_instance = "D2_S3_C30_e_trips.txt"

    output_base = os.path.join(project_root, "..", "output", "clusters_synthetic", synthetic_instance)

    txt_instance_path = os.path.join(project_root, "..", "input", "test_instances", synthetic_instance)
    # ^^^ Change this path to your actual .txt instance file

    print("Loading original TXT instance...")
    instance, trips = build_trips_from_txt_instance(txt_instance_path)
    print(f"Trips loaded from TXT: {len(trips)}")

    print(f"Total trips across all routes: {len(trips)}")

    print("Clustering trips (spatio-temporal)...")
    clusters = cluster_trips(trips)

    print(f"Clusters found: {len(clusters)}")

    for cid, c_trips in clusters.items():
        times = [t.start_time for t in c_trips]
        print(
            f"Cluster {cid}: "
            f"{len(c_trips)} trips | "
            f"Time span [{min(times):.1f}, {max(times):.1f}]"
        )

    print("Visualizing clusters...")
    plot_trip_clusters(trips, clusters)

    if os.path.exists(output_base):
        shutil.rmtree(output_base)

    export_clusters_to_txt_instances_verbatim(
        clusters=clusters,
        source_txt_path=txt_instance_path,
        output_root=output_base,
        out_filename="instance.txt"
    )
