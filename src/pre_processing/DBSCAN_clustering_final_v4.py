
# National Technical University of Athens
# Railways & Transport Lab
#
# DBSCAN clustering with CHAIN-FEASIBLE neighborhoods:
# - A and B are neighbors only if there exists a chronological order (A->B or B->A)
#   such that:
#     (1) deadhead time gap (next.start - prev.end) is NONNEGATIVE and <= TEMPORAL_EPS_MIN (after normalization)
#     (2) deadhead distance (prev.end -> next.start) <= SPATIAL_EPS_METERS (after normalization)
# - IMPORTANT: time and distance are evaluated in the SAME chosen direction (no mixing).
#
# This makes clusters much more "chain-y" than using abs() gaps or min() across directions independently.
#
# Output:
# - clusters are exported as GTFS trips.txt subsets
# - clusters_summary.csv contains size and time span

import os
import sys
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

# --- Add parent directory to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities.instance_generator import Trip
from utilities import haversine
from data_processing import extract_gtfs_trips_data

# ------------------------------------------------------------
# Configuration (tune here, not in logic)
# ------------------------------------------------------------

# Hard feasibility thresholds for "neighbor" relation
SPATIAL_EPS_METERS = 50.0      # max deadhead distance between consecutive trips (meters)
TEMPORAL_EPS_MIN   = 70.0       # max deadhead time gap between consecutive trips (minutes)

# DBSCAN parameters
DBSCAN_EPS = 0.9684                # we normalize so "feasible neighbor" tends to be <= 1
MIN_TRIPS_PER_CLUSTER = 3     # 2 => each core trip needs at least 1 neighbor (itself counts)

# Optional: cap cluster size for MILP tractability (split by time if exceeded)
MAX_TRIPS_PER_CLUSTER = 20

# NOTE about weights:
# Using max() below makes constraints "hard-ish". We keep equal weights by default,
# but weights effectively scale your thresholds: if SPACE_WEIGHT=0.5, you allow ~2x distance.
SPACE_WEIGHT = 0.7579
TIME_WEIGHT  = 0.4

# If True, overlapping pairs (no feasible order) are treated as incompatible (distance huge).
REQUIRE_NONNEGATIVE_GAP = False

START_TIME_WINDOW_MIN = 540.0   # Classify day per XX minutes

FIRST_TRIP_INDEX = 0
MAX_TRIPS_PER_ROUTE_PER_DIRECTION = 26


# Route-aware clustering: trips from different routes are treated as incompatible.
ROUTE_MISMATCH_PENALTY = 1e9
# ------------------------------------------------------------
# GTFS time helpers
# ------------------------------------------------------------

def gtfs_time_to_minutes(t: str) -> float:
    """Convert 'HH:MM:SS' (can exceed 24h) to minutes from midnight."""
    if pd.isna(t):
        return np.nan
    t = str(t).strip()
    if not t:
        return np.nan
    try:
        hh, mm, ss = t.split(":")
        return int(hh) * 60 + int(mm) + int(ss) / 60.0
    except Exception:
        return np.nan


def _load_stop_times(gtfs_folder_path: str) -> pd.DataFrame:
    stop_times_path = os.path.join(gtfs_folder_path, "stop_times.txt")
    if not os.path.exists(stop_times_path):
        raise FileNotFoundError(f"stop_times.txt not found at: {stop_times_path}")

    st = pd.read_csv(stop_times_path, dtype={"trip_id": str})

    required = {"trip_id", "stop_sequence", "arrival_time", "departure_time"}
    missing = required - set(st.columns)
    if missing:
        raise ValueError(f"stop_times.txt missing columns: {sorted(missing)}")

    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    return st


def load_trip_start_times_from_stop_times(gtfs_folder_path: str) -> dict[str, float]:
    """trip_id -> start_time_minutes (FIRST stop by min stop_sequence)."""
    st = _load_stop_times(gtfs_folder_path)
    idx = st.groupby("trip_id")["stop_sequence"].idxmin()
    first_rows = st.loc[idx].copy()

    start_time_str = first_rows["departure_time"].where(
        first_rows["departure_time"].notna() & (first_rows["departure_time"].astype(str).str.strip() != ""),
        first_rows["arrival_time"]
    )

    first_rows["start_time_min"] = start_time_str.apply(gtfs_time_to_minutes)
    first_rows = first_rows.dropna(subset=["start_time_min"])
    return dict(zip(first_rows["trip_id"].astype(str), first_rows["start_time_min"].astype(float)))


def load_trip_end_times_from_stop_times(gtfs_folder_path: str) -> dict[str, float]:
    """trip_id -> end_time_minutes (LAST stop by max stop_sequence)."""
    st = _load_stop_times(gtfs_folder_path)
    idx = st.groupby("trip_id")["stop_sequence"].idxmax()
    last_rows = st.loc[idx].copy()

    end_time_str = last_rows["departure_time"].where(
        last_rows["departure_time"].notna() & (last_rows["departure_time"].astype(str).str.strip() != ""),
        last_rows["arrival_time"]
    )

    last_rows["end_time_min"] = end_time_str.apply(gtfs_time_to_minutes)
    last_rows = last_rows.dropna(subset=["end_time_min"])
    return dict(zip(last_rows["trip_id"].astype(str), last_rows["end_time_min"].astype(float)))


# ------------------------------------------------------------
# Pruning
# ------------------------------------------------------------

def prune_trips_for_clustering(trips, max_trips_per_route_per_direction=6, first_trip_index=0):
    """
    Mimics prior pruning:
    - group by route_id
    - then by direction_id
    - sort by start_time
    - slice fixed number of trips
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for t in trips:
        grouped[t.route_id][t.direction_id].append(t)

    pruned = []
    for _, dir_dict in grouped.items():
        for _, d_trips in dir_dict.items():
            d_trips_sorted = sorted(d_trips, key=lambda t: t.start_time)
            pruned.extend(d_trips_sorted[first_trip_index:first_trip_index + max_trips_per_route_per_direction])

    if not pruned:
        raise ValueError("Pruning removed all trips. Check parameters.")
    return pruned


# ------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------

def export_clusters_to_gtfs(clusters, trips_txt_df, output_root):
    """Export each cluster to a folder with GTFS trips.txt + write clusters_summary.csv."""
    REQUIRED_COLUMNS = ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]

    os.makedirs(output_root, exist_ok=True)
    trips_txt_df = trips_txt_df.copy()

    if "trip_id" not in trips_txt_df.columns:
        raise ValueError("GTFS trips.txt must contain 'trip_id' column")

    missing_cols = set(REQUIRED_COLUMNS) - set(trips_txt_df.columns)
    if missing_cols:
        raise ValueError(f"GTFS trips.txt missing required columns: {sorted(missing_cols)}")

    trips_txt_df["trip_id"] = trips_txt_df["trip_id"].astype(str)
    trips_txt_df = trips_txt_df.set_index("trip_id", drop=False)

    summaries = []
    for cid, c_trips in clusters.items():
        cluster_dir = os.path.join(output_root, f"cluster_{cid}")
        os.makedirs(cluster_dir, exist_ok=True)

        starts = [t.start_time for t in c_trips]
        ends = [t.end_time for t in c_trips]
        summaries.append({
            "cluster_id": cid,
            "num_trips": len(c_trips),
            "time_start": float(min(starts)),
            "time_end": float(max(ends)),
        })

        cluster_trip_ids = [str(t.gtfs_trip_id) for t in c_trips]
        missing = set(cluster_trip_ids) - set(trips_txt_df.index)
        if missing:
            raise ValueError(f"Cluster {cid}: trip_ids not found in trips.txt: {list(missing)[:10]}")

        cluster_df = trips_txt_df.loc[cluster_trip_ids][REQUIRED_COLUMNS]
        cluster_df["shape_id"] = cluster_df["shape_id"].astype("Int64").astype(int)
        cluster_df.to_csv(os.path.join(cluster_dir, "trips.txt"), index=False)

    pd.DataFrame(summaries).to_csv(os.path.join(output_root, "clusters_summary.csv"), index=False)


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_trip_clusters(trips, clusters):
    """Plot trip start points colored by cluster id (for quick sanity checks)."""
    trip_id_to_cluster = {}
    for cid, c_trips in clusters.items():
        for t in c_trips:
            trip_id_to_cluster[t.id] = cid

    lats, lons, labels = [], [], []
    for t in trips:
        lats.append(t.start_point.y)
        lons.append(t.start_point.x)
        labels.append(trip_id_to_cluster.get(t.id, -1))

    lats = np.array(lats)
    lons = np.array(lons)
    labels = np.array(labels)

    unique_labels = sorted(set(labels))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(1, len(unique_labels)))

    plt.figure(figsize=(10, 8))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(lons[mask], lats[mask], c=[cmap(idx)], s=40, alpha=0.75, label=f"Cluster {label}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("DBSCAN Clusters (Chain-feasible neighbors)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Chain-feasible DBSCAN metric
# ------------------------------------------------------------

def spatio_temporal_distance_with_weights(a, b, space_w: float, time_w: float):
    """Chain-feasible distance used by DBSCAN.

    a, b are arrays:
      [start_lat, start_lon, end_lat, end_lon, start_time, end_time, route_code]

    We enforce "chain-feasible neighbor":
      choose direction (A->B or B->A) that is chronologically feasible (nonnegative gap)
      and has minimal gap. Then compute:
        temporal = gap / TEMPORAL_EPS_MIN
        spatial  = dist(prev.end -> next.start) / SPATIAL_EPS_METERS

      return max(time_w * temporal, space_w * spatial)

    For eps=1.0, neighbor means:
      temporal <= 1 and spatial <= 1 (up to weight scaling).
    """
    # directed gaps
    gap_ab = b[4] - a[5]  # B.start - A.end   (A -> B)
    gap_ba = a[4] - b[5]  # A.start - B.end   (B -> A)

    TIME_BIN_MIN = START_TIME_WINDOW_MIN  # or set explicitly


    # Route-aware similarity: if route codes differ, treat as incompatible.
    # This works only if the feature vector includes a route code at index 6.
    if len(a) > 6 and len(b) > 6 and int(a[6]) != int(b[6]):
        return ROUTE_MISMATCH_PENALTY

    if int(a[4] // TIME_BIN_MIN) != int(b[4] // TIME_BIN_MIN):
        return 1e9

    # directed distances
    d_ab = haversine.main(a[2], a[3], b[0], b[1])  # A.end -> B.start
    d_ba = haversine.main(b[2], b[3], a[0], a[1])  # B.end -> A.start

    if REQUIRE_NONNEGATIVE_GAP:
        candidates = []
        if gap_ab >= 0:
            candidates.append(("ab", gap_ab, d_ab))
        if gap_ba >= 0:
            candidates.append(("ba", gap_ba, d_ba))
        if not candidates:
            return 1e9  # overlap both ways => cannot chain
        direction, best_gap, best_dist = min(candidates, key=lambda x: x[1])
    else:
        # If you ever set this false, you allow overlaps but still choose the "closest" direction.
        options = [("ab", abs(gap_ab), d_ab), ("ba", abs(gap_ba), d_ba)]
        direction, best_gap, best_dist = min(options, key=lambda x: x[1])

    temporal = best_gap / TEMPORAL_EPS_MIN
    spatial = best_dist / SPATIAL_EPS_METERS

    return max(time_w * temporal, space_w * spatial)


def spatio_temporal_distance(a, b):
    """Default metric for the *initial* global clustering pass."""
    return spatio_temporal_distance_with_weights(a, b, SPACE_WEIGHT, TIME_WEIGHT)



# ------------------------------------------------------------
# Build Trip objects
# ------------------------------------------------------------

def build_trips_from_gtfs(
    route_id: int,
    day: str,
    gtfs_folder_path: str,
    trips_txt_df: pd.DataFrame,
    trip_id_to_start_time: dict[str, float],
    trip_id_to_end_time: dict[str, float],
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
    local_trip_id = 1

    route_trips_df = trips_txt_df[trips_txt_df["route_id"].astype(str) == str(route_id)]
    if route_trips_df.empty:
        raise ValueError(f"No GTFS trips found for route {route_id}")

    route_trips_df = route_trips_df.sort_index()
    route_trip_ids = route_trips_df.index.tolist()

    needed = len(go_times) + len(come_times)
    if len(route_trip_ids) < needed:
        raise ValueError(
            f"Not enough trip_ids in trips.txt for route {route_id}. Need {needed}, found {len(route_trip_ids)}."
        )

    go_trip_ids = route_trip_ids[:len(go_times)]
    come_trip_ids = route_trip_ids[len(go_times):len(go_times) + len(come_times)]

    def create_trips(times, first_coords, last_coords, avg_time, gtfs_trip_ids, direction_id):
        nonlocal local_trip_id, trips
        for i, fallback_start in enumerate(times):
            try:
                s_lat, s_lon = first_coords[i]
                e_lat, e_lon = last_coords[i]
            except IndexError:
                continue

            gid = str(gtfs_trip_ids[i])

            # Prefer stop_times for start/end; fallback to provided values
            st = trip_id_to_start_time.get(gid, None)
            et = trip_id_to_end_time.get(gid, None)

            start_time_val = float(st) if st is not None else float(fallback_start)
            end_time_val = float(et) if et is not None else float(start_time_val + avg_time)

            start = Point(s_lon, s_lat)
            end = Point(e_lon, e_lat)

            trip = Trip(
                id=local_trip_id,
                start_point=start,
                end_point=end,
                start_time=start_time_val,
                end_time=end_time_val,
                trip_type="REGULAR"
            )
            trip.route_id = route_id
            trip.direction_id = direction_id
            trip.trip_length = float(end_time_val - start_time_val)
            trip.gtfs_trip_id = gid

            trips.append(trip)
            local_trip_id += 1

    create_trips(go_times, go_first, go_last, avg_go_time, go_trip_ids, direction_id=0)
    create_trips(come_times, come_first, come_last, avg_come_time, come_trip_ids, direction_id=1)

    return trips


def build_trips_for_routes(
    route_ids: list[int],
    day: str,
    gtfs_folder_path: str,
    trips_txt_df: pd.DataFrame,
    trip_id_to_start_time: dict[str, float],
    trip_id_to_end_time: dict[str, float],
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
            trip_id_to_start_time=trip_id_to_start_time,
            trip_id_to_end_time=trip_id_to_end_time,
            number_of_removed_stops=number_of_removed_stops
        )

        # ensure unique Trip.id across routes
        for t in trips:
            t.id = global_trip_id
            global_trip_id += 1

        all_trips.extend(trips)

    if not all_trips:
        raise ValueError("No trips built for any route.")
    return all_trips

# ------------------------------------------------------------
# Cluster size helper
# ------------------------------------------------------------

def split_oversized_cluster_by_time(trips, max_size):
    """Deterministic fallback: split a list of trips into chunks by start_time."""
    c_sorted = sorted(trips, key=lambda t: t.start_time)
    return [c_sorted[i:i + max_size] for i in range(0, len(c_sorted), max_size)]


# ------------------------------------------------------------
# Clustering logic
# ------------------------------------------------------------

def _dbscan_partition(trips, metric_func=spatio_temporal_distance, verbose=True):
    """One DBSCAN pass. Returns a list[list[Trip]]. Keeps noise as singletons."""
    if not trips:
        return []

    X = np.array([
        [t.start_point.y, t.start_point.x,
         t.end_point.y, t.end_point.x,
         t.start_time, t.end_time,
         float(getattr(t, "route_id", -1))]
        for t in trips
    ])
    db = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=MIN_TRIPS_PER_CLUSTER,
        metric=metric_func
    )

    labels = db.fit_predict(X)
    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        print("DBSCAN label counts:", dict(zip(unique, counts)))

    by_label = defaultdict(list)
    noise = []
    for lab, t in zip(labels, trips):
        if lab == -1:
            noise.append(t)
        else:
            by_label[lab].append(t)

    parts = []
    for lab in sorted(by_label.keys()):
        parts.append(by_label[lab])
    for t in noise:
        parts.append([t])
    return parts


def cluster_trips(trips, max_size=MAX_TRIPS_PER_CLUSTER, max_depth=12):
    """Run a single DBSCAN pass using the chain-feasible metric.

    Subclustering has been removed. We still enforce the optional `max_size`
    cap by splitting oversized clusters deterministically by start_time.
    `max_depth` is kept only for backwards compatibility (unused).
    """
    parts = _dbscan_partition(trips, metric_func=spatio_temporal_distance, verbose=True)

    clusters = {}
    cid = 0
    for p in parts:
        if len(p) <= max_size:
            clusters[cid] = p
            cid += 1
        else:
            for chunk in split_oversized_cluster_by_time(p, max_size):
                clusters[cid] = chunk
                cid += 1
    return clusters

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    route_ids = [1033, 1034, 874, 871]

    day = "monday"
    gtfs_folder_path = os.path.join(project_root, "..", "input", "gtfs", "oasa_december_2025")
    output_base = os.path.join(project_root, "..", "output", "clusters_full")

    trips_txt_path = os.path.join(gtfs_folder_path, "trips.txt")
    trips_txt_df = pd.read_csv(trips_txt_path)
    trips_txt_df.set_index("trip_id", inplace=True, drop=False)

    print("Loading stop_times start/end times...")
    trip_id_to_start_time = load_trip_start_times_from_stop_times(gtfs_folder_path)
    trip_id_to_end_time = load_trip_end_times_from_stop_times(gtfs_folder_path)
    print(f"Loaded start times for {len(trip_id_to_start_time)} trip_ids.")
    print(f"Loaded end times   for {len(trip_id_to_end_time)} trip_ids.")

    print("Loading trips from GTFS...")
    trips = build_trips_for_routes(
        route_ids=route_ids,
        day=day,
        gtfs_folder_path=gtfs_folder_path,
        trips_txt_df=trips_txt_df,
        trip_id_to_start_time=trip_id_to_start_time,
        trip_id_to_end_time=trip_id_to_end_time
    )

    print(f"Trips before pruning: {len(trips)}")
    trips = prune_trips_for_clustering(
        trips,
        max_trips_per_route_per_direction=MAX_TRIPS_PER_ROUTE_PER_DIRECTION,
        first_trip_index=FIRST_TRIP_INDEX
    )
    print(f"Trips after pruning: {len(trips)}")
    print(f"Total trips across all routes: {len(trips)}")

    # Sanity check: durations
    bad = [t for t in trips if (t.end_time - t.start_time) < 0 or (t.end_time - t.start_time) > 300]
    print("Trips with suspicious duration (<0 or >300 min):", len(bad))
    for t in bad[:10]:
        print("trip_id", t.gtfs_trip_id, "start", t.start_time, "end", t.end_time, "dur", t.end_time - t.start_time)

    print("Clustering trips (DBSCAN, chain-feasible metric)...")
    clusters = cluster_trips(trips)

    print(f"Clusters produced (including singletons for noise): {len(clusters)}")
    for cid, c_trips in clusters.items():
        starts = [t.start_time for t in c_trips]
        ends = [t.end_time for t in c_trips]
        print(f"Cluster {cid}: {len(c_trips)} trips | Time span [{min(starts):.1f}, {max(ends):.1f}]")

    print("Visualizing clusters...")
    plot_trip_clusters(trips, clusters)

    # Export final clusters into the original output folder (no extra subfolders)
    print(f"Exporting final clusters into: {output_base}")
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    export_clusters_to_gtfs(
        clusters=clusters,
        trips_txt_df=trips_txt_df,
        output_root=output_base
    )
