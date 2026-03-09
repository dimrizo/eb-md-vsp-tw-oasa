# National Technical University of Athens
# Railways & Transport Lab
# Dimitrios Rizopoulos, Konstantinos Gkiotsalitis

import os
import sys
import datetime
import random
import json

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# --- Add parent directory to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities.instance_generator import Point, Depot, Trip, ChargingStation, ProblemInstance
from utilities import haversine
from data_processing import extract_gtfs_trips_data
from data_processing.read_file import read_comma_delimited_file
import plotting.plotting_utils_eb as plotting_utils_eb

# --- Constants for the Energy Buffer (EB) Model Extension ---
SMALL_M = 0.00001
BIG_M = 1000000

# --- Domain-specific constants (GTFS-based instance) ---
# Average deadhead speed: 26,000 m / 60 min ≈ 433.33 m/min (~26 km/h)
AVG_SPEED_M_PER_MIN = 26000.0 / 60.0

# Energy model & charging parameters (replacing old header-based ones)
PHI_MAX = 350.0                         # Max SOC / Energy level
PHI_MIN = 100.0                         # Min SOC / Energy level
CHARGING_RATE_KWH_PER_MINUTE = 3.0
THETA_FACTOR = 0.002 # 0.00072          # Energy per meter
TRAVEL_COST = 3.0                      # Cost per minute of travel
TIME_WINDOW_SLACK = 10

# ============================================================
# Instance Loader from GTFS
# ============================================================

def start_time_from_gtfs_trip_id(gtfs_trip_id: str) -> float:
    """
    Extract start time in minutes from GTFS trip_id ending in _HHMM.
    Example: 1033_day_1_1891_0630 -> 390
    """
    hhmm = gtfs_trip_id.strip().split("_")[-1]
    if len(hhmm) != 4 or not hhmm.isdigit():
        raise ValueError(f"Cannot parse start time from trip_id '{gtfs_trip_id}'")
    h = int(hhmm[:2])
    m = int(hhmm[2:])
    return h * 60 + m

def load_instance_from_gtfs_cluster(
    cluster_trips_txt_path: str,
    day: str,
    gtfs_folder_path: str,
    depot_filepath: str,
    buses_per_depot: list[int],
    buses_availability_times: list[float],
    buses_SoC: list[float],
    number_of_CS_per_depot: int = 1,
    number_of_removed_stops: int = 0,
    buses_state: list[bool] = None
) -> ProblemInstance:
    cluster_df = pd.read_csv(cluster_trips_txt_path)

    gtfs_trips_path = os.path.join(gtfs_folder_path, "trips.txt")
    gtfs_trips_df = pd.read_csv(gtfs_trips_path, dtype={"trip_id": str, "route_id": str})
    gtfs_trips_df = gtfs_trips_df.set_index("trip_id", drop=False)

    if "trip_id" not in cluster_df.columns:
        raise ValueError("Cluster trips.txt must contain 'trip_id' column")

    cluster_trip_ids = {
        str(t).strip() for t in cluster_df["trip_id"].astype(str)
    }

    if not cluster_trip_ids:
        raise ValueError("Cluster trips.txt is empty")
    
    route_info = extract_gtfs_trips_data.process_gtfs_data(
        day=day,
        gtfs_folder_path=gtfs_folder_path,
        number_of_removed_stops=number_of_removed_stops
    )

    trips = []
    relief_points = []
    
    trip_counter = 1

    print("Sample cluster trip_ids:")
    print(list(cluster_trip_ids)[:5])

    for route_id, route_data in route_info.items():

        go_times        = route_data[0]
        come_times      = route_data[1]
        avg_go_time     = route_data[2]
        avg_come_time   = route_data[3]
        go_last         = route_data[4]
        come_last       = route_data[5]

        # Correct GTFS trip IDs per route
        route_trips = gtfs_trips_df[gtfs_trips_df["route_id"].astype(str) == str(route_id)]

        go_trip_ids = route_trips[route_trips["direction_id"] == 0]["trip_id"].tolist()
        come_trip_ids = route_trips[route_trips["direction_id"] == 1]["trip_id"].tolist()

        go_first        = route_data[6]
        come_first      = route_data[7]

        def maybe_add_trip(gtfs_trip_id, first_coord, last_coord, avg_time):
            nonlocal trip_counter

            gtfs_trip_id = str(gtfs_trip_id).strip()

            if gtfs_trip_id not in cluster_trip_ids:
                return

            start_time = start_time_from_gtfs_trip_id(gtfs_trip_id)

            if trip_counter <= 5:
                print("GTFS extractor trip_id:", gtfs_trip_id)

            start_lat, start_lon = first_coord
            end_lat, end_lon = last_coord

            start_point = Point(start_lon, start_lat)
            end_point = Point(end_lon, end_lat)

            trip = Trip(
                id=trip_counter,
                start_point=start_point,
                end_point=end_point,
                start_time=float(start_time),
                end_time=float(start_time + avg_time),
                trip_type="REGULAR"
            )

            trip.trip_length = avg_time
            trip.start_time_window = (
                float(start_time) - TIME_WINDOW_SLACK,
                float(start_time) + TIME_WINDOW_SLACK
            )

            dist_m = haversine.main(start_lat, start_lon, end_lat, end_lon)
            trip.eta = dist_m * THETA_FACTOR

            trip.gtfs_trip_id = str(gtfs_trip_id)
            trip.route_id = route_id  # metadata only

            trips.append(trip)
            relief_points.extend([start_point, end_point])

            trip_counter += 1

        for gtfs_trip_id, first, last in zip(go_trip_ids, go_first, go_last):
            maybe_add_trip(
                gtfs_trip_id,
                first,
                last,
                avg_go_time
            )

        for gtfs_trip_id, first, last in zip(come_trip_ids, come_first, come_last):
            maybe_add_trip(
                gtfs_trip_id,
                first,
                last,
                avg_come_time
            )

    if not trips:
        raise ValueError("No GTFS trips matched the selected cluster")
    
    # ------------------------------------------------------------------
    # 2) Read depot coordinates
    # ------------------------------------------------------------------

    df = read_comma_delimited_file(depot_filepath)
    if df is None or df.empty:
        raise ValueError("Depot file could not be read or is empty.")
    
    # ---------------------------------------------------------------
    # Fleet configuration validation
    # ---------------------------------------------------------------

    if buses_per_depot is None or buses_availability_times is None or buses_SoC is None:
        raise ValueError("Fleet configuration inputs must be provided explicitly.")

    number_of_depots = len(buses_per_depot)

    if len(df) != number_of_depots:
        raise ValueError(
            f"Depot count mismatch: depots.txt has {len(df)} rows, "
            f"but buses_per_depot has length {number_of_depots}"
        )

    total_buses = sum(buses_per_depot)

    if len(buses_availability_times) != total_buses:
        raise ValueError(
            "Length of buses_availability_times must equal total number of buses"
        )

    if len(buses_SoC) != total_buses:
        raise ValueError(
            "Length of buses_SoC must equal total number of buses"
        )

    number_of_depots = len(df)

    K = sum(buses_per_depot)
    F_chargers = len(buses_per_depot) * number_of_CS_per_depot

    required_columns = {"lon", "lat"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Depot file must contain columns: {required_columns}")

    depot_points = [
        Point(row["lon"], row["lat"])
        for _, row in df.iterrows()
    ]

    if len(depot_points) < 2:
        raise ValueError("At least two depots are required.")

    # Use exactly two depots
    depot_points = depot_points[:2]

    # ------------------------------------------------------------------
    # 3) Instantiate depots and charging stations
    # ------------------------------------------------------------------

    depots = []
    charging_stations = []

    cs_id = 1

    for depot_id, (point, bus_count) in enumerate(zip(depot_points, buses_per_depot), start=1):

        depots.append(
            Depot(
                id=depot_id,
                location=point,
                vehicle_count=bus_count
            )
        )

        for _ in range(number_of_CS_per_depot):
            charging_stations.append(
                ChargingStation(
                    id=cs_id,
                    location=point,
                    time_window=(0.0, BIG_M)
                )
            )
            cs_id += 1

    # ------------------------------------------------------------------
    # 4) Build ProblemInstance and meta info
    # ------------------------------------------------------------------
    grid_size = (60, 60)  # irrelevant for GTFS geometry but required by class

    instance = ProblemInstance(
        grid_size=grid_size,
        trips=trips,
        depots=depots,
        relief_points=relief_points,
        charging_stations=charging_stations
    )

    greek_l = 2.0  # keep as generic waiting/penalty parameter

    instance.meta = {
        "K": K,
        "T": len(trips),
        "F": F_chargers,
        "lambda": greek_l,
        "p_max": PHI_MAX,
        "p_min": PHI_MIN,
        "travel_cost": TRAVEL_COST,
        "charging_rate": CHARGING_RATE_KWH_PER_MINUTE,
        "theta_factor": THETA_FACTOR,
        "cluster_trips_file": os.path.basename(cluster_trips_txt_path),
        "day": day
    }

    instance.meta.update({
        "buses_per_depot": buses_per_depot,
        "buses_availability_times": buses_availability_times,
        "buses_initial_soc": buses_SoC,
        "number_of_cs_per_depot": number_of_CS_per_depot,
        "buses_state": buses_state
    })

    return instance

def load_instance_from_gtfs(
    route_id: int,
    day: str,
    gtfs_folder_path: str,
    depot_filepath: str,
    number_of_removed_stops: int = 0,
    first_go_trip: int = 0,
    num_go_trips: int = None,
    first_come_trip: int = 0,
    num_come_trips: int = None,
    buses_per_depot: list[int] = None,
    buses_availability_times: list[float] = None,
    buses_SoC: list[float] = None,
    number_of_CS_per_depot: int = 1
) -> ProblemInstance:

    """
    Build a ProblemInstance from GTFS for a single route and both directions.

    - Each GTFS trip (departure) becomes a Trip object.
    - Trip duration = average travel time per direction (from GTFS extractor).
    - Trip energy η is computed from haversine distance between first/last stops.
    """

    # ------------------------------------------------------------------
    # 1) Extract GTFS data for all routes and pick the desired route
    # ------------------------------------------------------------------
    route_info = extract_gtfs_trips_data.process_gtfs_data(
        day=day,
        gtfs_folder_path=gtfs_folder_path,
        number_of_removed_stops=number_of_removed_stops
    )

    if route_id not in route_info:
        raise ValueError(f"Route ID {route_id} not found in GTFS data.")

    # route_data structure (per extract_gtfs_trips_data):
    route_data = route_info[route_id]

    go_times = route_data[0]
    come_times = route_data[1]
    avg_go_travel_time = route_data[2]
    avg_come_travel_time = route_data[3]

    go_last_stops_coords = route_data[4]
    come_last_stops_coords = route_data[5]
    go_first_stops_coords = route_data[6]
    come_first_stops_coords = route_data[7]

    # NEW: full stop sequences for each trip (coords)
    go_trip_stop_coords = route_data[8]
    come_trip_stop_coords = route_data[9]

    # Trip pruning logic (mirrors attached file)
    def prune(lst, start, n):
        if n is None:
            return lst[start:]
        return lst[start:start + n]

    go_times = prune(go_times, first_go_trip, num_go_trips)
    come_times = prune(come_times, first_come_trip, num_come_trips)

    go_last_stops_coords = prune(go_last_stops_coords, first_go_trip, num_go_trips)
    go_first_stops_coords = prune(go_first_stops_coords, first_go_trip, num_go_trips)
    go_trip_stop_coords = prune(go_trip_stop_coords, first_go_trip, num_go_trips)

    come_last_stops_coords = prune(come_last_stops_coords, first_come_trip, num_come_trips)
    come_first_stops_coords = prune(come_first_stops_coords, first_come_trip, num_come_trips)
    come_trip_stop_coords = prune(come_trip_stop_coords, first_come_trip, num_come_trips)

    if not go_times and not come_times:
        raise ValueError(f"Route {route_id} has no trips for day '{day}'.")

    # ------------------------------------------------------------------
    # 2) Define depot and charging stations (all at same coordinate)
    # ------------------------------------------------------------------

    number_of_vehicles_per_depot = 4
    number_of_cs_per_depot = 1

    # ------------------------------------------------------------------
    # 2) Read depot coordinates
    # ------------------------------------------------------------------

    df = read_comma_delimited_file(depot_filepath)
    if df is None or df.empty:
        raise ValueError("Depot file could not be read or is empty.")
    
    # ---------------------------------------------------------------
    # Fleet configuration validation
    # ---------------------------------------------------------------

    if buses_per_depot is None or buses_availability_times is None or buses_SoC is None:
        raise ValueError("Fleet configuration inputs must be provided explicitly.")

    number_of_depots = len(buses_per_depot)

    if len(df) != number_of_depots:
        raise ValueError(
            f"Depot count mismatch: depots.txt has {len(df)} rows, "
            f"but buses_per_depot has length {number_of_depots}"
        )

    total_buses = sum(buses_per_depot)

    if len(buses_availability_times) != total_buses:
        raise ValueError(
            "Length of buses_availability_times must equal total number of buses"
        )

    if len(buses_SoC) != total_buses:
        raise ValueError(
            "Length of buses_SoC must equal total number of buses"
        )

    number_of_depots = len(df)

    K = number_of_depots * number_of_vehicles_per_depot
    F_chargers = number_of_depots * number_of_cs_per_depot
    
    required_columns = {"lon", "lat"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Depot file must contain columns: {required_columns}")

    depot_points = [
        Point(row["lon"], row["lat"])
        for _, row in df.iterrows()
    ]

    if len(depot_points) < 2:
        raise ValueError("At least two depots are required.")

    # Use exactly two depots
    depot_points = depot_points[:2]

    # ------------------------------------------------------------------
    # 3) Instantiate depots and charging stations
    # ------------------------------------------------------------------

    depots = []
    charging_stations = []

    cs_id = 1

    for depot_id, (point, bus_count) in enumerate(zip(depot_points, buses_per_depot), start=1):

        depots.append(
            Depot(
                id=depot_id,
                location=point,
                vehicle_count=bus_count
            )
        )

        for _ in range(number_of_CS_per_depot):
            charging_stations.append(
                ChargingStation(
                    id=cs_id,
                    location=point,
                    time_window=(0.0, BIG_M)
                )
            )
            cs_id += 1

    # ------------------------------------------------------------------
    # 3) Build Trip objects, one per GTFS departure in both directions
    # ------------------------------------------------------------------
    trips = []
    relief_points = depot_points

    trip_counter = 1

    def build_trips_for_direction(
        start_times,
        first_coords_list,
        last_coords_list,
        avg_travel_time,
        stop_sequences_coords   # NEW: list of [[lat,lon], ...] per trip
    ):
        nonlocal trip_counter, trips, relief_points

        for idx, start_minutes in enumerate(start_times):
            try:
                start_lat, start_lon = first_coords_list[idx]
                end_lat, end_lon = last_coords_list[idx]
                stops_coords = stop_sequences_coords[idx]
            except IndexError:
                # misalignment paranoia
                continue

            if len(stops_coords) < 2:
                # can't compute segment distances with <2 stops
                continue

            start_point = Point(start_lon, start_lat)
            end_point = Point(end_lon, end_lat)

            relief_points.append(start_point)
            relief_points.append(end_point)

            trip_length = float(avg_travel_time)

            start_time = float(start_minutes)
            end_time = start_time + trip_length

            # NEW: segment-wise distance -> eta
            total_dist_m = 0.0
            for i in range(len(stops_coords) - 1):
                lat1, lon1 = stops_coords[i]
                lat2, lon2 = stops_coords[i + 1]
                total_dist_m += haversine.main(lat1, lon1, lat2, lon2)

            eta_value = total_dist_m * THETA_FACTOR

            trip = Trip(
                id=trip_counter,
                start_point=start_point,
                end_point=end_point,
                start_time=start_time,
                end_time=end_time,
                trip_type="REGULAR"
            )

            trip.start_time_window = (start_time - TIME_WINDOW_SLACK,
                                    start_time + TIME_WINDOW_SLACK)
            trip.trip_length = trip_length
            trip.eta = eta_value

            trips.append(trip)
            trip_counter += 1

    # Direction 0
    build_trips_for_direction(
        go_times,
        go_first_stops_coords,
        go_last_stops_coords,
        avg_go_travel_time,
        go_trip_stop_coords
    )

    # Direction 1
    build_trips_for_direction(
        come_times,
        come_first_stops_coords,
        come_last_stops_coords,
        avg_come_travel_time,
        come_trip_stop_coords
    )

    T_trips = len(trips)

    # ------------------------------------------------------------------
    # 4) Build ProblemInstance and meta info
    # ------------------------------------------------------------------
    grid_size = (60, 60)  # irrelevant for GTFS geometry but required by class

    instance = ProblemInstance(
        grid_size=grid_size,
        trips=trips,
        depots=depots,
        relief_points=relief_points,
        charging_stations=charging_stations
    )

    greek_l = 2.0  # keep as generic waiting/penalty parameter

    instance.meta = {
        "K": K,
        "T": T_trips,
        "F": F_chargers,
        "lambda": greek_l,
        "p_max": PHI_MAX,
        "p_min": PHI_MIN,
        "travel_cost": TRAVEL_COST,
        "charging_rate": CHARGING_RATE_KWH_PER_MINUTE,
        "theta_factor": THETA_FACTOR,
        "route_id": route_id,
        "day": day
    }

    instance.meta.update({
        "buses_per_depot": buses_per_depot,
        "buses_availability_times": buses_availability_times,
        "buses_initial_soc": buses_SoC,
        "number_of_cs_per_depot": number_of_CS_per_depot
    })

    return instance

# ============================================================
# Solver
# ============================================================

def solve_md_vsp_tw_from_instance(instance: ProblemInstance, waiting_cost_lambda: float = 8.0):
    """
    Solves the MD-VSP-TW with an Energy Buffer extension and returns the report.

    IMPORTANT:
    - Trip execution duration is given by trip.trip_length (computed in the loader)
    - Trip start_time and end_time define the earliest and latest SERVICE START time (time windows).
    """
    model = gp.Model("MD-VSP-TW-EB")

    # model.setParam('OutputFlag', 1)

    # Vehicles and depot mapping
    vehicles, vehicle_to_depot = [], {}
    for depot in instance.depots:
        for i in range(1, depot.vehicle_count + 1):
            vehicle_id = f"D{depot.id}_V{i}"
            vehicles.append(vehicle_id)
            vehicle_to_depot[vehicle_id] = depot

    trips_map = {f"T{trip.id}": trip for trip in instance.trips}
    trip_ids = list(trips_map.keys())

    # Give each charging station a unique ID (e.g., 'C1', 'C2', ...)
    charging_station_map = {f"C{i+1}": cs for i, cs in enumerate(instance.charging_stations)}
    charging_station_ids = list(charging_station_map.keys())
    charging_station_set = set(charging_station_ids)
    internal_nodes = trip_ids + charging_station_ids

    # Origin / destination node labels for vehicles
    origin_nodes = {v_id: f"O{vehicle_to_depot[v_id].id}" for v_id in vehicles}
    dest_nodes = {v_id: f"D{vehicle_to_depot[v_id].id}" for v_id in vehicles}

    all_nodes = (
        set(trip_ids)
        | set(charging_station_ids)
        | set(origin_nodes.values())
        | set(dest_nodes.values())
    )

    arc_starting_nodes = (
            set(trip_ids)
            | set(charging_station_ids)
            | set(origin_nodes.values())
        )
    
    target_nodes = (
        set(trip_ids)
        | set(charging_station_ids)
        | set(dest_nodes.values())
    )

    # ============================================================
    # Arcs and costs – using Haversine distances + average speed
    # ============================================================

    arcs, base_costs, elapsed_times, energy_consumption = {}, {}, {}, {}

    for k in vehicles:
        depot = vehicle_to_depot[k]
        o_node = origin_nodes[k]
        d_node = dest_nodes[k]

        # --------------------------------------------------------
        # O_k → Trip_i
        # --------------------------------------------------------
        for trip_id, trip in trips_map.items():
            
            l_i, u_i = trip.start_time_window
            
            dist_m = haversine.main(
                depot.location.y, depot.location.x,
                trip.start_point.y, trip.start_point.x
            )
            
            t_oi = dist_m / AVG_SPEED_M_PER_MIN  # minutes

            # Valid inequality: t_oi <= u_i
            if t_oi <= u_i:
                arc = (k, o_node, trip_id)
                arcs[arc] = 1
                base_costs[arc] = instance.meta["travel_cost"] * t_oi
                elapsed_times[arc] = t_oi
                energy_consumption[arc] = dist_m * THETA_FACTOR

        # --------------------------------------------------------
        # Trip_i → Trip_j
        # --------------------------------------------------------
        for trip1_id, trip1 in trips_map.items():
            l_i, u_i = trip1.start_time_window
            dur_i = trip1.trip_length

            for trip2_id, trip2 in trips_map.items():
                if trip1_id == trip2_id:
                    continue

                l_j, u_j = trip2.start_time_window

                dist_m = haversine.main(
                    trip1.end_point.y, trip1.end_point.x,
                    trip2.start_point.y, trip2.start_point.x
                )

                t_ij = dist_m / AVG_SPEED_M_PER_MIN

                # Valid inequality: l_i + dur_i + t_ij <= u_j
                if l_i + dur_i + t_ij <= u_j:
                    arc = (k, trip1_id, trip2_id)
                    arcs[arc] = 1
                    base_costs[arc] = instance.meta["travel_cost"] * t_ij
                    elapsed_times[arc] = t_ij
                    energy_consumption[arc] = dist_m * THETA_FACTOR

        # --------------------------------------------------------
        # Trip_i → D_k
        # --------------------------------------------------------
        for trip_id, trip in trips_map.items():
            
            dist_m = haversine.main(
                trip.end_point.y, trip.end_point.x,
                depot.location.y, depot.location.x
            )

            t_id = dist_m / AVG_SPEED_M_PER_MIN

            arc = (k, trip_id, d_node)
            arcs[arc] = 1
            base_costs[arc] = instance.meta["travel_cost"] * t_id
            elapsed_times[arc] = t_id
            energy_consumption[arc] = dist_m * THETA_FACTOR

        # --------------------------------------------------------
        # Trip_i → Charging_c
        # --------------------------------------------------------
        for trip_id, trip in trips_map.items():
            l_i, u_i = trip.start_time_window
            dur_i = trip.trip_length

            for cs_id, cs_obj in charging_station_map.items():

                l_c, u_c = getattr(cs_obj, "time_window", (0, float("inf")))

                dist_m = haversine.main(
                    trip.end_point.y, trip.end_point.x,
                    cs_obj.location.y, cs_obj.location.x
                )

                t_ic = dist_m / AVG_SPEED_M_PER_MIN

                # Valid inequality: l_i + dur_i + t_ic <= u_c
                # if l_i + dur_i + t_ic <= u_c:
                arc = (k, trip_id, cs_id)
                arcs[arc] = 1
                base_costs[arc] = instance.meta["travel_cost"] * t_ic
                elapsed_times[arc] = t_ic
                energy_consumption[arc] = dist_m * THETA_FACTOR

        # --------------------------------------------------------
        # Charging_c → Trip_j
        # --------------------------------------------------------
        for cs_id, cs_obj in charging_station_map.items():

            l_c, u_c = getattr(cs_obj, "time_window", (0, float("inf")))
            tau_c = getattr(cs_obj, "min_charge_time", 0)  # minimal charging duration

            for trip_id, trip in trips_map.items():
                
                l_j, u_j = trip.start_time_window

                dist_m = haversine.main(
                    cs_obj.location.y, cs_obj.location.x,
                    trip.start_point.y, trip.start_point.x
                )
                
                t_cj = dist_m / AVG_SPEED_M_PER_MIN

                # Valid inequality: l_c + τ_c + t_cj <= u_j
                # if l_c + tau_c + t_cj <= u_j:
                arc = (k, cs_id, trip_id)
                arcs[arc] = 1
                base_costs[arc] = instance.meta["travel_cost"] * t_cj
                elapsed_times[arc] = t_cj
                energy_consumption[arc] = dist_m * THETA_FACTOR

    # --- Gurobi Variables ---
    x = model.addVars(arcs.keys(), vtype=GRB.BINARY, name="x")

    time_vars_keys = set((k, i) for k, i, j in arcs.keys()) | set((k, j) for k, i, j in arcs.keys())

    T = model.addVars(time_vars_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    w = model.addVars(arcs.keys(), vtype=GRB.CONTINUOUS, lb=0.0, name="w")

    starting_node_keys = set((k, i) for k in vehicles for i in origin_nodes.values() for j in all_nodes if (k, i, j) in arcs.keys())
    ending_node_keys = set((k, j) for k in vehicles for i in all_nodes for j in dest_nodes.values() if (k, i, j) in arcs.keys())

    w_o = model.addVars(starting_node_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="w_o")

    # --- Energy Variables ---
    energy_vars_keys = set((k, i) for k, i in time_vars_keys)
    E_pre = model.addVars(energy_vars_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="e")
    E_bar = model.addVars(energy_vars_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="ebar", ub=PHI_MAX)

    # g_i^k can be NEGATIVE (charging)
    G = model.addVars(energy_vars_keys, vtype=GRB.CONTINUOUS, lb=-PHI_MAX, name="g", ub=PHI_MAX)

    # CT_j^k: Charging completion time for vehicle k at station j
    ct_keys = [(k, j) for k in vehicles for j in charging_station_ids]
    CT = model.addVars(ct_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="CT")

    # y_j^{k1, k2}: Binary to order vehicles at charging station j
    y_keys = [(j, k1, k2) for j in charging_station_ids for k1 in vehicles for k2 in vehicles if k1 != k2]
    Y = model.addVars(y_keys, vtype=GRB.BINARY, name="Y")

    tau = model.addVars(ct_keys, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1440, name='tau') #required time period to recharge vehicle k at charging event i

    buses_state = instance.meta.get("buses_state")

    if buses_state is None:
        raise ValueError("buses_state missing from instance.meta")

    # vehicles is already an ordered list like ["D1_V1", "D1_V2", ...]
    b_s = {k: buses_state[i] for i, k in enumerate(vehicles)}

    # Objective function (32)
    waiting_cost_lambda = instance.meta["lambda"]
    initial_soc = instance.meta["buses_initial_soc"]

    lambda_1 = waiting_cost_lambda
    lambda_2 = 150
    lambda_3 = 0.5

    # Objective function (32)
    objective = gp.quicksum(base_costs[k, i, j] * x[k, i, j] for k, i, j in arcs.keys()) + \
                gp.quicksum(lambda_1 * w[k, i, j] for k, i, j in arcs.keys() if (j in trip_ids) or (j in charging_station_ids)) + \
                gp.quicksum(lambda_2 * b_s[k] * x[k, i, j] for k, i, j in arcs.keys() if i in origin_nodes.values()) + \
                gp.quicksum(lambda_3 * w_o[k, i] for k, i in starting_node_keys)
    model.setObjective(objective, GRB.MINIMIZE)

    print("\r")
    print("Vehicle fleet available for this problem instance: ")
    for row in vehicles:
        print(row)

    # Constraint (#33)
    # model.addConstrs((x.sum('*', '*', j) == 1 for j in trip_ids), name="CoverTrip")
    for j in trip_ids:
        model.addConstr(gp.quicksum(x[k, i, j] for k in vehicles for i in all_nodes if (k, i, j) in arcs.keys()) == 1, name="CoverTrip")
    
    for k in vehicles:
        for j in charging_station_ids:
            model.addConstr(gp.quicksum(x[k, i, j] for i in trip_ids if (k, i, j) in arcs.keys()) <= 1, name=f"SingleUseCharger_{j}")

    # Constraint (#34)
    model.addConstrs((x.sum(k, '*', v) - x.sum(k, v, '*') == 0 for k in vehicles for v in internal_nodes), name="FlowConservation")

    # Constraint (#35)
    model.addConstrs((x.sum(k, origin_nodes[k], '*') == x.sum(k, '*', dest_nodes[k]) for k in vehicles), name="ReturnToDepot")

    # Constraint (#36)
    # model.addConstrs((x.sum(k, origin_nodes[k], '*') <= 1 for k in vehicles), name="StartOnce")
    for k in vehicles:
        model.addConstr(gp.quicksum(x[k, i, j] for i in origin_nodes for j in trip_ids if (k, i, j) in arcs.keys()) <= 1, name="StartOnce")

    depot_to_vehicles = {}
    for depot in instance.depots:
        depot_id_str = f"O{depot.id}"
        depot_to_vehicles[depot_id_str] = [
            v_id for v_id in vehicles if origin_nodes[v_id] == depot_id_str
        ]

    depots_map = {depot.id: depot for depot in instance.depots}
    for origin_node_id, vehicles_in_depot in depot_to_vehicles.items():
        depot_id_num = int(origin_node_id[1:])
        depot_obj = depots_map[depot_id_num]

        # Constraint (#37)
        model.addConstr(
            gp.quicksum(x[k, origin_node_id, j] for k in vehicles_in_depot for j in trip_ids if (k, origin_node_id, j) in x) <= depot_obj.vehicle_count,
            name=f"DepotCapacity_{origin_node_id}"
        )

    # ==================================================
    # Time-window constraints
    # ==================================================
    availability = instance.meta["buses_availability_times"]
    for k, i in T.keys():

        # Trip nodes
        if i in trips_map:
            trip = trips_map[i]
            T[k, i].lb, T[k, i].ub = trip.start_time_window[0], trip.start_time_window[1]

        elif i in charging_station_set:
            cs_obj = charging_station_map[i]
            # T[k, i].lb, T[k, i].ub = cs_obj.time_window[0], cs_obj.time_window[1]
            T[k, i].lb, T[k, i].ub = 0, BIG_M

        # Depot nodes (O#, D#)
        elif i.startswith("O") or i.startswith("D"):
            idx = vehicles.index(k)
            T[k, i].lb = availability[idx]
            T[k, i].ub = BIG_M

    # ==================================================
    # Time continuity
    # ==================================================
    for k, i, j in arcs.keys():
        
        t_ij = elapsed_times[k, i, j]

        if i in trip_ids:
            trip = trips_map[i]
            trip_service = trip.trip_length
        elif i in charging_station_ids:
            trip_service = tau[k, i]
        else:
            trip_service = 0

        # Time propagation — unchanged
        model.addConstr(T[k, j] >= T[k, i] + trip_service + t_ij - BIG_M*(1 - x[k, i, j]), name=f"TimeProp_{k}_{i}_{j}")

        if j.startswith("D"):
            model.addConstr(T[k, j] <= T[k, i] + trip_service + t_ij + BIG_M*(1 - x[k, i, j]), name=f"TimeProp_{k}_{i}_{j}_D")

        # Waiting constraints for O, V, F - edo orizetai to waiting time
        if (j in trip_ids) or (j in charging_station_ids) or (j in origin_nodes.values()):
            
            model.addConstr(w[k, i, j] >= T[k, j] - (T[k, i] + trip_service + t_ij) - BIG_M*(1 - x[k, i, j]))

            # model.addConstr(w[k,i,j] <= T[k,j] - (T[k, i] + trip_service + t_ij) + BIG_M*(1 - x[k, i, j]))

            # model.addConstr(w[k,i,j] <= BIG_M * x[k, i, j])

        if i in origin_nodes.values():
            idx = vehicles.index(k)
            model.addConstr(w_o[k, i] >= (1-buses_state[idx])*(T[k, i] - availability[idx]) - BIG_M*(1 - x[k, i, j]))    

    # ==================================================
    # Energy Consumption constraints
    # ==================================================
    for k in vehicles:
        o_node = origin_nodes[k]
        d_node = dest_nodes[k]

        # Constraint (#44)
        vehicle_index = vehicles.index(k)

        model.addConstr(E_bar[k, o_node] == initial_soc[vehicle_index], name=f"EB_StartSOC_{k}")

        # Nodes for this vehicle
        nodes_for_k = [i for k_i, i in E_pre.keys() if k_i == k]

        for i in internal_nodes:
            # Constraint (#45)
            model.addConstr(E_bar[k, i] == E_pre[k, i] - G[k, i], name=f"EB_BufferUpdate_{k}_{i}")

        for i in arc_starting_nodes: 
            for j in target_nodes: 
                
                if (k, i, j) in arcs.keys(): 
            
                    theta_ij = energy_consumption[(k, i, j)]
                
                    # Constraint (#46) 
                    model.addConstr(E_pre[k, j] >= E_bar[k, i] - theta_ij - BIG_M * (1 - x[k, i, j]), name=f"EB_PropLB_{k}_{i}_{j}") 
                
                    # Constraint (#47) 
                    model.addConstr(E_pre[k, j] <= E_bar[k, i] - theta_ij + BIG_M * (1 - x[k, i, j]), name=f"EB_PropUB_{k}_{i}_{j}")

    for k in vehicles:
        o_node = origin_nodes[k]
        d_node = dest_nodes[k]

        nodes_for_k = [i for k_i, i in E_pre.keys() if k_i == k]

        for i in nodes_for_k:
            if i.startswith('O'):
                continue

            # Trip nodes: charging/consumption based on ETA
            if i in trip_ids:
                trip_eta = trips_map[i].eta
                
                # Constraint (#48)
                model.addConstr(G[k, i] == trip_eta, name=f"EB_TripEta_{k}_{i}")

            # Charging stations: refill to PHI_MAX
            if i in charging_station_set:
                # Constraint (#49)
                model.addConstr(G[k, i] == E_pre[k, i] - PHI_MAX, name=f"EB_DepotRefillLogic_{k}_{i}")

            # Constraint (#50) - Min energy level before refill
            model.addConstr(E_pre[k, i] >= PHI_MIN, name=f"EB_MinLevelPre_{k}_{i}")

    # ==================================================
    # Continuous time constraints for charging stations
    # ==================================================

    # τ[k,j] = (E_bar[k,j] - E_pre[k,j]) / CHARGING_RATE_KWH_PER_MINUTE
    for k in vehicles:
        for j in charging_station_set:
            for i in trip_ids:
                if (k, i, j) in arcs.keys():
                    model.addConstr(tau[k, j] >= ((E_bar[k, j] - E_pre[k, j]) / CHARGING_RATE_KWH_PER_MINUTE) - BIG_M * (1 - x[k, i, j]), name=f"TauDef_{k}_{j}")
                    model.addConstr(tau[k, j] <= ((E_bar[k, j] - E_pre[k, j]) / CHARGING_RATE_KWH_PER_MINUTE) + BIG_M * (1 - x[k, i, j]), name=f"TauDef_{k}_{j}")

    for k in vehicles:
        for j in charging_station_set:
            for i in trip_ids:
                if (k, i, j) in arcs.keys():
                    
                    # Constraint (#51)
                    # model.addConstr(CT[k, j] <= T[k, j] + tau[k, j] + BIG_M * (1 - x[k, i, j]), name=f"ChargeCompTime_UB1_{k}_{i}_{j}")
                    
                    # Constraint (#52)
                    model.addConstr(CT[k, j] >= T[k, j] + tau[k, j] - BIG_M * (1 - x[k, i, j]), name=f"ChargeCompTime_LB1_{k}_{i}_{j}")

    # for k in vehicles:
    #     for j in charging_station_set:
    #         model.addConstr(CT[k, j] <= BIG_M * gp.quicksum(x[k, i, j] for i in trip_ids if (k, i, j) in arcs.keys()), name=f"ChargeCompTime_zero_{k}_{j}") # Constraint (53)

    # Charging order constraints
    for j in charging_station_ids:
        for k1 in vehicles:
            for k2 in vehicles:
                if k1 != k2:
                    # Constraint (#53)
                    model.addConstr(
                        T[k1, j] <= T[k2, j] + BIG_M * Y[j, k1, k2], name=f"ChargeOrder_Arr_Time_1_{j}_{k1}_{k2}")
                    # Constraint (#54)
                    model.addConstr(
                        T[k1, j] >= CT[k2, j] + SMALL_M - BIG_M * (1 - Y[j, k1, k2]), name=f"ChargeOrder_Comp_Time_2_{j}_{k1}_{k2}")

    # Constraint for zero-ing Y - turns out that it is optional                
    # for j in charging_station_ids:
    #     for k1 in vehicles:
    #         for k2 in vehicles:
    #             if k1 != k2:
    #                 model.addConstr(Y[j, k1, k2] <= BIG_M * gp.quicksum(x[k1, i, j] for i in trip_ids if (k, i, j) in arcs.keys()) , name=f"zero_Y_1{k}_{j}") # Constraint (53)
    #                 model.addConstr(Y[j, k1, k2] <=  BIG_M * gp.quicksum(x[k2, i, j] for i in trip_ids if (k, i, j) in arcs.keys()), name=f"zero_Y_2{k}_{j}") # Constraint (53)

    # ============================================================
    # Valid Inequalities
    # ============================================================

    # Set 1 is included in the arc generation process above.

    # ============================================================
    # Valid Inequality Set 2: SOC reachability from trip j
    # ============================================================

    # Precompute min energy needed from each trip to nearest charger
    min_energy_to_cs = {}

    for trip_id, trip in trips_map.items():
        best = float("inf")
        for cs_id, cs in charging_station_map.items():
            dist_m = haversine.main(
                trip.end_point.y, trip.end_point.x,
                cs.location.y, cs.location.x
            )
            energy = dist_m * THETA_FACTOR
            if energy < best:
                best = energy
        min_energy_to_cs[trip_id] = best

    for k in vehicles:
        for trip_id in trip_ids:
            
            theta_min = min_energy_to_cs[trip_id]

            # Set 2 inequality:
            # E_bar[k, trip] >= PHI_MIN + (energy required to reach nearest charger)
            model.addConstr(E_bar[k, trip_id] >= PHI_MIN + theta_min, name=f"VI2_SoCReach_{k}_{trip_id}")
    
    # ============================================================
    # Valid Inequality Set 4: Time tightening constraints
    # ============================================================

    for (k, i, j) in arcs.keys():

        # elapsed travel time
        t_ij = elapsed_times[(k, i, j)]

        if i in trip_ids:
            trip = trips_map[i]
            trip_service = trip.trip_length
        elif i in charging_station_ids:
            trip_service = tau[k, i]
        else:
            trip_service = 0

        # Only tighten when j is a trip or a charging station with a known time window
        if j in trip_ids:
            # Latest allowable start time at trip j
            u_j = trips_map[j].start_time_window[1]

            # Set 4 inequality:
            # T[k,i] + t_tilde + t_ij <= u_j + BIG_M*(1 - x[k,i,j])
            model.addConstr(T[k, i] + trip_service + t_ij <= u_j + BIG_M*(1 - x[k, i, j]), name=f"VI4_timeTrip_{k}_{i}_{j}")

        # elif j in charging_station_set:
        #     # charging stations also have a time window (earliest, latest)
        #     u_j = charging_station_map[j].time_window[1]

        #     # analogous Set 4 inequality for chargers:
        #     model.addConstr(T[k, i] + t_tilde + t_ij <= u_j + BIG_M*(1 - x[k, i, j]), name=f"VI4_timeCS_{k}_{i}_{j}")

    # ============================================================
    # Valid Inequality Set 5: Conflicting-arc inequalities
    # ============================================================

    # Outgoing-arc conflicts: sum_j x[k, v, j] ≤ 1
    for k in vehicles:
        for v in trip_ids:
            outgoing_arcs = [(k, v, j) for (_k, i, j) in arcs.keys() if _k == k and i == v]
            if outgoing_arcs:
                model.addConstr(
                    gp.quicksum(x[a] for a in outgoing_arcs) <= 1, name=f"VI5_out_{k}_{v}")

    # Incoming-arc conflicts: sum_i x[k, i, v] ≤ 1
    for k in vehicles:
        for v in trip_ids:
            incoming_arcs = [(k, i, v) for (_k, i, j) in arcs.keys() if _k == k and j == v]
            if incoming_arcs:
                model.addConstr(
                    gp.quicksum(x[a] for a in incoming_arcs) <= 1, name=f"VI5_in_{k}_{v}")

    # model.setParam('MIPGap', 0.01)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:

        print("\n===== Gurobi Performance Summary =====")

        print(f"Constraints (CNS):         {model.NumConstrs:,}")
        print(f"Branch & Bound Nodes (NE): {model.NodeCount:,}")
        print(f"Simplex Iterations (SI):   {model.IterCount:,}")

        print(f"Solve Time (CT):           {model.Runtime/60:.2f} minutes")
        print(f"Optimality Gap (OG):       {model.MIPGap*100:.2f}%")

        print(f"Solution Performance (SP): {model.ObjVal:.2f}")

    # --- Reporting Logic (Updated to include Energy Variables) ---
    report_lines, schedules = [], {}
    variable_report_str = ""
    solution_data = {}
    default_time_var = type('obj', (object,), {'X': 0.0})  # Helper for safe .X access

    if model.status == GRB.OPTIMAL:

        bus_end_states = {}

        used_vehicles = {
            k for (k, i, j) in x.keys()
            if x[k, i, j].X > 0.5
        }

        for k in used_vehicles:
            final_node = dest_nodes[k]  # e.g. "D1"

            arrival_time = T.get((k, final_node), default_time_var).X
            soc = E_pre.get((k, final_node), default_time_var).X

            depot_id = int(final_node[1:])  # strip "D"

            bus_end_states[k] = {
                "depot_id": depot_id,
                "arrival_time": round(arrival_time, 4),
                "soc": round(soc, 4)
            }

        bus_end_states_dir = os.path.join(bus_end_states_output_dir)
        os.makedirs(bus_end_states_dir, exist_ok=True)

        bus_end_states_path = os.path.join(bus_end_states_dir, f"bus_end_states_{timestamp}.json")

        with open(bus_end_states_path, "w") as f:
            json.dump(bus_end_states, f, indent=4)

        # New dictionary to collect variable results for JSON export
        solution_vars_json = {}
        TOL = 1e-4  # Tolerance for non-negative values

        # --- Helper function to extract and save variable values ---
        def extract_vars(gurobi_vars, var_name):
            data = {}
            for key, var in gurobi_vars.items():
                if var.X > TOL:
                    str_key = str(key).replace("'", "").replace(" ", "")
                    data[str_key] = round(var.X, 4)
            solution_vars_json[var_name] = data

        # Extract main variables
        extract_vars(x, "x")
        extract_vars(T, "T")
        extract_vars(w, "w")
        extract_vars(E_pre, "E_pre")
        extract_vars(E_bar, "E_bar")
        extract_vars(G, "G")

        # CT and Y
        try:
            extract_vars(CT, "CT")
            extract_vars(Y, "Y")
        except NameError:
            pass

        solution_data["solution_vars"] = solution_vars_json

        report_lines.append(f"Optimal solution found with total cost: {model.ObjVal:.2f}")
        used_vehicles = {k for k, i, j in x.keys() if x[k, i, j].X > 0.5}
        report_lines.append(f"Total vehicles used: {len(used_vehicles)} out of {len(vehicles)}")

        # Schedule reconstruction
        for k in sorted(list(used_vehicles)):
            current_node = origin_nodes[k]
            route = [origin_nodes[k]]

            start_time = T.get((k, origin_nodes[k]), default_time_var).X
            start_ebar = E_bar.get((k, origin_nodes[k]), default_time_var).X
            route_str = [f"{origin_nodes[k]} (Time: {start_time:.2f}, E_bar: {start_ebar:.1f})"]

            while current_node not in dest_nodes.values():
                found_next = False
                vehicle_arcs = [(i, j) for _k, i, j in arcs.keys() if _k == k]
                for i_node, j_node in vehicle_arcs:
                    if i_node == current_node and x.get((k, i_node, j_node)) and x[k, i_node, j_node].X > 0.5:
                        route.append(j_node)
                        time_val = T.get((k, j_node), default_time_var).X

                        epre_val = E_pre.get((k, j_node), default_time_var).X
                        gbar_val = E_bar.get((k, j_node), default_time_var).X
                        g_val = G.get((k, j_node), default_time_var).X

                        energy_info = f" (E_pre: {epre_val:.1f}, G: {g_val:.1f}, E_bar: {gbar_val:.1f})"

                        route_str.append(f"{j_node} (Time: {time_val:.2f}{energy_info})")
                        current_node = j_node
                        found_next = True
                        break
                if not found_next:
                    break
            schedules[k] = route
            report_lines.append(f" - Vehicle {k} schedule: {' -> '.join(route_str)}")

        var_report_lines = []

        # 1. Active arcs
        var_report_lines.append("\n--- Active Arc Variables (x[k,i,j] = 1) ---")
        sorted_x_keys = sorted([key for key in x.keys() if x[key].X > 0.5])
        for (k, i, j) in sorted_x_keys:
            var_report_lines.append(
                f"  x[{k}, {i}, {j}] = {x[k, i, j].X:.0f} "
                f"(Cost: {base_costs[k, i, j]}, Cons: {energy_consumption.get((k, i, j), 0):.1f})"
            )

        # 2. Node start times
        var_report_lines.append("\n--- Node Start Times (T[k,i] > 0) ---")
        solution_nodes = set((k, i) for (k, i, j) in sorted_x_keys) | set((k, j) for (k, i, j) in sorted_x_keys)
        sorted_T_keys = sorted(
            [key for key in T.keys() if key in solution_nodes and T.get(key) is not None and T[key].X > 0.0001]
        )
        for (k, i) in sorted_T_keys:
            var_report_lines.append(f"  T[{k}, {i}] = {T[k, i].X:.2f}")

        # 3. Waiting costs
        var_report_lines.append("\n--- Incurred Waiting Costs (w[k,i,j] > 0) ---")
        sorted_w_keys = sorted([key for key in w.keys() if w[key].X > 0.0001])
        for (k, i, j) in sorted_w_keys:
            var_report_lines.append(f"  w[{k}, {i}, {j}] = {w[k, i, j].X:.2f}")

        # 4. Energy variables
        var_report_lines.append("\n--- Energy Levels (E_pre, G, E_bar) ---")
        sorted_E_keys = sorted([key for key in E_pre.keys() if key in solution_nodes])
        for (k, i) in sorted_E_keys:
            e_pre_val = E_pre.get((k, i), default_time_var).X
            g_val = G.get((k, i), default_time_var).X
            e_bar_val = E_bar.get((k, i), default_time_var).X
            if e_bar_val > 0.0001:
                var_report_lines.append(
                    f"  {k}, {i}: E_pre={e_pre_val:.1f} | G={g_val:.1f} | "
                    f"E_bar={e_bar_val:.1f} (Min/Max: {PHI_MIN}/{PHI_MAX})"
                )

        # 5. Charging completion times
        var_report_lines.append("\n--- Charging Completion Times (CT[k, j] > 0) ---")
        solution_cs_nodes = set((k, i) for (k, i) in sorted_E_keys if i in charging_station_set)
        sorted_CT_keys = sorted(
            [key for key in CT.keys() if key in solution_cs_nodes and CT.get(key) is not None and CT[key].X > 0.0001]
        )
        for (k, j) in sorted_CT_keys:
            var_report_lines.append(f"  CT[{k}, {j}] = {CT[k, j].X:.2f}")

        # 6. Charging order variables
        var_report_lines.append("\n--- Charging Order Variables (Y[j, k1, k2] = 1) ---")
        sorted_Y_keys = sorted([key for key in Y.keys() if Y[key].X > 0.5])
        for (j, k1, k2) in sorted_Y_keys:
            var_report_lines.append(f"  Y[{j}, {k1}, {k2}] = {Y[j, k1, k2].X:.0f}")

        variable_report_str = "\n".join(var_report_lines)

        if solution_data:
            json_output_path = os.path.join(output_dir, "solution_variables.json")
            with open(json_output_path, "w") as f:
                json.dump(solution_data["solution_vars"], f, indent=4)
            print(f"Solution variables saved to '{json_output_path}'")
    else:
        report_lines.append("No optimal solution was found or the model was infeasible.")

    return "\n".join(report_lines), schedules, variable_report_str, bus_end_states, model.status, model.ObjVal

# ============================================================
# Reporting Function (Non-Plotting)
# ============================================================

def get_instance_report(instance: ProblemInstance) -> str:
    """
    Generates a formatted string report for the instance,
    including t_ij values between all relevant nodes.
    """
    report_lines = []
    report_lines.append("--- Instance Generation Summary ---")
    total_vehicles = sum(depot.vehicle_count for depot in instance.depots)
    report_lines.append(f"  - Trips to cover: {len(instance.trips)}")
    report_lines.append(f"  - Depots: {len(instance.depots)}")
    report_lines.append(f"  - Total vehicles available: {total_vehicles}")

    report_lines.append("\n--- Depot and Vehicle Information ---")
    for depot in instance.depots:
        report_lines.append(
            f"Depot ID: {depot.id} | "
            f"Location: ({depot.location.x:5.6f}, {depot.location.y:5.6f}) | "
            f"Vehicles: {depot.vehicle_count}"
        )

    report_lines.append("\n--- Detailed Trip Information ---")
    point_to_id_map = {point: i + 1 for i, point in enumerate(instance.relief_points)}

    for trip in instance.trips:
        start_rp_id = point_to_id_map.get(trip.start_point, "N/A")
        end_rp_id = point_to_id_map.get(trip.end_point, "N/A")

        trip_duration = getattr(trip, "trip_length", trip.end_time - trip.start_time)
        eta_info = f" | ETA: {trip.eta:.4f}" if hasattr(trip, "eta") else ""

        report_lines.append(
            f"Trip ID: {trip.id: <3} | "
            f"Type: {trip.trip_type: <7} | "
            f"Starts at RP #{start_rp_id:<3} -> Ends at RP #{end_rp_id:<3} | "
            f"Start/End Time: {trip.start_time:7.2f} to {trip.end_time:7.2f} | "
            f"Duration: {trip_duration:.2f}{eta_info}"
        )

    # -----------------------------------------------------------
    # t_ij reporting
    # -----------------------------------------------------------
    report_lines.append("\n--- Deadhead Travel Times (t_ij, minutes) ---")

    trips = instance.trips
    depots = instance.depots

    # Helper for deadhead time using haversine + speed
    def dist(p1, p2):
        dist_m = haversine.main(p1.y, p1.x, p2.y, p2.x)
        return dist_m / AVG_SPEED_M_PER_MIN

    # Depot -> Trip
    report_lines.append("\nDepot → Trip:")
    for depot in depots:
        for trip in trips:
            tij = dist(depot.location, trip.start_point)
            report_lines.append(
                f"  Depot {depot.id} -> Trip {trip.id}: t_ij = {tij:.2f}"
            )

    # Trip -> Trip
    report_lines.append("\nTrip → Trip:")
    for t1 in trips:
        for t2 in trips:
            if t1.id == t2.id:
                continue
            tij = dist(t1.end_point, t2.start_point)
            report_lines.append(
                f"  Trip {t1.id} -> Trip {t2.id}: t_ij = {tij:.2f}"
            )

    # Trip -> Depot
    report_lines.append("\nTrip → Depot:")
    for trip in trips:
        for depot in depots:
            tij = dist(trip.end_point, depot.location)
            report_lines.append(
                f"  Trip {trip.id} -> Depot {depot.id}: t_ij = {tij:.2f}"
            )

    return "\n".join(report_lines)


# ============================================================
# Main Execution Block
# ============================================================

if __name__ == "__main__":
    """
    This block runs only when the script is executed directly.
    """

    random.seed(43)

    # 1. Load instance from GTFS
    print("--- Part 1: Loading GTFS-based instance ---")

    # GTFS settings
    route_id = 1033 # 550 -> 959, 831 -> 874, 217 -> 1033, 229 -> 1034, Β1 -> 871, Χ14 -> 993, 451 -> 1060. 
    day = "monday"
    gtfs_folder_path = os.path.join(project_root, "..", "input", "gtfs", "oasa_third_results_section")
    depot_filepath = os.path.join(project_root, "..", "input", "depots.txt")

    cluster_id = 1

    cluster_trips_txt = os.path.join(
        project_root,
        "..",
        "output",
        "clusters",
        f"cluster_{cluster_id}",
        "trips.txt"
    )

    instance = load_instance_from_gtfs_cluster(
        cluster_trips_txt_path=cluster_trips_txt,
        day=day,
        gtfs_folder_path=gtfs_folder_path,
        depot_filepath=depot_filepath,
        buses_per_depot=[4, 4],
        buses_availability_times=[0.0] * 8,
        buses_SoC=[350.0] * 8,
        number_of_CS_per_depot=1,
        buses_state = [True] * 8
    )

    print(f"Instance successfully loaded for cluster {cluster_id} on {day}.")

    # 2. Set up output directory
    print("--- Part 2: Preparing Output Directory ---")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, "..", "output", f"EB_TW_run_{timestamp}")
    bus_end_states_output_dir = os.path.join(project_root, "..", "output", "bus_end_states")
    os.makedirs(output_dir, exist_ok=True)
    bus_blocks_dir = os.path.join(output_dir, "bus_blocks")
    os.makedirs(bus_blocks_dir, exist_ok=True)

    # 3. Save instance report and (temporarily disable) plots
    print(f"--- Part 3: Saving instance details to '{output_dir}' ---")
    instance_report = get_instance_report(instance)

    # Commented out plotting as requested
    plotting_utils_eb.save_instance_plot(instance, os.path.join(output_dir, "instance_plot.png"))
    trip_y = plotting_utils_eb.save_dag_plot(instance, os.path.join(output_dir, "instance_dag.png"))

    # 4. Solve instance
    print("--- Part 4: Solving the Instance ---")
    solution_report, schedules, variable_report_str, bus_end_states, _, _ = solve_md_vsp_tw_from_instance(instance)

    print("\r")

    print(schedules)

    print("\r")

    # Commented out solution plots as requested
    if schedules:
        plotting_utils_eb.save_solution_dag_plot(
            instance,
            schedules,
            os.path.join(output_dir, "solved_instance_dag.png"),
            trip_y=trip_y
        )
        plotting_utils_eb.save_solution_plot(instance, schedules, bus_blocks_dir)

    # 5. Write combined report to file
    print("--- Part 5: Writing Full Report to File ---")
    full_report_path = os.path.join(output_dir, "instance_and_solution_report.txt")
    with open(full_report_path, "w", encoding='utf-8') as f:
        f.write("--- INSTANCE INFORMATION ---\n")
        f.write(instance_report)
        f.write("\n\n--- SOLUTION ---\n")
        f.write(solution_report)

        if variable_report_str:
            f.write("\n\n--- SOLUTION VARIABLES ---")
            f.write(variable_report_str)

    print(f"\nProcess complete. All results saved in '{output_dir}'")
