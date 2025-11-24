import os
import sys
import datetime
import math
import random
import json

import gurobipy as gp
from gurobipy import GRB

# --- Add parent directory to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities.instance_generator import Point, Depot, Trip, ChargingStation, ProblemInstance
import plotting.plotting_utils_eb as plotting_utils_eb
from shapely.geometry import Point

import csv

# --- Constants for the Energy Buffer (EB) Model Extension ---
SMALL_M = 0.0001
BIG_M = 10000

# ============================================================
# Euclidean distance helper
# ============================================================

def euclidean_distance(lat1, lon1, lat2, lon2):
    """
    Compute Euclidean distance between two coordinates (lat, lon).
    Here coordinates are on a grid, so plain Euclidean is fine.
    """
    dx = lon2 - lon1
    dy = lat2 - lat1
    return math.sqrt(dx * dx + dy * dy)

# ============================================================
# Instance Loader
# ============================================================

def load_instance_from_txt(path: str) -> ProblemInstance:
    """
    Loads a ProblemInstance from the custom text file format:
    HEADER (1 row) + BODY (node rows as matrix):
      - Row 1        : Header (global parameters)
      - Rows 2–5     : Depots (2 origin depots, 2 destination depots)
      - Next T rows  : Trip nodes
      - Next F rows  : Charging station nodes

    Also computes for each Trip:
      - ETA       = Euclidean distance * theta_factor
      - trip_length = Euclidean distance * travel_cost  (used as execution time)
    """

    # ============================================================
    # 1) READ FILE
    # ============================================================

    with open(path, "r") as f:
        reader = list(csv.reader(f, delimiter="\t"))

    if len(reader) < 5:
        raise ValueError("Instance file must have at least 5 rows (header + 4 depot rows).")

    # ---------------------------
    # HEADER
    # ---------------------------

    header = list(map(float, reader[0]))

    if len(header) < 9:
        raise ValueError("Header row must contain at least 9 values.")

    # Index mapping from problem description
    K = int(header[0])          # Number of Vehicles
    T_trips = int(header[1])    # Number of Trips
    F_chargers = int(header[2]) # Number of Chargers
    greek_l = header[3]         # Lambda (waiting/penalty parameter)
    p_max = header[4]           # Max SOC / Energy level
    p_min = header[5]           # Min SOC / Energy level
    travel_cost = header[6]     # Travel cost coefficient (used for trip_length)
    charging_rate = header[7]   # CHARGING_RATE_KWH_PER_MINUTE
    theta_factor = header[8]    # Energy consumption factor (THETA_FACTOR)

    # Map header values into the global EB constants used by the solver
    global PHI_MAX, PHI_MIN, CHARGING_RATE_KWH_PER_MINUTE, THETA_FACTOR
    PHI_MAX = p_max
    PHI_MIN = p_min
    CHARGING_RATE_KWH_PER_MINUTE = charging_rate
    THETA_FACTOR = theta_factor

    # ---------------------------
    # BODY
    # ---------------------------
    body = reader[1:]  # everything after header

    expected_min_rows = 4 + T_trips + F_chargers
    if len(body) < expected_min_rows:
        raise ValueError(
            f"Body has {len(body)} rows but at least {expected_min_rows} "
            f"are required (4 depots + {T_trips} trips + {F_chargers} chargers)."
        )

    # Slice body according to the spec
    depot_rows = body[0:4]
    trip_rows = body[4:4 + T_trips]
    charger_rows = body[4 + T_trips:4 + T_trips + F_chargers]

    # ============================================================
    # 2) PARSE DEPOTS
    # ============================================================
    depots = []
    trips = []
    charging_stations = []
    relief_points = []

    # Split K vehicles between first and second origin depots
    # (if K is odd, first origin gets floor(K/2), second gets the rest)
    vehicles_depot_1 = K // 2
    vehicles_depot_2 = K - vehicles_depot_1

    for idx, row in enumerate(depot_rows):
        # Row structure (body): [Node ID, lat_o, lon_o, lat_d, lon_d, t_earliest, t_latest]
        node_id = int(row[0])
        lat_o, lon_o = float(row[1]), float(row[2])
        lat_d, lon_d = float(row[3]), float(row[4])
        t_earliest = float(row[5])
        t_latest = float(row[6])

        # Use origin coordinates for the depot location
        p_o = Point(lon_o, lat_o)
        p_d = Point(lon_d, lat_d)

        # Store both origin and destination points as relief points (for reporting/plotting)
        relief_points.append(p_o)
        relief_points.append(p_d)

        # Determine how many vehicles this depot has
        if idx == 0:
            vehicle_count = vehicles_depot_1   # First origin depot
        elif idx == 1:
            vehicle_count = vehicles_depot_2   # Second origin depot
        else:
            # Destination depots: no vehicles originate here
            vehicle_count = 0

        depot = Depot(
            id=node_id,
            location=p_o,
            vehicle_count=vehicle_count
        )
        depots.append(depot)

    # ============================================================
    # 3) PARSE TRIPS (NEXT T ROWS)
    # Assign:
    #   trip.eta        = Euclidean distance * theta_factor
    #   trip.trip_length = Euclidean distance * travel_cost (used as execution time)
    # ============================================================

    for row in trip_rows:
        node_id = int(row[0])

        lat_o, lon_o = float(row[1]), float(row[2])
        lat_d, lon_d = float(row[3]), float(row[4])
        t_earliest = float(row[5])
        t_latest = float(row[6])

        p_o = Point(lon_o, lat_o)
        p_d = Point(lon_d, lat_d)

        relief_points.append(p_o)
        relief_points.append(p_d)

        # Compute Euclidean distance between task origin and destination
        distance = euclidean_distance(lat_o, lon_o, lat_d, lon_d)

        # Compute ETA and trip_length
        eta_value = distance * theta_factor

        # Build Trip object
        trip = Trip(
            id=node_id,
            start_point=p_o,
            end_point=p_d,
            start_time=t_earliest,
            end_time=t_latest,
            trip_type="REGULAR"
        )

        # Assign time window and energy-related attributes
        trip.start_time_window = (t_earliest, t_latest)
        trip.eta = eta_value
        trip.trip_length = distance

        trips.append(trip)

    # ============================================================
    # 4) PARSE CHARGING STATIONS 
    # ============================================================
    for row in charger_rows:
        node_id = int(row[0])

        lat_o, lon_o = float(row[1]), float(row[2])
        lat_d, lon_d = float(row[3]), float(row[4])  # usually same as origin for chargers
        t_earliest = float(row[5])
        t_latest = float(row[6])

        p_o = Point(lon_o, lat_o)
        p_d = Point(lon_d, lat_d)

        relief_points.append(p_o)
        relief_points.append(p_d)

        cs = ChargingStation(
            id=node_id,
            location=p_o,
            time_window=(t_earliest, t_latest)
        )
        charging_stations.append(cs)

    # ============================================================
    # 5) BUILD ProblemInstance
    # ============================================================
    grid_size = (60, 60)  # keep consistent with generator/grid used elsewhere

    instance = ProblemInstance(
        grid_size=grid_size,
        trips=trips,
        depots=depots,
        relief_points=relief_points,
        charging_stations=charging_stations
    )

    # Attach global parameters for solver / reporting
    instance.meta = {
        "K": K,
        "T": T_trips,
        "F": F_chargers,
        "lambda": greek_l,
        "p_max": p_max,
        "p_min": p_min,
        "travel_cost": travel_cost,
        "charging_rate": charging_rate,
        "theta_factor": theta_factor
    }

    return instance

# ============================================================
# Solver
# ============================================================

def solve_md_vsp_tw_from_instance(instance: ProblemInstance, waiting_cost_lambda: float = 2.0):
    """
    Solves the MD-VSP-TW with an Energy Buffer extension and returns the report.

    IMPORTANT:
    - Trip execution duration is given by trip.trip_length (computed in the loader)
    - Trip start_time and end_time define the earliest and latest SERVICE START time (time windows).
    """
    model = gp.Model("MD-VSP-TW-EB")

    model.setParam('OutputFlag', 1)

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

    # ============================================================
    # Arcs and costs – using pure Euclidean distances
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
            t_oi = euclidean_distance(
                depot.location.y, depot.location.x,
                trip.start_point.y, trip.start_point.x
            )

            # feasible if arrival before trip latest start
            if t_oi <= trip.start_time_window[1]:
                arc = (k, o_node, trip_id)
                arcs[arc] = 1
                base_costs[arc] = instance.meta["travel_cost"] * t_oi
                elapsed_times[arc] = t_oi
                energy_consumption[arc] = t_oi * THETA_FACTOR

        # --------------------------------------------------------
        # Trip_i → Trip_j
        # --------------------------------------------------------
        for trip1_id, trip1 in trips_map.items():
            for trip2_id, trip2 in trips_map.items():
                if trip1_id == trip2_id:
                    continue

                # pure deadhead Euclidean distance
                t_ij = euclidean_distance(
                    trip1.end_point.y, trip1.end_point.x,
                    trip2.start_point.y, trip2.start_point.x
                )

                # feasibility: earliest_start(i) + t̃[i] + t[i,j] ≤ latest_start(j)
                if trip1.start_time_window[0] + trip1.trip_length + t_ij <= trip2.start_time_window[1]:
                    arc = (k, trip1_id, trip2_id)
                    arcs[arc] = 1
                    base_costs[arc] = instance.meta["travel_cost"] * t_ij
                    elapsed_times[arc] = t_ij
                    energy_consumption[arc] = t_ij * THETA_FACTOR

        # --------------------------------------------------------
        # Trip_i → D_k
        # --------------------------------------------------------
        for trip_id, trip in trips_map.items():
            t_id = euclidean_distance(
                trip.end_point.y, trip.end_point.x,
                depot.location.y, depot.location.x
            )

            arc = (k, trip_id, d_node)
            arcs[arc] = 1
            base_costs[arc] = instance.meta["travel_cost"] * t_id
            elapsed_times[arc] = t_id
            energy_consumption[arc] = t_id * THETA_FACTOR

        # --------------------------------------------------------
        # Trip_i → Charging_c
        # --------------------------------------------------------
        for trip_id, trip in trips_map.items():
            for cs_id, cs_obj in charging_station_map.items():
                t_ic = euclidean_distance(
                    trip.end_point.y, trip.end_point.x,
                    cs_obj.location.y, cs_obj.location.x
                )

                arc = (k, trip_id, cs_id)
                arcs[arc] = 1
                base_costs[arc] = instance.meta["travel_cost"] * t_ic
                elapsed_times[arc] = t_ic
                energy_consumption[arc] = t_ic * THETA_FACTOR

        # --------------------------------------------------------
        # Charging_c → Trip_j
        # --------------------------------------------------------
        for cs_id, cs_obj in charging_station_map.items():
            for trip_id, trip in trips_map.items():
                t_cj = euclidean_distance(
                    cs_obj.location.y, cs_obj.location.x,
                    trip.start_point.y, trip.start_point.x
                )

                if t_cj <= trip.start_time_window[1]:
                    arc = (k, cs_id, trip_id)
                    arcs[arc] = 1
                    base_costs[arc] = instance.meta["travel_cost"] * t_cj
                    elapsed_times[arc] = t_cj
                    energy_consumption[arc] = t_cj * THETA_FACTOR

    # --- Gurobi Variables ---
    x = model.addVars(arcs.keys(), vtype=GRB.BINARY, name="x")

    time_vars_keys = set((k, i) for k, i, j in arcs.keys()) | set((k, j) for k, i, j in arcs.keys())

    T = model.addVars(time_vars_keys, vtype=GRB.CONTINUOUS, name="T")
    w = model.addVars(arcs.keys(), vtype=GRB.CONTINUOUS, lb=0.0, name="w")

    # --- Energy Variables ---
    energy_vars_keys = set((k, i) for k, i in time_vars_keys)
    E_pre = model.addVars(energy_vars_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="e")
    E_bar = model.addVars(energy_vars_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="ebar", ub=PHI_MAX)

    # g_i^k can be NEGATIVE (charging)
    G = model.addVars(energy_vars_keys, vtype=GRB.CONTINUOUS, lb=-PHI_MAX, name="g", ub=BIG_M)

    # CT_j^k: Charging completion time for vehicle k at station j
    ct_keys = [(k, j) for k in vehicles for j in charging_station_ids]
    CT = model.addVars(ct_keys, vtype=GRB.CONTINUOUS, name="CT")

    # y_j^{k1, k2}: Binary to order vehicles at charging station j
    y_keys = [(j, k1, k2) for j in charging_station_ids for k1 in vehicles for k2 in vehicles if k1 != k2]
    Y = model.addVars(y_keys, vtype=GRB.BINARY, name="Y")

    # Objective function (32)
    objective = gp.quicksum(base_costs[k, i, j] * x[k, i, j] for k, i, j in arcs.keys()) + \
                gp.quicksum(waiting_cost_lambda * w[k, i, j]
                            for k, i, j in arcs.keys()
                            if i in internal_nodes and j in internal_nodes)
    
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraint (#33)
    model.addConstrs((x.sum('*', '*', j) == 1 for j in trip_ids), name="CoverTrip")

    # Constraint (#34)
    model.addConstrs((x.sum(k, '*', v) - x.sum(k, v, '*') == 0 for k in vehicles for v in internal_nodes), name="FlowConservation")

    # Constraint (#35)
    model.addConstrs((x.sum(k, origin_nodes[k], '*') == x.sum(k, '*', dest_nodes[k]) for k in vehicles), name="ReturnToDepot")

    # Constraint (#36)
    model.addConstrs((x.sum(k, origin_nodes[k], '*') <= 1 for k in vehicles), name="StartOnce")

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

    # --- Time window constraints for T[k, i] ---
    for k, i in T.keys():

        # Constraint (#38a)
        # Trip nodes (T#)
        if i in trips_map:
            trip = trips_map[i]
            T[k, i].lb, T[k, i].ub = trip.start_time_window[0], trip.start_time_window[1]

        # Constraint (#38b)
        # Charging station nodes (C#)
        elif i in charging_station_set:
            cs_obj = charging_station_map[i]
            T[k, i].lb, T[k, i].ub = cs_obj.time_window[0], cs_obj.time_window[1]

        # Constraint (#39)
        # Depot nodes (O#, D#)
        elif i.startswith("O") or i.startswith("D"):
            T[k, i].lb, T[k, i].ub = 0, BIG_M

    # --- Time propagation & waiting cost ---
    for k, i, j in arcs.keys():
        t_ij = elapsed_times[k, i, j]
        
        if i in trip_ids:
            trip = trips_map[i]
            t_tilde = trip.trip_length
        else:
            t_tilde = 0

        # Constraint (#40) - Time propagation
        model.addConstr(T[k, j] >= T[k, i] + t_tilde + t_ij - BIG_M * (1 - x[k, i, j]), name=f"TimeProp_{k}_{i}_{j}")

        # Waiting cost only between internal nodes (trips or CSs)
        if i in internal_nodes and j in internal_nodes:
            # Constraint (#41)
            model.addConstr(w[k, i, j] >= (T[k, j] - T[k, i] - t_tilde - t_ij) - BIG_M * (1 - x[k, i, j]), name=f"WaitCostLB_{k}_{i}_{j}")

            # Constraint (#42)
            model.addConstr(w[k, i, j] <= (T[k, j] - T[k, i] - t_tilde - t_ij) + BIG_M * (1 - x[k, i, j]), name=f"WaitCostUB_{k}_{i}_{j}")

            # Constraint (#43)
            model.addConstr(w[k, i, j] <= BIG_M * x[k, i, j], name=f"WaitCostZero_{k}_{i}_{j}")

    # ==================================================
    # Energy Consumption constraints
    # ==================================================
    for k in vehicles:
        o_node = origin_nodes[k]
        d_node = dest_nodes[k]

        # Constraint (#44)
        model.addConstr(E_bar[k, o_node] == PHI_MAX, name=f"EB_StartMax_{k}")

        # Nodes for this vehicle
        nodes_for_k = [i for k_i, i in E_pre.keys() if k_i == k]

        for i in internal_nodes:
            # Constraint (#45)
            model.addConstr(E_bar[k, i] == E_pre[k, i] - G[k, i], name=f"EB_BufferUpdate_{k}_{i}")

        # Energy propagation on arcs
        for i, j in [(i, j) for _k, i, j in arcs.keys() if _k == k]:
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
    for k in vehicles:
        for j in charging_station_set:
            for i in trip_ids:
                charging_duration = (E_bar[k, j] - E_pre[k, j]) / CHARGING_RATE_KWH_PER_MINUTE
                # Constraint (#51)
                model.addConstr(CT[k, j] <= T[k, j] + charging_duration + BIG_M * (1 - x[k, i, j]), name=f"ChargeCompTime_UB1_{k}_{i}_{j}")
                # Constraint (#52)
                model.addConstr(CT[k, j] >= T[k, j] + charging_duration - BIG_M * (1 - x[k, i, j]), name=f"ChargeCompTime_LB1_{k}_{i}_{j}")
            model.addConstr(CT[k, j] <= BIG_M * gp.quicksum(x[k, i, j] for i in trip_ids), name=f"ChargeCompTime_zero_{k}_{j}") # Constraint (53)

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

    return "\n".join(report_lines), schedules, variable_report_str

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
            f"Location: ({depot.location.x:5.2f}, {depot.location.y:5.2f}) | "
            f"Vehicles: {depot.vehicle_count}"
        )

    report_lines.append("\n--- Detailed Trip Information ---")
    point_to_id_map = {point: i + 1 for i, point in enumerate(instance.relief_points)}

    for trip in instance.trips:
        start_rp_id = point_to_id_map.get(trip.start_point, "N/A")
        end_rp_id = point_to_id_map.get(trip.end_point, "N/A")

        trip_duration = getattr(trip, "trip_length", trip.end_time - trip.start_time)
        eta_info = f" | ETA: {trip.eta:.2f}" if hasattr(trip, "eta") else ""

        report_lines.append(
            f"Trip ID: {trip.id: <3} | "
            f"Type: {trip.trip_type: <5} | "
            f"Starts at RP #{start_rp_id:<3} -> Ends at RP #{end_rp_id:<3} | "
            f"Start/End Time: {trip.start_time:7.2f} to {trip.end_time:7.2f} | "
            f"Duration: {trip_duration:.2f}{eta_info}"
        )

    # -----------------------------------------------------------
    # NEW SECTION: t_ij reporting
    # -----------------------------------------------------------
    report_lines.append("\n--- Deadhead Travel Times (t_ij) ---")

    trips = instance.trips
    depots = instance.depots

    # Helper for Euclidean distance
    def dist(p1, p2):
        return euclidean_distance(p1.y, p1.x, p2.y, p2.x)

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

    # 1. Load instance
    print("--- Part 1: Loading instance from file ---")
    instance_path = os.path.join(project_root, "..", "input", "test_instances", "D2_S2_C10_a_trips.txt")
    instance = load_instance_from_txt(instance_path)
    print("Instance successfully loaded.")

    # 2. Set up output directory
    print("--- Part 2: Preparing Output Directory ---")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, "..", "output", f"EB_TW_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    bus_blocks_dir = os.path.join(output_dir, "bus_blocks")
    os.makedirs(bus_blocks_dir, exist_ok=True)

    # 3. Save instance report and plots
    print(f"--- Part 3: Saving instance details and plots to '{output_dir}' ---")
    instance_report = get_instance_report(instance)

    plotting_utils_eb.save_instance_plot(instance, os.path.join(output_dir, "instance_plot.png"))
    trip_y = plotting_utils_eb.save_dag_plot(instance, os.path.join(output_dir, "instance_dag.png"))

    # 4. Solve instance
    print("--- Part 4: Solving the Instance ---")
    solution_report, schedules, variable_report_str = solve_md_vsp_tw_from_instance(instance)

    # 5. Save solution plots if a solution was found
    print(schedules)
    if schedules:
        plotting_utils_eb.save_solution_dag_plot(
            instance,
            schedules,
            os.path.join(output_dir, "solved_instance_dag.png"),
            trip_y=trip_y
        )
        plotting_utils_eb.save_solution_plot(instance, schedules, bus_blocks_dir)

    # 6. Write combined report to file
    print("--- Part 5: Writing Full Report to File ---")
    full_report_path = os.path.join(output_dir, "instance_and_solution_report.txt")
    with open(full_report_path, "w", encoding='utf-8') as f:
        f.write("--- INSTANCE INFORMATION ---\n")
        f.write(instance_report)
        f.write("\n\n--- SOLUTION ---\n")
        f.write(solution_report)

        if variable_report_str:  # Only write if the string is not empty
            f.write("\n\n--- SOLUTION VARIABLES ---")
            f.write(variable_report_str)

    print(f"\nProcess complete. All results saved in '{output_dir}'")
