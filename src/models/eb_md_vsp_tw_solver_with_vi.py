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
from shapely.geometry import Point
import csv
from scipy.spatial import distance

from typing import Optional

import glob

# --- Add parent directory to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities.instance_generator import Point, Depot, Trip, ChargingStation, ProblemInstance
import plotting.plotting_utils_eb as plotting_utils_eb

# --- Constants for the Energy Buffer (EB) Model Extension ---
SMALL_M = 0.00001
BIG_M = 1000000

# ============================================================
# Instance Loader
# ============================================================

def reset_bus_end_states_dir(dir_path: str) -> None:
    if os.path.isdir(dir_path):
        for f in glob.glob(os.path.join(dir_path, "bus_end_states_*.json")):
            os.remove(f)
    else:
        os.makedirs(dir_path, exist_ok=True)

def latest_bus_end_states_file(folder: str) -> Optional[str]:
    files = glob.glob(os.path.join(folder, "bus_end_states_*.json"))
    return max(files, key=os.path.getmtime) if files else None

def load_instance_from_txt(path: str):
    """
    Loads a ProblemInstance matching the logic of Model 2 (EB-MDVSPTW...py).
    
    Structure:
      - Header
      - K Origin Depot Rows (for Vehicles 1..K)
      - K Destination Depot Rows (for Vehicles 1..K)
      - T Trip Rows
      - F Charger Rows
    """

    bus_end_states_dir = os.path.join(project_root, "..", "output", "bus_end_states")
    latest_states_file = latest_bus_end_states_file(bus_end_states_dir)

    bus_end_states = None
    if latest_states_file:
        with open(latest_states_file, "r") as f:
            bus_end_states = json.load(f)

    # ============================================================
    # 1) READ FILE
    # ============================================================

    with open(path, "r") as f:
        reader = list(csv.reader(f, delimiter="\t"))

    if len(reader) < 5:
        raise ValueError("Instance file must have at least 5 rows.")

    # ---------------------------
    # HEADER
    # ---------------------------

    # header = list(map(float, reader[0]))

    # NEW: (Force Integer truncation for counts and rates, matching Script B)
    header_float = list(map(float, reader[0]))
    header = [float(x) for x in header_float]

    # Model 2 reads these exact indices 
    K = int(header[0])           # Vehicles
    T_trips = int(header[1])     # Trips
    F_chargers = int(header[2])  # Charging Events
    greek_l = header[3]          
    p_max = header[4]            
    p_min = header[5]            
    travel_cost = header[6]      
    charging_rate = header[7]    
    theta_factor = header[8]

    # Map header values into global constants
    global PHI_MAX, PHI_MIN, CHARGING_RATE_KWH_PER_MINUTE, THETA_FACTOR
    PHI_MAX = p_max
    PHI_MIN = p_min
    CHARGING_RATE_KWH_PER_MINUTE = charging_rate
    THETA_FACTOR = theta_factor

    # ---------------------------
    # BODY SLICING (Dynamic based on K)
    # ---------------------------
    body = reader[1:]  # everything after header

    # Model 2 logic: 2 * Vehicles are depot rows (Origins + Destinations) 
    num_depot_rows = 2 * K
    
    # Calculate expected rows based on header counts
    # Note: Sometimes F_chargers in header matches rows exactly, sometimes we just read the rest.
    # We'll use the counts to be safe but allow reading to end of file for chargers.
    expected_min_rows = num_depot_rows + T_trips
    
    if len(body) < expected_min_rows:
        raise ValueError(
            f"Body has {len(body)} rows but at least {expected_min_rows} "
            f"are required ({num_depot_rows} depot rows + {T_trips} trips)."
        )

    # Slice the data
    # Rows 0 to K-1: Origins
    # Rows K to 2K-1: Destinations
    # Rows 2K to 2K+T-1: Trips
    # Remaining: Chargers
    
    depot_rows = body[0 : num_depot_rows]
    trip_rows = body[num_depot_rows : num_depot_rows + T_trips]
    start_chargers = num_depot_rows + T_trips
    end_chargers = start_chargers + F_chargers
    charger_rows = body[start_chargers : end_chargers]

    # ============================================================
    # 2) PARSE DEPOTS
    # ============================================================

    depots = []
    trips = []
    charging_stations = []
    relief_points = []

    # Precompute equal vehicle distribution among the first K origin rows
    num_depots = K
    base = K // num_depots
    remainder = K % num_depots

    assigned = 0  # how many depots have received the remainder bonus

    # Model 2 logic: O[j] = data[k] and D[j] = data[k + Vehicles]
    # This means the first K rows are Origins. The next K rows are Destinations.

    for idx, row in enumerate(depot_rows):
        if not row:
            continue

        node_id = int(row[0])
        lat_o, lon_o = float(row[1]), float(row[2])

        p = Point(lon_o, lat_o)
        relief_points.append(p)

        if idx < K:
            extra = 1 if assigned < remainder else 0
            vehicle_count = base + extra
            if extra == 1:
                assigned += 1

            depot = Depot(
                id=node_id,
                location=p,
                vehicle_count=vehicle_count
            )
            depots.append(depot)

        else:
            pass

    # ============================================================
    # 3) PARSE TRIPS
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

        # Compute pure Euclidean distance (Model 2: data_header_decimal[8] is consumption, [6] is cost)
        # Model 2 uses 'distance.euclidean' for travel time 
        dist_val = distance.euclidean([lat_o, lon_o], [lat_d, lon_d])

        # Compute ETA (Energy)
        eta_value = dist_val * theta_factor

        trip = Trip(
            id=node_id,
            start_point=p_o,
            end_point=p_d,
            start_time=t_earliest,
            end_time=t_latest,
            trip_type="REGULAR"
        )

        trip.start_time_window = (t_earliest, t_latest)
        trip.eta = eta_value
        
        # CRITICAL: Model 2 uses pure distance for time, NOT multiplied by cost 
        # (travel_cost is applied only in the objective function)
        trip.trip_length = dist_val

        trips.append(trip)

    # ============================================================
    # 4) PARSE CHARGING STATIONS
    # ============================================================
    
    # Dictionary to track unique physical locations to merge time slots (if desired)
    # OR just read them linearly if you are using the "No Pruning" solver fix.
    # Given you are removing pruning in the solver, we can read them as list.

    for row in charger_rows:
        if len(row) < 7: continue 
        
        node_id = int(row[0])
        lat_o, lon_o = float(row[1]), float(row[2])
        t_earliest = float(row[5])
        t_latest = float(row[6])

        p_o = Point(lon_o, lat_o)
        relief_points.append(p_o)

        cs = ChargingStation(
            id=node_id,
            location=p_o,
            time_window=(t_earliest, t_latest)
        )
        charging_stations.append(cs)

    # ============================================================
    # 5) BUILD ProblemInstance
    # ============================================================

    grid_size = (60, 60)

    instance = ProblemInstance(
        grid_size=grid_size,
        trips=trips,
        depots=depots,
        relief_points=relief_points,
        charging_stations=charging_stations
    )

    instance.meta = {
        "K": K,
        "T": T_trips,
        "F": len(charging_stations),
        "lambda": greek_l,
        "p_max": p_max,
        "p_min": p_min,
        "travel_cost": travel_cost,
        "charging_rate": charging_rate,
        "theta_factor": theta_factor
    }

    # --------------------------------------------------
    # Inject bus_end_states into instance.meta (forward pass)
    # --------------------------------------------------

    total_vehicles = sum(d.vehicle_count for d in depots)

    # Defaults if this is the first run
    initial_soc = [PHI_MAX] * total_vehicles
    availability_times = [0.0] * total_vehicles
    buses_state = [True] * total_vehicles

    if bus_end_states:
        vehicles_sorted = []
        for depot in depots:
            for i in range(1, depot.vehicle_count + 1):
                vehicles_sorted.append(f"D{depot.id}_V{i}")

        for idx, v_id in enumerate(vehicles_sorted):
            if v_id in bus_end_states:
                initial_soc[idx] = bus_end_states[v_id]["soc"]
                availability_times[idx] = bus_end_states[v_id]["arrival_time"]
                buses_state[idx] = bus_end_states[v_id]["buses_state"]

    instance.meta.update({
        "buses_initial_soc": initial_soc,
        "buses_availability_times": availability_times,
        "bus_end_states_source": latest_states_file,
        "buses_state": buses_state
    })

    return instance

# ============================================================
# Solver
# ============================================================

def solve_md_vsp_tw_from_instance(
    instance: ProblemInstance,
    waiting_cost_lambda: float = 1.3,
    output_dir: str = None,
    bus_end_states_dir: str = None,
    run_timestamp: str = None
):
    """
    Solves the MD-VSP-TW with an Energy Buffer extension and returns the report.

    IMPORTANT:
    - Trip execution duration is given by trip.trip_length (computed in the loader)
    - Trip start_time and end_time define the earliest and latest SERVICE START time (time windows).
    """

    model = gp.Model("MD-VSP-TW-EB")

    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', 18000)

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
            l_i, u_i = trip.start_time_window
            
            t_oi = distance.euclidean([depot.location.y, depot.location.x], [trip.start_point.y, trip.start_point.x])

            # === VALID INEQUALITY 1 (Depot→Trip): t_oi <= u_i ===
            # (CAN COMMENT OUT THIS "if" TO REMOVE THE VALID INEQUALITY)
            if t_oi <= u_i:
                arc = (k, o_node, trip_id)
                arcs[arc] = 1
                base_costs[arc] = instance.meta["travel_cost"] * t_oi
                elapsed_times[arc] = t_oi
                energy_consumption[arc] = t_oi * THETA_FACTOR

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

                t_ij = distance.euclidean([trip1.end_point.y, trip1.end_point.x], [trip2.start_point.y, trip2.start_point.x])

                # === VALID INEQUALITY 1 (Trip→Trip): l_i + dur_i + t_ij <= u_j ===
                # (CAN COMMENT OUT THIS "if" TO REMOVE THE VALID INEQUALITY)
                if l_i + dur_i + t_ij <= u_j:
                    arc = (k, trip1_id, trip2_id)
                    arcs[arc] = 1
                    base_costs[arc] = instance.meta["travel_cost"] * t_ij
                    elapsed_times[arc] = t_ij
                    energy_consumption[arc] = t_ij * THETA_FACTOR

        # --------------------------------------------------------
        # Trip_i → D_k
        # --------------------------------------------------------
        for trip_id, trip in trips_map.items():
    
            t_id = distance.euclidean([trip.end_point.y, trip.end_point.x], [depot.location.y, depot.location.x])

            # (NO time-window VI from Set 1 applies here)
            arc = (k, trip_id, d_node)
            arcs[arc] = 1
            base_costs[arc] = instance.meta["travel_cost"] * t_id
            elapsed_times[arc] = t_id
            energy_consumption[arc] = t_id * THETA_FACTOR

        # --------------------------------------------------------
        # Trip_i → Charging_c
        # --------------------------------------------------------
        for trip_id, trip in trips_map.items():
            l_i, u_i = trip.start_time_window
            dur_i = trip.trip_length

            for cs_id, cs_obj in charging_station_map.items():

                # You may define cs_obj.time_window; otherwise use full horizon:
                l_c, u_c = getattr(cs_obj, "time_window", (0, float("inf")))

                t_ic = distance.euclidean([trip.end_point.y, trip.end_point.x], [cs_obj.location.y, cs_obj.location.x])

                # === VALID INEQUALITY 1 (Trip→Charging): l_i + dur_i + t_ic <= u_c ===
                # (CAN COMMENT OUT THIS "if" TO REMOVE THE VALID INEQUALITY)
                # if l_i + dur_i + t_ic <= u_c:
                arc = (k, trip_id, cs_id)
                arcs[arc] = 1
                base_costs[arc] = instance.meta["travel_cost"] * t_ic
                elapsed_times[arc] = t_ic
                energy_consumption[arc] = t_ic * THETA_FACTOR

        # --------------------------------------------------------
        # Charging_c → Trip_j
        # --------------------------------------------------------
        for cs_id, cs_obj in charging_station_map.items():

            # Again: if no window exists, assume full availability
            l_c, u_c = getattr(cs_obj, "time_window", (0, float("inf")))
            tau_c = getattr(cs_obj, "min_charge_time", 0)  # minimal charging duration

            for trip_id, trip in trips_map.items():
                l_j, u_j = trip.start_time_window

                t_cj = distance.euclidean([cs_obj.location.y, cs_obj.location.x], [trip.start_point.y, trip.start_point.x])

                # === VALID INEQUALITY 1 (Charging→Trip):
                #     l_c + τ_c + t_cj <= u_j
                # (CAN COMMENT OUT THIS "if" TO REMOVE THE VALID INEQUALITY)
                # if l_c + tau_c + t_cj <= u_j:
                arc = (k, cs_id, trip_id)
                arcs[arc] = 1
                base_costs[arc] = instance.meta["travel_cost"] * t_cj
                elapsed_times[arc] = t_cj
                energy_consumption[arc] = t_cj * THETA_FACTOR

    # --- Gurobi Variables ---
    x = model.addVars(arcs.keys(), vtype=GRB.BINARY, name="x")

    time_vars_keys = set((k, i) for k, i, j in arcs.keys()) | set((k, j) for k, i, j in arcs.keys())

    T = model.addVars(time_vars_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    w = model.addVars(arcs.keys(), vtype=GRB.CONTINUOUS, lb=0.0, name="w")

    starting_node_keys = set((k, i) for k in vehicles for i in origin_nodes.values() for j in all_nodes if (k, i, j) in arcs.keys())
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

    lambda_1 = waiting_cost_lambda
    lambda_2 = 0 # this is kept to zero in order to have comparable objective function values with classical formulations
    lambda_3 = waiting_cost_lambda # same as lamda 2

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

    # Mathematical expression (37)
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
        if (j in trip_ids) or (j in charging_station_ids):
            
            model.addConstr(w[k,i,j] >= T[k, j] - (T[k, i] + trip_service + t_ij) - BIG_M*(1 - x[k, i, j]))

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

        initial_soc = instance.meta["buses_initial_soc"]

        vehicle_index = vehicles.index(k)

        model.addConstr(E_bar[k, o_node] == initial_soc[vehicle_index], name=f"EB_StartSOC_{k}") # Constraint (#44)

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
    #                 model.addConstr(Y[j, k1, k2] <= BIG_M * gp.quicksum(x[k2, i, j] for i in trip_ids if (k, i, j) in arcs.keys()), name=f"zero_Y_2{k}_{j}") # Constraint (53)

    # ============================================================
    # Valid Inequalities
    # ============================================================

    # Set 1 is included in the arc generation process above, 
    # Set 3 can not be included in this model because of the continuous charging time horizon.

    # ============================================================
    # Valid Inequality Set 2: SOC reachability from trip j
    # ============================================================

    # Precompute min energy needed from each trip to nearest charger
    min_energy_to_cs = {}

    for trip_id, trip in trips_map.items():
        best = float("inf")
        for cs_id, cs in charging_station_map.items():
            dist = distance.euclidean([trip.end_point.y, trip.end_point.x], [cs.location.y, cs.location.x])
            energy = dist * THETA_FACTOR
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
    # - For each vehicle k and internal node v:
    #     * at most one outgoing arc
    #     * at most one incoming arc
    #   This strengthens the LP relaxation beyond flow conservation.
    # ============================================================

    # Outgoing-arc conflicts: sum_j x[k, v, j] ≤ 1
    for k in vehicles:
        for v in trip_ids:
            outgoing_arcs = [(k, v, j) for (_k, i, j) in arcs.keys() if _k == k and i == v]
            if outgoing_arcs:
                model.addConstr(gp.quicksum(x[a] for a in outgoing_arcs) <= 1, name=f"VI5_out_{k}_{v}")

    # Incoming-arc conflicts: sum_i x[k, i, v] ≤ 1
    for k in vehicles:
        for v in trip_ids:
            incoming_arcs = [(k, i, v) for (_k, i, j) in arcs.keys() if _k == k and j == v]
            if incoming_arcs:
                model.addConstr(gp.quicksum(x[a] for a in incoming_arcs) <= 1, name=f"VI5_in_{k}_{v}")

    # model.setParam('MIPGap', 0.2)
    # model.setParam("Presolve", 0)

    # 1) Encode the 2006 solution arcs
    # target_arcs = {
    #     # D11_V1 route
    #     ('D11_V1', 'O11', 'T8'),
    #     ('D11_V1', 'T8',  'T2'),
    #     ('D11_V1', 'T2',  'C1'),
    #     ('D11_V1', 'C1',  'T9'),
    #     ('D11_V1', 'T9',  'T5'),
    #     ('D11_V1', 'T5',  'D11'),

    #     # D12_V1 route
    #     ('D12_V1', 'O12', 'T1'),
    #     ('D12_V1', 'T1',  'T4'),
    #     ('D12_V1', 'T4',  'T10'),
    #     ('D12_V1', 'T10', 'C2'),
    #     ('D12_V1', 'C2',  'T3'),
    #     ('D12_V1', 'T3',  'T7'),
    #     ('D12_V1', 'T7',  'T6'),
    #     ('D12_V1', 'T6',  'D12'),
    # }

    # # 2) Force those arcs on, all others off
    # for (k, i, j) in x.keys():
    #     if (k, i, j) in target_arcs:
    #         x[k, i, j].LB = 1.0
    #         x[k, i, j].UB = 1.0
    #     else:
    #         x[k, i, j].UB = 0.0

    model.update()
    model.write("model.lp")
    model.optimize()

    default_time_var = type('obj', (object,), {'X': 0.0})  # Helper for safe .X access

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:

        bus_end_states = {}
        buses_state = {}

        for k in vehicles:

            used = any(
                x[k, i, j].X > 0.5
                for (kk, i, j) in x.keys()
                if kk == k
            )

            final_node = dest_nodes[k]

            returned = any(
                (k, i, final_node) in x and x[k, i, final_node].X > 0.5
                for i in all_nodes
            )

            if not returned:
                continue

            arrival_time = T[k, final_node].X
            soc = E_pre[k, final_node].X
            depot_id = int(final_node[1:])

            bus_end_states[k] = {
                "buses_state": 0 if used else 1,
                "depot_id": depot_id,
                "arrival_time": round(arrival_time, 4),
                "soc": round(soc, 4),
            }

        if bus_end_states_dir is None or run_timestamp is None:
            raise RuntimeError(
                "bus_end_states_dir and run_timestamp must be provided by the caller"
            )

        os.makedirs(bus_end_states_dir, exist_ok=True)

        bus_end_states_path = os.path.join(
            bus_end_states_dir,
            f"bus_end_states_{run_timestamp}.json"
        )

        with open(bus_end_states_path, "w") as f:
            json.dump(bus_end_states, f, indent=4)

        VAR = model.NumVars
        CNS = model.NumConstrs
        NE  = model.NodeCount
        SI  = model.IterCount

        SP  = model.ObjVal
        LB  = model.ObjBound
        LBG = (SP - LB) / SP * 100 if SP != 0 else 0
        OG  = model.MIPGap * 100

        CPT  = model.Runtime # computation time in minutes

        # Header
        print("\n" + "="*110)
        print(f"{'Instance':<12} {'VAR':>7} {'CNS':>7} {'B&B NE':>10} {'SI':>10} "
            f"{'SP':>12} {'LB':>12} {'LBG (%)':>10} {'OG (%)':>10} {'CPT (s)':>10}")
        print("-"*110)

        # Row
        print(f"{'Instance':<12} {VAR:>7,} {CNS:>7,} {NE:>10,} {SI:>10,} "
            f"{SP:>12.2f} {LB:>12.2f} {LBG:>10.2f} {OG:>10.2f} {CPT:>10.2f}")

        print("="*110)

        # --- Reporting Logic (Updated to include Energy Variables) ---
        report_lines, schedules = [], {}
        variable_report_str = ""
        solution_data = {}
        default_time_var = type('obj', (object,), {'X': 0.0})  # Helper for safe .X access

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
            extract_vars(tau, "tau")
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

    solver_status = model.Status
    objective_value = model.ObjVal if model.SolCount > 0 else None

    return report_lines, schedules, variable_report_str, {
        "VAR": VAR,
        "CNS": CNS,
        "NE": NE,
        "SI": SI,
        "SP": SP,
        "LB": LB,
        "LBG": LBG,
        "OG": OG,
        "CPT": CPT
    }, bus_end_states, solver_status, objective_value

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
    # t_ij reporting
    # -----------------------------------------------------------
    report_lines.append("\n--- Deadhead Travel Times (t_ij) ---")

    trips = instance.trips
    depots = instance.depots

    # Helper for Euclidean distance
    def dist(p1, p2):
        return distance.euclidean([p1.y, p1.x], [p2.y, p2.x])

    # Depot -> Trip
    report_lines.append("\nDepot → Trip:")
    for depot in depots:
        for trip in trips:
            tij = dist(depot.location, trip.start_point)
            report_lines.append(f"  Depot {depot.id} -> Trip {trip.id}: t_ij = {tij:.2f}")

    # Trip -> Trip
    report_lines.append("\nTrip → Trip:")
    for t1 in trips:
        for t2 in trips:
            if t1.id == t2.id:
                continue
            tij = dist(t1.end_point, t2.start_point)
            report_lines.append(f"  Trip {t1.id} -> Trip {t2.id}: t_ij = {tij:.2f}")

    # Trip -> Depot
    report_lines.append("\nTrip → Depot:")
    for trip in trips:
        for depot in depots:
            tij = dist(trip.end_point, depot.location)
            report_lines.append(f"  Trip {trip.id} -> Depot {depot.id}: t_ij = {tij:.2f}")

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
    # If user passed an argument, use it; otherwise fall back to the default
    if len(sys.argv) > 1:
        instance_path = sys.argv[1]
    else:
        instance_path = os.path.join(project_root, "..", "input", "test_instances", "instance_2.txt")

    bus_end_states_output_dir = os.path.join(project_root, "..", "output", "bus_end_states")
    
    # reset_bus_end_states_dir(bus_end_states_output_dir)

    instance = load_instance_from_txt(instance_path)

    print("Instance successfully loaded.")

    # 2. Set up output director
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
    solution_report, schedules, variable_report_str, stats, _, _, _ = solve_md_vsp_tw_from_instance(
        instance,
        output_dir=output_dir,
        bus_end_states_dir=bus_end_states_output_dir,
        run_timestamp=timestamp
    )

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
        f.write("\n".join(solution_report))

        if variable_report_str:  # Only write if the string is not empty
            f.write("\n\n--- SOLUTION VARIABLES ---")
            f.write(variable_report_str)

    print(f"\nProcess complete. All results saved in '{output_dir}'")
