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

from utilities.instance_generator import GkiotsalitisInstanceGenerator, ProblemInstance
import plotting.plotting_utils_eb as plotting_utils_eb
from utilities import haversine

# --- Constants for the Energy Buffer (EB) Model Extension ---
PHI_MAX = 350.0                         # Max energy level
PHI_MIN = 200.0                          # Min required energy level
THETA_FACTOR = 0.00001                  # in kWh/meter
ETA_TRIP = 0.0                          # Used as fallback/default
SMALL_M = 0.0001
BIG_M = 10000
AVG_U_METERS_PER_MINUTE = 800000        # Average speed: 26,000 meters / 60 minutes = 433.33 meters/minute
CHARGING_RATE_KWH_PER_MINUTE = 3.0      # Charging rate: 180 kWh / 60 minutes = 3 kWh/minute

# --- Main Solver Function ---

def solve_md_vsp_tw_from_instance(instance: ProblemInstance, waiting_cost_lambda: float = 2.0):
    """
    Solves the MD-VSP-TW with an Energy Buffer extension and returns the report.
    """
    model = gp.Model("MD-VSP-TW-EB")

    model.setParam('OutputFlag', 1)

    # (Instance processing logic remains unchanged)
    vehicles, vehicle_to_depot = [], {}
    for depot in instance.depots:
        for i in range(1, depot.vehicle_count + 1):
            vehicle_id = f"D{depot.id}_V{i}"; vehicles.append(vehicle_id)
            vehicle_to_depot[vehicle_id] = depot

    trips_map = {f"T{trip.id}": trip for trip in instance.trips}
    trip_ids = list(trips_map.keys())

    # Give each charging station a unique ID (e.g., 'C1', 'C2', ...)
    charging_station_map = {f"C{i+1}": cs for i, cs in enumerate(instance.charging_stations)}
    charging_station_ids = list(charging_station_map.keys())
    charging_station_set = set(charging_station_ids)
    internal_nodes = trip_ids + charging_station_ids
    
    # -------------------------------------

    origin_nodes = {v_id: f"O{vehicle_to_depot[v_id].id}" for v_id in vehicles}
    dest_nodes = {v_id: f"D{vehicle_to_depot[v_id].id}" for v_id in vehicles}

    # Arcs calculation, adding new energy data
    arcs, base_costs, elapsed_times, energy_consumption = {}, {}, {}, {}
    for k in vehicles:
        depot, o_node, d_node = vehicle_to_depot[k], origin_nodes[k], dest_nodes[k]
        
        # O_k -> Trip_i Arcs
        for trip_id, trip in trips_map.items():
            deadhead_distance = haversine.main(depot.location.y, depot.location.x, trip.start_point.y, trip.start_point.x)
            deadhead_time = deadhead_distance / AVG_U_METERS_PER_MINUTE
            if 0 + deadhead_time <= trip.start_time_window[1]:
                arc = (k, o_node, trip_id); arcs[arc] = 1
                base_costs[arc] = math.floor(10 * deadhead_time) + 5000
                elapsed_times[arc] = deadhead_time
                energy_consumption[arc] = deadhead_distance * THETA_FACTOR
                
        # Trip_i -> Trip_j Arcs
        for trip1_id, trip1 in trips_map.items():
            for trip2_id, trip2 in trips_map.items():
                if trip1_id == trip2_id: continue
                deadhead_distance = haversine.main(trip1.end_point.y, trip1.end_point.x, trip2.start_point.y, trip2.start_point.x)
                deadhead_time = deadhead_distance / AVG_U_METERS_PER_MINUTE
                trip1_duration = trip1.end_time - trip1.start_time
                t_ij = trip1_duration + deadhead_time
                if trip1.start_time_window[0] + t_ij <= trip2.start_time_window[1]:
                    arc = (k, trip1_id, trip2_id); arcs[arc] = 1
                    base_costs[arc] = math.floor(10 * deadhead_time)
                    elapsed_times[arc] = t_ij
                    energy_consumption[arc] = deadhead_distance * THETA_FACTOR

        # Trip_i -> D_k Arcs
        for trip_id, trip in trips_map.items():
            deadhead_distance = haversine.main(trip.end_point.y, trip.end_point.x, depot.location.y, depot.location.x)
            deadhead_time = deadhead_distance / AVG_U_METERS_PER_MINUTE
            trip_duration = trip.end_time - trip.start_time
            arc = (k, trip_id, d_node); arcs[arc] = 1
            base_costs[arc] = math.floor(10 * deadhead_time)
            elapsed_times[arc] = trip_duration + deadhead_time
            energy_consumption[arc] = deadhead_distance * THETA_FACTOR

        # Trip_i -> Charging_c Arcsa
        for trip1_id, trip1 in trips_map.items():
            for cs_id, cs_obj in charging_station_map.items():
                
                p1 = trip1.end_point
                p2 = cs_obj.location
                
                deadhead_distance = haversine.main(p1.y, p1.x, p2.y, p2.x) # This line now works
                deadhead_time = deadhead_distance / AVG_U_METERS_PER_MINUTE
                
                trip1_duration = trip1.end_time - trip1.start_time
                # Elapsed time from T_i start to C_c arrival
                t_ij = trip1_duration + deadhead_time
                
                arc = (k, trip1_id, cs_id); arcs[arc] = 1
                base_costs[arc] = math.floor(10 * deadhead_time)
                elapsed_times[arc] = t_ij
                energy_consumption[arc] = deadhead_distance * THETA_FACTOR

        # Charging_c -> Trip_j Arcs (Leaving charging station to start a new trip)
        for cs_id, cs_obj in charging_station_map.items():
            for trip2_id, trip2 in trips_map.items():
                
                p1 = cs_obj.location
                p2 = trip2.start_point
                
                deadhead_distance = haversine.main(p1.y, p1.x, p2.y, p2.x) # This line now works
                deadhead_time = deadhead_distance / AVG_U_METERS_PER_MINUTE
                
                # Check feasibility against T_j's time window.
                # Use a relaxed check, actual feasibility must be ensured by Gurobi constraints 
                # involving T_charge (which is missing but required for correctness).
                if 0 + deadhead_time <= trip2.start_time_window[1]:
                    arc = (k, cs_id, trip2_id); arcs[arc] = 1
                    base_costs[arc] = math.floor(10 * deadhead_time)
                    elapsed_times[arc] = deadhead_time # Only deadhead time!
                    energy_consumption[arc] = deadhead_distance * THETA_FACTOR

    # --- Gurobi Variables ---
    x = model.addVars(arcs.keys(), vtype=GRB.BINARY, name="x")
    
    time_vars_keys = set((k, i) for k, i, j in arcs.keys()) | set((k, j) for k, i, j in arcs.keys())
    
    T = model.addVars(time_vars_keys, vtype=GRB.CONTINUOUS, name="T")
    w = model.addVars(arcs.keys(), vtype=GRB.CONTINUOUS, lb=0.0, name="w")

    # --- NEW: Energy Variables ---
    energy_vars_keys = set((k, i) for k, i in time_vars_keys)
    # e_j^k (Energy BEFORE replenishment/final level)
    E_pre = model.addVars(energy_vars_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="e")
    # \bar{e}_j^k (Energy AFTER replenishment/buffer)
    E_bar = model.addVars(energy_vars_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="ebar", ub=PHI_MAX)

    # CRITICAL: g_i^k must be able to be NEGATIVE for charging. Set lb appropriately.
    G = model.addVars(energy_vars_keys, vtype=GRB.CONTINUOUS, lb=-PHI_MAX, name="g", ub=BIG_M)

    # CT_j^k: Charging Completion Time for vehicle k at charging station j
    ct_keys = [(k, j) for k in vehicles for j in charging_station_ids]
    CT = model.addVars(ct_keys, vtype=GRB.CONTINUOUS, name="CT")

    # y_j^{k1, k2}: Binary to order vehicles at charging station j
    y_keys = [(j, k1, k2) for j in charging_station_ids for k1 in vehicles for k2 in vehicles if k1 != k2]
    Y = model.addVars(y_keys, vtype=GRB.BINARY, name="Y")

    # Mathematical expression (32)
    objective = gp.quicksum(base_costs[k, i, j] * x[k, i, j] for k, i, j in arcs.keys()) + \
                 gp.quicksum(waiting_cost_lambda * w[k, i, j] for k, i, j in arcs.keys() if i in internal_nodes and j in internal_nodes)

    model.setObjective(objective, GRB.MINIMIZE)

    # Mathematical expression (33)
    model.addConstrs((x.sum('*', '*', j) == 1 for j in trip_ids), name="CoverTrip")

    # Mathematical expression (34)
    model.addConstrs((x.sum(k, '*', v) - x.sum(k, v, '*') == 0 for k in vehicles for v in internal_nodes), name="FlowConservation")

    # Mathematical expression (35)
    model.addConstrs((x.sum(k, origin_nodes[k], '*') == x.sum(k, '*', dest_nodes[k]) for k in vehicles), name="ReturnToDepot")

    # Mathematical expression (36)
    model.addConstrs((x.sum(k, origin_nodes[k], '*') <= 1 for k in vehicles), name="StartOnce")

    # Mathematical expression (37)
    depot_to_vehicles = {}
    for depot in instance.depots:
        depot_id_str = f"O{depot.id}" 
        depot_to_vehicles[depot_id_str] = [v_id for v_id in vehicles if origin_nodes[v_id] == depot_id_str]

    depots_map = {depot.id: depot for depot in instance.depots}
    for origin_node_id, vehicles_in_depot in depot_to_vehicles.items():
        depot_id_num = int(origin_node_id[1:])
        depot_obj = depots_map[depot_id_num]
        
        model.addConstr(
            gp.quicksum(x[k, origin_node_id, j] for k in vehicles_in_depot for j in trip_ids if (k, origin_node_id, j) in x) <= depot_obj.vehicle_count, name=f"DepotCapacity_{origin_node_id}")

    # --- Constraints (24) to (29) - Time Windows and Waiting Cost ---
    
    for k, i in T.keys():
        
        # Mathematical expression (38)
        if i in trips_map:
            # Case 1: Trip Nodes (T#)
            trip = trips_map[i]
            T[k, i].lb, T[k, i].ub = trip.start_time_window[0], trip.start_time_window[1] 
            
        # --- NEW: Case 2: Charging Station Nodes (C#) ---
        elif i in charging_station_set:
            # Look up the CS object from the instance data
            cs_obj = charging_station_map[i]
            # FIX 3: Access time_window directly from the cs_obj
            T[k, i].lb, T[k, i].ub = cs_obj.time_window[0], cs_obj.time_window[1]
        # ----------------------------------------------------
            
        # Mathematical expression (39)
        elif i.startswith("O") or i.startswith("D"): # Case 3: Depot Nodes (O# and D#)
            T[k, i].lb, T[k, i].ub = 0, BIG_M
    
    for k, i, j in arcs.keys():
        t_ij = elapsed_times[k, i, j]
        # Mathematical expression (40)
        model.addConstr(T[k, j] >= T[k, i] + t_ij - BIG_M * (1 - x[k, i, j]), name=f"TimeProp_{k}_{i}_{j}") 
        # model.addConstr(T[k, j] <= T[k, i] + t_ij + BIG_M * (1 - x[k, i, j]), name=f"TimeProp_{k}_{i}_{j}")
        
        if i in internal_nodes and j in internal_nodes:
            # Mathematical expression (41)
            model.addConstr(w[k, i, j] >= (T[k, j] - T[k, i] - t_ij) - BIG_M * (1 - x[k, i, j]), name=f"WaitCostLB_{k}_{i}_{j}") 
            # Mathematical expression (42)
            model.addConstr(w[k, i, j] <= (T[k, j] - T[k, i] - t_ij) + BIG_M * (1 - x[k, i, j]), name=f"WaitCostUB_{k}_{i}_{j}")
            # Mathematical expression (43)
            model.addConstr(w[k, i, j] <= BIG_M * x[k, i, j], name=f"WaitCostZero_{k}_{i}_{j}") 

    # ===============================================
    # Energy Consumption constraints
    # ===============================================
    
    for k in vehicles:
        o_node, d_node = origin_nodes[k], dest_nodes[k]
        
        # Mathematical expression (44)
        model.addConstr(E_bar[k, o_node] == PHI_MAX, name=f"EB_StartMax_{k}")

        # CRITICAL FIX: Get only the nodes i that belong to the current vehicle k's paths.
        nodes_for_k = [i for k_i, i in E_pre.keys() if k_i == k]

        for i in internal_nodes: # Use the restricted node set      
            # Mathematical expression (45)
            model.addConstr(E_bar[k, i] == E_pre[k, i] - G[k, i], name=f"EB_BufferUpdate_{k}_{i}")

        # Energy propagation constraints (linking nodes i and j)
        for i, j in [(i, j) for _k, i, j in arcs.keys() if _k == k]:
            
            theta_ij = energy_consumption[(k, i, j)]
            
            # Mathematical expression (46)
            model.addConstr(E_pre[k, j] >= E_bar[k, i] - theta_ij - BIG_M * (1 - x[k, i, j]), name=f"EB_PropLB_{k}_{i}_{j}")

            # Mathematical expression (47)
            model.addConstr(E_pre[k, j] <= E_bar[k, i] - theta_ij + BIG_M * (1 - x[k, i, j]), name=f"EB_PropUB_{k}_{i}_{j}")

    for k in vehicles:
        o_node, d_node = origin_nodes[k], dest_nodes[k]
        
        nodes_for_k = [i for k_i, i in E_pre.keys() if k_i == k]

        for i in nodes_for_k:
            if i.startswith('O'):
                continue

            # Mathematical expression (48)
            if i in trip_ids:
                trip_eta = trips_map[i].eta
                model.addConstr(G[k, i] == trip_eta, name=f"EB_TripEta_{k}_{i}")
            
            # Mathematical expression (49)
            if i in charging_station_set:
                model.addConstr(G[k, i] == E_pre[k, i] - PHI_MAX, name=f"EB_DepotRefillLogic_{k}_{i}")

            # Mathematical expression (50)
            model.addConstr(E_pre[k, i] >= PHI_MIN, name=f"EB_MinLevelPre_{k}_{i}")
            # E_bar min level enforcement
            # if i != d_node:
            #     model.addConstr(E_bar[k, i] >= PHI_MIN, name=f"EB_MinLevelBar_{k}_{i}")

    # ==================================================
    # Continuous time constraints for charging stations
    # ==================================================

    for k in vehicles:
        for j in charging_station_set:
            for i in trip_ids:
                charging_duration = (E_bar[k, j] - E_pre[k, j]) / CHARGING_RATE_KWH_PER_MINUTE
                model.addConstr(CT[k, j] <= T[k, j] + charging_duration + BIG_M * (1 - x[k, i, j]), name=f"ChargeCompTime_UB1_{k}_{i}_{j}") # Constraint (51)
                model.addConstr(CT[k, j] >= T[k, j] + charging_duration - BIG_M * (1 - x[k, i, j]), name=f"ChargeCompTime_LB1_{k}_{i}_{j}") # Constraint (52)
            # model.addConstr(CT[k, j] <= BIG_M * gp.quicksum(x[k, i, j] for i in trip_ids), name=f"ChargeCompTime_zero_{k}_{j}") # Constraint (53)

    for j in charging_station_ids:
        for k1 in vehicles:
            for k2 in vehicles:
                if k1 != k2:
                    model.addConstr(T[k1, j] <= T[k2, j] + BIG_M * Y[j, k1, k2], name=f"ChargeOrder_Arr_Time_1_{j}_{k1}_{k2}") # Constraint (54)
                    model.addConstr(T[k1, j] >= CT[k2, j] + SMALL_M - BIG_M * (1 - Y[j, k1, k2]), name=f"ChargeOrder_Comp_Time_2_{j}_{k1}_{k2}") # Constraint (55)
                    # model.addConstr(Y[j, k1, k2] <= gp.quicksum(x[k1, i, j] for i in trip_ids), name=f"protection_1_{j}_{k1}_{k2}") # Constraint (56)
                    # model.addConstr(Y[j, k1, k2] <= gp.quicksum(x[k2, i, j] for i in trip_ids) + SMALL_M, name=f"protection_2_{j}_{k1}_{k2}") # Constraint (57)

    model.setParam('MIPGap', 0.01)
            
    model.optimize()
    
    # --- Reporting Logic (Updated to include Energy Variables) ---
    report_lines, schedules = [], {}
    variable_report_str = ""
    solution_data = {}
    default_time_var = type('obj', (object,), {'X': 0.0}) # Helper for safe .X access
    
    if model.status == GRB.OPTIMAL:

        # New dictionary to collect variable results for JSON export
        solution_vars_json = {}
        TOL = 1e-4 # Tolerance for non-negative values
        
        # --- Helper function to extract and save variable values ---
        def extract_vars(gurobi_vars, var_name):
            data = {}
            for key, var in gurobi_vars.items():
                if var.X > TOL:
                    # Convert the tuple key (k, i, j) to a readable string key
                    str_key = str(key).replace("'", "").replace(" ", "")
                    data[str_key] = round(var.X, 4)
            solution_vars_json[var_name] = data

        # Extract all main variable types
        extract_vars(x, "x")
        extract_vars(T, "T")
        extract_vars(w, "w")
        extract_vars(E_pre, "E_pre")
        extract_vars(E_bar, "E_bar")
        extract_vars(G, "G")
        
        # Assuming CT and Y are now defined for charging
        try:
            extract_vars(CT, "CT")
            extract_vars(Y, "Y")
        except NameError:
            # Handle case where CT or Y might not be defined/used
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
                        
                        # --- NEW: Get energy levels at the next node ---
                        epre_val = E_pre.get((k, j_node), default_time_var).X
                        gbar_val = E_bar.get((k, j_node), default_time_var).X
                        g_val = G.get((k, j_node), default_time_var).X
                        
                        energy_info = f" (E_pre: {epre_val:.1f}, G: {g_val:.1f}, E_bar: {gbar_val:.1f})"
                        
                        route_str.append(f"{j_node} (Time: {time_val:.2f}{energy_info})")
                        current_node = j_node; found_next = True; break
                if not found_next: break
            schedules[k] = route
            report_lines.append(f" - Vehicle {k} schedule: {' -> '.join(route_str)}")
            
        var_report_lines = []
        
        # 1. Report 'x' variables (Active Arcs)
        var_report_lines.append("\n--- Active Arc Variables (x[k,i,j] = 1) ---")
        sorted_x_keys = sorted([key for key in x.keys() if x[key].X > 0.5]) 
        for (k, i, j) in sorted_x_keys:
            var_report_lines.append(f"  x[{k}, {i}, {j}] = {x[k,i,j].X:.0f} (Cost: {base_costs[k,i,j]}, Cons: {energy_consumption.get((k,i,j), 0):.1f})")

        # 2. Report 'T' variables (Node Times)
        var_report_lines.append("\n--- Node Start Times (T[k,i] > 0) ---")
        solution_nodes = set((k, i) for (k, i, j) in sorted_x_keys) | set((k, j) for (k, i, j) in sorted_x_keys)
        sorted_T_keys = sorted([key for key in T.keys() if key in solution_nodes and T.get(key) is not None and T[key].X > 0.0001])
        for (k, i) in sorted_T_keys:
            var_report_lines.append(f"  T[{k}, {i}] = {T[k,i].X:.2f}")

        # 3. Report 'w' variables (Waiting Costs)
        var_report_lines.append("\n--- Incurred Waiting Costs (w[k,i,j] > 0) ---")
        sorted_w_keys = sorted([key for key in w.keys() if w[key].X > 0.0001])
        for (k, i, j) in sorted_w_keys:
            var_report_lines.append(f"  w[{k}, {i}, {j}] = {w[k,i,j].X:.2f}")

        # 4. Report Energy Variables (E_pre, G, E_bar)
        var_report_lines.append("\n--- Energy Levels (E_pre, G, E_bar) ---")
        
        sorted_E_keys = sorted([key for key in E_pre.keys() if key in solution_nodes])
        for (k, i) in sorted_E_keys:
            e_pre_val = E_pre.get((k, i), default_time_var).X
            g_val = G.get((k, i), default_time_var).X
            e_bar_val = E_bar.get((k, i), default_time_var).X
            # Only report if E_bar is relevant (part of the path)
            if e_bar_val > 0.0001: 
                 var_report_lines.append(f"  {k}, {i}: E_pre={e_pre_val:.1f} | G={g_val:.1f} | E_bar={e_bar_val:.1f} (Min/Max: {PHI_MIN}/{PHI_MAX})")

        # 5. Report Charging Completion Time (CT)
        var_report_lines.append("\n--- Charging Completion Times (CT[k, j] > 0) ---")
        # Filter for charging station nodes that are part of the solution
        solution_cs_nodes = set((k, i) for (k, i) in sorted_E_keys if i in charging_station_set)

        sorted_CT_keys = sorted([key for key in CT.keys() if key in solution_cs_nodes and CT.get(key) is not None and CT[key].X > 0.0001])
        for (k, j) in sorted_CT_keys:
            var_report_lines.append(f"  CT[{k}, {j}] = {CT[k, j].X:.2f}")

        # 6. Report Charging Order Variables (Y)
        var_report_lines.append("\n--- Charging Order Variables (Y[j, k1, k2] = 1) ---")
        sorted_Y_keys = sorted([key for key in Y.keys() if Y[key].X > 0.5])
        for (j, k1, k2) in sorted_Y_keys:
            # Y[j, k1, k2] = 1 means k1 arrives before k2 at station j
            var_report_lines.append(f"  Y[{j}, {k1}, {k2}] = {Y[j, k1, k2].X:.0f}")
        
        variable_report_str = "\n".join(var_report_lines)

        if solution_data:
            json_output_path = os.path.join(output_dir, "solution_variables.json")
            with open(json_output_path, "w") as f:
                json.dump(solution_data["solution_vars"], f, indent=4)
            print(f"Solution variables saved to '{json_output_path}'")
            
    else:
        report_lines.append("No optimal solution was found or the model was infeasible.")
    
    return "\n".join(report_lines), schedules, variable_report_str

# --- Reporting Function (Non-Plotting) ---

def get_instance_report(instance: ProblemInstance) -> str:
    """
    Generates a formatted string report for the instance.
    (Unchanged)
    """
    report_lines = []
    report_lines.append("--- Instance Generation Summary ---")
    total_vehicles = sum(depot.vehicle_count for depot in instance.depots)
    report_lines.append(f"  - Trips to cover: {len(instance.trips)}")
    report_lines.append(f"  - Depots: {len(instance.depots)}")
    report_lines.append(f"  - Total vehicles available: {total_vehicles}")
    report_lines.append("\n--- Depot and Vehicle Information ---")
    for depot in instance.depots:
        report_lines.append(f"Depot ID: {depot.id} | Location: ({depot.location.x:5.2f}, {depot.location.y:5.2f}) | Vehicles: {depot.vehicle_count}")
    report_lines.append("\n--- Detailed Trip Information ---")
    point_to_id_map = {point: i + 1 for i, point in enumerate(instance.relief_points)}
    for trip in instance.trips:
        start_rp_id = point_to_id_map.get(trip.start_point, "N/A")
        end_rp_id = point_to_id_map.get(trip.end_point, "N/A")
        # Add eta reporting
        eta_info = f" | ETA: {trip.eta:.2f}" if hasattr(trip, 'eta') and trip.eta is not None else ""
        report_lines.append(
            f"Trip ID: {trip.id: <3} | "
            f"Type: {trip.trip_type: <5} | "
            f"Starts at RP #{start_rp_id: <3} -> Ends at RP #{end_rp_id: <3} | "
            f"Time: {trip.start_time:7.2f} to {trip.end_time:7.2f}{eta_info}"
        )
    return "\n".join(report_lines)

# --- Main Execution Block (ADAPTED) ---

if __name__ == "__main__":
    """
    This block runs only when the script is executed directly.
    """

    random.seed(43)
    
    # 1. Generate instance
    print("--- Part 1: Generating a Gkiotsalitis (EB-VSP-TW) instance ---")
    # ***ADAPTATION 3: Use the GkiotsalitisInstanceGenerator and provide n_charging_stations***
    generator = GkiotsalitisInstanceGenerator(n_trips=12, n_depots=1, n_relief_points=4, n_charging_stations=2, problem_class='A')
    instance = generator.generate()
    print("Instance generation complete.")

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
        plotting_utils_eb.save_solution_dag_plot(instance, schedules, os.path.join(output_dir, "solved_instance_dag.png"), trip_y=trip_y)
        plotting_utils_eb.save_solution_plot(instance, schedules, bus_blocks_dir)
    
    # 6. Write combined report to file
    print("--- Part 5: Writing Full Report to File ---")
    full_report_path = os.path.join(output_dir, "instance_and_solution_report.txt")
    with open(full_report_path, "w", encoding='utf-8') as f:
        f.write("--- INSTANCE INFORMATION ---\n")
        f.write(instance_report)
        f.write("\n\n--- SOLUTION ---\n")
        f.write(solution_report)

        # --- CHANGE 2: Append the variable report to the file ---
        if variable_report_str: # Only write if the string is not empty
            f.write("\n\n--- SOLUTION VARIABLES ---")
            f.write(variable_report_str)
        
    print(f"\nProcess complete. All results saved in '{output_dir}'")