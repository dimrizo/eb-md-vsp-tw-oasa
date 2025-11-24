import os
import sys
import datetime
import math
import random
# Matplotlib and NetworkX imports are no longer needed here
import gurobipy as gp
from gurobipy import GRB

# --- Add parent directory to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities.instance_generator import DesaulniersInstanceGenerator, ProblemInstance
# --- NEW: Import your plotting module ---
import plotting.plotting_utils as plotting_utils

from utilities import haversine

BIG_M = 10000
AVG_U_METERS_PER_MINUTE = 800000 # Average speed: 26,000 meters / 60 minutes = 433.33 meters/minute

# --- Main Solver Function ---

def solve_md_vsp_tw_from_instance(instance: ProblemInstance, waiting_cost_lambda: float = 2.0):
    """
    Solves the MD-VSP-TW and returns both a text report and the solution schedules.
    (This function is MODIFIED to fix a bug)
    """
    model = gp.Model("MD-VSP-TW")
    
    model.setParam('OutputFlag', 1)

    # (Instance processing logic remains unchanged)
    vehicles, vehicle_to_depot = [], {}
    for depot in instance.depots:
        for i in range(1, depot.vehicle_count + 1):
            vehicle_id = f"D{depot.id}_V{i}"; vehicles.append(vehicle_id)
            vehicle_to_depot[vehicle_id] = depot
    
    trips_map = {f"T{trip.id}": trip for trip in instance.trips}
    trip_ids = list(trips_map.keys())
    origin_nodes = {v_id: f"O{vehicle_to_depot[v_id].id}" for v_id in vehicles}
    dest_nodes = {v_id: f"D{vehicle_to_depot[v_id].id}" for v_id in vehicles}
    
    arcs, base_costs, elapsed_times = {}, {}, {}
    for k in vehicles:
        depot, o_node, d_node = vehicle_to_depot[k], origin_nodes[k], dest_nodes[k]
        for trip_id, trip in trips_map.items():
            deadhead_distance = haversine.main(depot.location.y, depot.location.x, trip.start_point.y, trip.start_point.x)
            deadhead_time = deadhead_distance / AVG_U_METERS_PER_MINUTE
            if 0 + deadhead_time <= trip.start_time_window[1]:
                arc = (k, o_node, trip_id); arcs[arc] = 1
                base_costs[arc] = math.floor(10 * deadhead_time) + 5000
                elapsed_times[arc] = deadhead_time
        for trip1_id, trip1 in trips_map.items():
            for trip2_id, trip2 in trips_map.items():
                if trip1_id == trip2_id: continue
                deadhead_distance = haversine.main(depot.location.y, depot.location.x, trip.start_point.y, trip.start_point.x)
                deadhead_time = deadhead_distance / AVG_U_METERS_PER_MINUTE
                trip1_duration = trip1.end_time - trip1.start_time
                t_ij = trip1_duration + deadhead_time
                if trip1.start_time_window[0] + t_ij <= trip2.start_time_window[1]:
                    arc = (k, trip1_id, trip2_id); arcs[arc] = 1
                    base_costs[arc] = math.floor(10 * deadhead_time)
                    elapsed_times[arc] = t_ij
        for trip_id, trip in trips_map.items():
            deadhead_distance = haversine.main(depot.location.y, depot.location.x, trip.start_point.y, trip.start_point.x)
            deadhead_time = deadhead_distance / AVG_U_METERS_PER_MINUTE
            trip_duration = trip.end_time - trip.start_time
            arc = (k, trip_id, d_node); arcs[arc] = 1
            base_costs[arc] = math.floor(10 * deadhead_time)
            elapsed_times[arc] = trip_duration + deadhead_time

    # (Gurobi variable and objective definitions remain unchanged)
    x = model.addVars(arcs.keys(), vtype=GRB.BINARY, name="x")
    time_vars_keys = set((k, i) for k, i, j in arcs.keys()) | set((k, j) for k, i, j in arcs.keys())
    T = model.addVars(time_vars_keys, vtype=GRB.CONTINUOUS, name="T")
    w = model.addVars(arcs.keys(), vtype=GRB.CONTINUOUS, lb=0.0, name="w")

    # Mathematical expression (18)
    objective = gp.quicksum(base_costs[k, i, j] * x[k, i, j] for k, i, j in arcs.keys()) + \
                gp.quicksum(w[k, i, j] for k, i, j in arcs.keys() if i in trip_ids and j in trip_ids)
    model.setObjective(objective, GRB.MINIMIZE)

    # Mathematical expression (19)
    model.addConstrs((x.sum('*', '*', j) == 1 for j in trip_ids), name="CoverTrip")

    # Mathematical expression (20)
    model.addConstrs((x.sum(k, '*', v) - x.sum(k, v, '*') == 0 for k in vehicles for v in trip_ids), name="FlowConservation")

    # Mathematical expression (21)
    model.addConstrs((x.sum(k, origin_nodes[k], '*') == x.sum(k, '*', dest_nodes[k]) for k in vehicles), name="ReturnToDepot")

    # Mathematical expression (22)
    model.addConstrs((x.sum(k, origin_nodes[k], '*') <= 1 for k in vehicles), name="StartOnce")

    # Mathematical expression (23)
    depot_to_vehicles = {}
    for depot in instance.depots:
        depot_id_str = f"O{depot.id}" # Use the origin node string ID consistent with 'origin_nodes'
        depot_to_vehicles[depot_id_str] = [v_id for v_id in vehicles if origin_nodes[v_id] == depot_id_str]

    depots_map = {depot.id: depot for depot in instance.depots}
    for origin_node_id, vehicles_in_depot in depot_to_vehicles.items():
        depot_id_num = int(origin_node_id[1:])
        depot_obj = depots_map[depot_id_num]
        
        model.addConstr(
            gp.quicksum(x[k, origin_node_id, j] for k in vehicles_in_depot for j in trip_ids if (k, origin_node_id, j) in x) <= depot_obj.vehicle_count, name=f"DepotCapacity_{origin_node_id}")
    
    for k, i in T.keys():
        if i in trips_map:
            trip = trips_map[i]
            T[k, i].lb, T[k, i].ub = trip.start_time_window[0], trip.start_time_window[1]           # Mathematical expression (24)
        elif i.startswith("O"):
            T[k, i].lb, T[k, i].ub = 0, BIG_M                                                       # Mathematical expression (25)
    
    for k, i, j in arcs.keys():
        t_ij = elapsed_times[k, i, j]
        model.addConstr(T[k, j] >= T[k, i] + t_ij - BIG_M * (1 - x[k, i, j]), name=f"TimeProp_{k}_{i}_{j}") # Mathematical expression (26)
        if i in trip_ids and j in trip_ids:
            model.addConstr(w[k, i, j] >= waiting_cost_lambda * (T[k, j] - T[k, i] - t_ij) - BIG_M * (1 - x[k, i, j]))  # Mathematical expression (27)
            model.addConstr(w[k, i, j] <= waiting_cost_lambda * (T[k, j] - T[k, i] - t_ij) + BIG_M * (1 - x[k, i, j]))  # Mathematical expression (28)
            model.addConstr(w[k, i, j] <= BIG_M * x[k, i, j])                                                           # Mathematical expression (29)

    model.setParam('MIPGap', 0.01)

    model.optimize()
    report_lines, schedules = [], {}

    variable_report_str = ""
    
    if model.status == GRB.OPTIMAL:
        report_lines.append(f"Optimal solution found with total cost: {model.ObjVal:.2f}")
        used_vehicles = {k for k, i, j in x.keys() if x[k, i, j].X > 0.5}
        report_lines.append(f"Total vehicles used: {len(used_vehicles)} out of {len(vehicles)}")
        for k in sorted(list(used_vehicles)):
            current_node, route = origin_nodes[k], [origin_nodes[k]]
            
            # Use T.get() for safer access, provide a default object with .X attribute if key is missing
            default_time_var = type('obj', (object,), {'X': 0.0}) 
            start_time = T.get((k, origin_nodes[k]), default_time_var).X
            route_str = [f"{origin_nodes[k]} (Time: {start_time:.2f})"]
            
            while current_node not in dest_nodes.values():
                found_next = False
                # Filter arcs for the current vehicle k
                vehicle_arcs = [(i, j) for _k, i, j in arcs.keys() if _k == k]
                for i_node, j_node in vehicle_arcs:
                    if i_node == current_node and x.get((k, i_node, j_node)) and x[k, i_node, j_node].X > 0.5:
                        route.append(j_node)
                        # Use .get() for safe access
                        time_val = T.get((k, j_node), default_time_var).X
                        route_str.append(f"{j_node} (Time: {time_val:.2f})")
                        current_node = j_node; found_next = True; break
                if not found_next: break
            schedules[k] = route
            report_lines.append(f"  - Vehicle {k} schedule: {' -> '.join(route_str)}")
            
        var_report_lines = []
        
        # 1. Report 'x' variables (Active Arcs)
        var_report_lines.append("\n--- Active Arc Variables (x[k,i,j] = 1) ---")
        sorted_x_keys = sorted([key for key in x.keys() if x[key].X > 0.5]) 
        for (k, i, j) in sorted_x_keys:
            var_report_lines.append(f"  x[{k}, {i}, {j}] = {x[k,i,j].X:.0f}")

        # 2. Report 'T' variables (Node Times)
        var_report_lines.append("\n--- Node Start Times (T[k,i] > 0) ---")
        # Report only times for nodes that are part of the solution
        solution_nodes = set((k, i) for (k, i, j) in sorted_x_keys) | set((k, j) for (k, i, j) in sorted_x_keys)
        sorted_T_keys = sorted([key for key in T.keys() if key in solution_nodes and T.get(key) is not None and T[key].X > 0.0001])
        for (k, i) in sorted_T_keys:
            var_report_lines.append(f"  T[{k}, {i}] = {T[k,i].X:.2f}")

        # 3. Report 'w' variables (Waiting Costs)
        var_report_lines.append("\n--- Incurred Waiting Costs (w[k,i,j] > 0) ---")
        sorted_w_keys = sorted([key for key in w.keys() if w[key].X > 0.0001])
        for (k, i, j) in sorted_w_keys:
            var_report_lines.append(f"  w[{k}, {i}, {j}] = {w[k,i,j].X:.2f}")
        
        variable_report_str = "\n".join(var_report_lines)
            
    else:
        report_lines.append("No optimal solution was found or the model was infeasible.")
    
    # --- CHANGE 3: Return the new string ---
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
    report_lines.append(f"  - Trips to cover: {len(instance.trips)}")
    report_lines.append(f"  - Depots: {len(instance.depots)}")
    report_lines.append(f"  - Total vehicles available: {total_vehicles}")
    report_lines.append("\n--- Depot and Vehicle Information ---")
    for depot in instance.depots:
        report_lines.append(f"Depot ID: {depot.id} | Location: ({depot.location.x:5.2f}, {depot.location.y:5.2f}) | Vehicles: {depot.vehicle_count}")
    report_lines.append("\n--- Detailed Trip Information ---")
    point_to_id_map = {point: i + 1 for i, point in enumerate(instance.relief_points)}
    for trip in instance.trips:
        start_rp_id = point_to_id_map.get(trip.start_point, "N/A")
        end_rp_id = point_to_id_map.get(trip.end_point, "N/A")
        report_lines.append(
            f"Trip ID: {trip.id: <3} | "
            f"Type: {trip.trip_type: <5} | "
            f"Starts at RP #{start_rp_id: <3} -> Ends at RP #{end_rp_id: <3} | "
            f"Time: {trip.start_time:7.2f} to {trip.end_time:7.2f}"
        )
    return "\n".join(report_lines)

# --- ALL PLOTTING FUNCTIONS HAVE BEEN REMOVED ---
# (save_instance_plot, save_dag_plot, save_solution_dag_plot, save_solution_plot)
# They are now all inside plotting_utils.py

# --- Main Execution Block (UPDATED with new print statements) ---

if __name__ == "__main__":
    """
    This block runs only when the script is executed directly.
    """

    random.seed(38)
    
    # 1. Generate instance
    print("--- Part 1: Generating a Desaulniers et al. (TW) instance ---")
    generator = DesaulniersInstanceGenerator(n_trips=20, n_depots=3, n_relief_points=10, problem_class='A')
    instance = generator.generate()
    print("Instance generation complete.")

    # 2. Set up output directory
    print("--- Part 2: Preparing Output Directory ---")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, "..", "output", f"TW_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    bus_blocks_dir = os.path.join(output_dir, "bus_blocks")
    os.makedirs(bus_blocks_dir, exist_ok=True)
    
    # 3. Save instance report and plots
    print(f"--- Part 3: Saving instance details and plots to '{output_dir}' ---")
    instance_report = get_instance_report(instance)
    
    plotting_utils.save_instance_plot(instance, os.path.join(output_dir, "instance_plot.png"))
    trip_y = plotting_utils.save_dag_plot(instance, os.path.join(output_dir, "instance_dag.png"))
    
    # 4. Solve instance
    print("--- Part 4: Solving the Instance ---")
    solution_report, schedules, variable_report_str = solve_md_vsp_tw_from_instance(instance)
    
    # 5. Save solution plots if a solution was found
    if schedules:
        plotting_utils.save_solution_dag_plot(instance, schedules, os.path.join(output_dir, "solved_instance_dag.png"), trip_y=trip_y)
        plotting_utils.save_solution_plot(instance, schedules, bus_blocks_dir)
    
    # 6. Write combined report to file
    print("--- Part 5: Writing Full Report to File ---")
    full_report_path = os.path.join(output_dir, "instance_and_solution_report.txt")
    with open(full_report_path, "w") as f:
        f.write("--- INSTANCE INFORMATION ---\n")
        f.write(instance_report)
        f.write("\n\n--- SOLUTION ---\n")
        f.write(solution_report)

        # --- CHANGE 2: Append the variable report to the file ---
        if variable_report_str: # Only write if the string is not empty
            f.write("\n\n--- SOLUTION VARIABLES ---")
            f.write(variable_report_str)
        
    print(f"\nProcess complete. All results saved in '{output_dir}'")