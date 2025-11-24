import os
import datetime
import math
import sys
import random

# Imports for the solver
import gurobipy as gp
from gurobipy import GRB

# --- Core Project Imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utilities.instance_generator import CarpanetoInstanceGenerator, ProblemInstance

# --- NEW: Import your plotting module ---
import plotting.plotting_utils as plotting_utils

from utilities import haversine

BIG_M = 10000
AVG_U_METERS_PER_MINUTE = 800000 # Average speed: 26,000 meters / 60 minutes = 433.33 meters/minute

def get_instance_report(instance: ProblemInstance) -> str:
    """
    Generates a formatted string report with details about the instance.
    (This function is unchanged as it does not plot)
    """
    report_lines = []
    
    # Summary
    report_lines.append("--- Instance Generation Summary ---")
    total_vehicles = sum(depot.vehicle_count for depot in instance.depots)
    report_lines.append(f"  - Trips to cover: {len(instance.trips)}")
    report_lines.append(f"  - Depots: {len(instance.depots)}")
    report_lines.append(f"  - Total vehicles available: {total_vehicles}")
    
    # Depot Details
    report_lines.append("\n--- Depot and Vehicle Information ---")
    for depot in instance.depots:
        report_lines.append(
            f"Depot ID: {depot.id} | "
            f"Location: ({depot.location.x:5.2f}, {depot.location.y:5.2f}) | "
            f"Vehicles Available: {depot.vehicle_count}"
        )
        
    # Trip Details
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

def solve_md_vsp_from_instance(instance: ProblemInstance):
    """
    Solves the MD-VSP and returns both a text report and the solution schedules.
    (This function is unchanged)
    """
    model = gp.Model("MD-VSP_from_Instance")
    model.setParam('OutputFlag', 0)
    
    vehicles, vehicle_to_depot = [], {}
    for depot in instance.depots:
        for i in range(1, depot.vehicle_count + 1):
            vehicle_id = f"D{depot.id}_V{i}"; vehicles.append(vehicle_id)
            vehicle_to_depot[vehicle_id] = depot
    
    trip_ids = [f"T{trip.id}" for trip in instance.trips]
    trips_map = {f"T{trip.id}": trip for trip in instance.trips}
    origin_nodes = {v_id: f"O{vehicle_to_depot[v_id].id}" for v_id in vehicles}
    dest_nodes = {v_id: f"D{vehicle_to_depot[v_id].id}" for v_id in vehicles}
    arcs, costs = {k: [] for k in vehicles}, {}

    for k in vehicles:
        depot, o_node, d_node = vehicle_to_depot[k], origin_nodes[k], dest_nodes[k]
        for trip_id, trip in trips_map.items():
            travel_distance = haversine.main(depot.location.y, depot.location.x, trip.start_point.y, trip.start_point.x)
            travel_time = travel_distance / AVG_U_METERS_PER_MINUTE
            if depot.location != trip.start_point:
                arcs[k].append((o_node, trip_id)); costs[k, o_node, trip_id] = math.floor(10 * travel_time) + 5000
        for trip1_id, trip1 in trips_map.items():
            for trip2_id, trip2 in trips_map.items():
                if trip1_id == trip2_id: continue
                travel_distance = haversine.main(depot.location.y, depot.location.x, trip1.end_point.y, trip1.end_point.x)
                travel_time = travel_distance / AVG_U_METERS_PER_MINUTE
                if trip1.end_time + travel_time <= trip2.start_time:
                    idle_time = trip2.start_time - (trip1.end_time + travel_time)
                    arcs[k].append((trip1_id, trip2_id)); costs[k, trip1_id, trip2_id] = math.floor(10 * travel_time + 2 * idle_time)
        for trip_id, trip in trips_map.items():
            travel_distance = haversine.main(depot.location.y, depot.location.x, trip1.end_point.y, trip1.end_point.x)
            travel_time = travel_distance / AVG_U_METERS_PER_MINUTE
            arcs[k].append((trip_id, d_node)); costs[k, trip_id, d_node] = math.floor(10 * travel_time)

    x = model.addVars(costs.keys(), vtype=GRB.BINARY, name="x")

    # Mathematical expression (1)
    model.setObjective(gp.quicksum(costs[k, i, j] * x[k, i, j] for (k, i, j) in costs), GRB.MINIMIZE)

    # Mathematical expression (2)
    model.addConstrs((gp.quicksum(x[k, i, j] for k in vehicles for (i, j2) in arcs[k] if j2 == j) == 1 for j in trip_ids), name="CoverTrip")
    
    # Mathematical expression (3)
    model.addConstrs((gp.quicksum(x[k, i, j] for (i, j) in arcs[k] if j == v) - gp.quicksum(x[k, u, l] for (u, l) in arcs[k] if u == v) == 0 for k in vehicles for v in trip_ids), name="FlowConservation")

    # Mathematical expression (4)
    model.addConstrs((
        gp.quicksum(x[k, origin_nodes[k], j] for j in trip_ids) == gp.quicksum(x[k, i, dest_nodes[k]] for i in trip_ids) for k in vehicles
    ), name="ReturnToDepot")

    # Mathematical expression (5)
    model.addConstrs((gp.quicksum(x[k, origin_nodes[k], j] for j in trip_ids) <= 1 for k in vehicles), name="StartOnce")

    # Mathematical expression (6)
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

    model.optimize()
    
    report_lines, schedules = [], {}
    # --- CHANGE 1: Initialize the new string ---
    all_vars_report_str = "" 
    
    if model.status == GRB.OPTIMAL:
        report_lines.append(f"Optimal solution found with total cost: {model.ObjVal:.2f}")
        used_vehicles = {k for k, i, j in x.keys() if x[k, i, j].X > 0.5}
        report_lines.append(f"Total vehicles used: {len(used_vehicles)} out of {len(vehicles)}")
        for k in sorted(list(used_vehicles)):
            current_node, route = origin_nodes[k], [origin_nodes[k]]
            while current_node not in dest_nodes.values():
                found_next = False
                for i_node, j_node in arcs[k]:
                    if i_node == current_node and x.get((k, i_node, j_node)) and x[k, i_node, j_node].X > 0.5:
                        route.append(j_node); current_node = j_node; found_next = True; break
                if not found_next: break
            schedules[k] = route
            vehicle_cost = sum(costs.get((k, route[i], route[i+1]), 0) for i in range(len(route) - 1))
            report_lines.append(f"  - Vehicle {k} schedule: {' -> '.join(route)} | Total Cost: {vehicle_cost:.2f}")
            
        # --- CHANGE 2: Populate the new string ---
        var_report_lines = []
        var_report_lines.append("\n--- All Active Arc Variables (x[k,i,j] = 1) ---")
        
        # Sort the keys for a cleaner report
        sorted_x_keys = sorted([key for key in x.keys() if x[key].X > 0.5]) 
        
        for (k, i, j) in sorted_x_keys:
            # We use .X to get the value, which will be 1.0
            var_report_lines.append(f"  x[{k}, {i}, {j}] = {x[k,i,j].X:.0f}  (Cost: {costs.get((k,i,j), 0):.2f})")
        
        all_vars_report_str = "\n".join(var_report_lines)

    else:
        report_lines.append("No optimal solution was found.")
    
    # --- CHANGE 3: Return the new string ---
    return "\n".join(report_lines), schedules, all_vars_report_str


# --- Main Execution Script ---

def generate_solve_and_save():
    """
    A single function to generate, solve, and save all results for a Carpaneto instance.
    """
    print("--- Part 1: Generating a Carpaneto et al. instance ---")
    
    generator = CarpanetoInstanceGenerator(n_trips=20, n_depots=3, n_relief_points=5, problem_class='A')
    instance = generator.generate()
    print("Instance generation complete.")

    print("--- Part 2: Preparing Output Directory ---")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, "..", "output", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    bus_blocks_dir = os.path.join(output_dir, "bus_blocks")
    os.makedirs(bus_blocks_dir, exist_ok=True)
    
    plot_filepath = os.path.join(output_dir, "instance_plot.png")
    dag_filepath = os.path.join(output_dir, "instance_dag.png")
    solution_dag_filepath = os.path.join(output_dir, "solved_instance_dag.png")
    report_filepath = os.path.join(output_dir, "instance_and_solution_report.txt")

    print(f"--- Part 3: Saving instance details and plots to '{output_dir}' ---")
    instance_report_str = get_instance_report(instance)
    
    # --- UPDATED PLOTTING CALLS ---
    plotting_utils.save_instance_plot(instance, plot_filepath)
    trip_y = plotting_utils.save_dag_plot(instance, dag_filepath)
    # --- END UPDATED PLOTTING CALLS ---
    
    print("--- Part 4: Solving the Instance ---")
    solution_report_str, schedules, all_vars_report_str = solve_md_vsp_from_instance(instance)
    
    if schedules:
        # --- UPDATED PLOTTING CALLS ---
        plotting_utils.save_solution_dag_plot(instance, schedules, solution_dag_filepath, trip_y=trip_y) # <- reuse y
        plotting_utils.save_solution_plot(instance, schedules, bus_blocks_dir)
        # --- END UPDATED PLOTTING CALLS ---
    
    print("--- Part 5: Writing Full Report to File ---")
    with open(report_filepath, "w") as f:
        f.write("========================================\n"); f.write("      INSTANCE INFORMATION\n")
        f.write("========================================\n"); f.write(instance_report_str)
        f.write("\n\n========================================\n"); f.write("      SOLUTION REPORT\n")
        f.write("========================================\n")
        if solution_report_str: f.write(solution_report_str)
        else: f.write("No solution was found.")

        # --- CHANGE 2: Append the variable report to the file ---
        if all_vars_report_str: # Only write if the string is not empty
            f.write("\n\n========================================\n"); f.write("      SOLUTION VARIABLES\n")
            f.write("========================================\n")
            f.write(all_vars_report_str)
            
    print(f"\nProcess complete. All results saved in '{output_dir}'")

if __name__ == "__main__":
    random.seed(39)
    generate_solve_and_save()