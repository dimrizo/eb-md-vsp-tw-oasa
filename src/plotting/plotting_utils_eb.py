"""
Reusable Plotting Utilities for Vehicle Scheduling Problems

This module provides functions to visualize ProblemInstance objects and
their corresponding solution schedules.

It is designed to be "smart" and can handle two types of ProblemInstance:
1. Carpaneto-style: Trips have 'start_time' and 'end_time' attributes.
2. Desaulniers-style: Trips have 'start_time_window' (a tuple) and
   also 'start_time' and 'end_time' for service duration.

ADAPTED: Now supports Charging Station nodes ('C#') for EB-VSP-TW models with 
a fixed layout to ensure visual separation.
"""

import os
import sys
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional

# This path logic allows this module to find the 'utilities' folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the data structure this module needs to plot
try:
    from utilities.instance_generator import ProblemInstance, Point
except ImportError:
    print("Error: Could not import ProblemInstance. Make sure 'utilities/instance_generator.py' is accessible.")
    # Define dummy classes to allow type hints to work if import fails
    class Point: pass
    class ProblemInstance: pass

# --- Plotting Helper Functions ---

def _compute_trip_y(instance: ProblemInstance) -> Dict[str, float]:
    """
    Internal helper to compute deterministic Y-axis positions for trip nodes.
    This ensures both DAG plots use the same layout for trips.
    """
    # Deterministic order (by trip id)
    trip_nodes = [f"T{t.id}" for t in sorted(instance.trips, key=lambda t: t.id)]
    n = len(trip_nodes)
    if n == 0:
        return {}
    # Even spacing, centered at 0
    y_vals = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
    # Optional scaling: more vertical spacing if many depots
    scale = max(1, len(instance.depots))
    y_vals = y_vals * scale
    return {trip_nodes[i]: float(y_vals[i]) for i in range(n)}

# --- Public Plotting Functions ---

def save_instance_plot(instance: ProblemInstance, save_path: str):
    """
    Generates and saves a plot of the instance grid to a specified file path.
    Shows depots, relief points, and Charging Stations.
    """
    relief_x = [p.x for p in instance.relief_points]
    relief_y = [p.y for p in instance.relief_points]
    depot_x = [d.location.x for d in instance.depots]
    depot_y = [d.location.y for d in instance.depots]
    
    # --- NEW: Charging Station Points ---
    cs_x = [cs.location.x for cs in instance.charging_stations]
    cs_y = [cs.location.y for cs in instance.charging_stations]
    # ---

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(relief_x, relief_y, c='blue', marker='o', label='Relief Points', alpha=0.6)
    ax.scatter(depot_x, depot_y, c='red', marker='s', s=120, label='Depots', edgecolors='black')
    
    # --- NEW: Plot Charging Stations ---
    ax.scatter(cs_x, cs_y, c='green', marker='^', s=100, label='Charging Stations', edgecolors='black', zorder=3)
    # ---

    for i, point in enumerate(instance.relief_points):
        ax.text(point.x + 0.5, point.y + 0.5, str(i + 1), fontsize=9, color='navy')
    for depot in instance.depots:
        ax.text(depot.location.x + 0.5, depot.location.y + 0.5, f"Depot {depot.id}",
                fontsize=10, color='darkred', weight='bold')
    # --- NEW: Label Charging Stations ---
    for i, cs in enumerate(instance.charging_stations):
        ax.text(cs.location.x + 0.5, cs.location.y - 1.0, f"CS {i + 1}", 
                fontsize=10, color='darkgreen', weight='bold')
    # ---

    ax.set_title(f'Generated Instance on a {instance.grid_size[0]}x{instance.grid_size[1]} Grid')
    ax.set_xlabel('X Coordinate'); ax.set_ylabel('Y Coordinate')
    ax.set_xlim(-2, instance.grid_size[0] + 2); ax.set_ylim(-2, instance.grid_size[1] + 2)
    ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    plt.savefig(save_path)
    plt.close(fig)

def save_dag_plot(instance: ProblemInstance, save_path: str) -> Dict[str, float]:
    """
    Generates and saves a Directed Acyclic Graph (DAG) plot of all feasible arcs.
    
    FIXED: Uses a simpler, more robust layout for Depots and Charging Stations.
    
    Returns the computed y-positions for trips, to be reused by the solution plot.
    """
    G = nx.DiGraph()

    origin_nodes = [f"O{d.id}"for d in instance.depots]
    dest_nodes = [f"D{d.id}"for d in instance.depots]
    trip_nodes = [f"T{t.id}"for t in instance.trips]
    # --- NEW: Charging Station Nodes ---
    cs_nodes = [f"C{i + 1}" for i, cs in enumerate(instance.charging_stations)]
    cs_map = {node_id: cs for node_id, cs in zip(cs_nodes, instance.charging_stations)}
    # ---

    G.add_nodes_from(origin_nodes, type='origin')
    G.add_nodes_from(dest_nodes, type='dest')
    G.add_nodes_from(trip_nodes, type='trip')
    # --- NEW ---
    G.add_nodes_from(cs_nodes, type='charging')
    # ---

    # --- Instance-Type Detection ---
    is_tw_instance = False
    if instance.trips and \
       hasattr(instance.trips[0], 'start_time_window') and \
       instance.trips[0].start_time_window is not None:
        is_tw_instance = True
    # ---
    
    # Placeholder for average speed for feasibility check (e.g., from eb_md_vsp_tw_solver.py)
    AVG_U_METERS_PER_MINUTE = 800000 / 60 

    # --- Arc Generation (Standard O-T, T-T, T-D) ---
    for depot in instance.depots:
        for trip in instance.trips:
            travel_time = math.sqrt((depot.location.x - trip.start_point.x) ** 2 + (depot.location.y - trip.start_point.y) ** 2)
            if is_tw_instance:
                if 0 + travel_time <= trip.start_time_window[1]:
                    G.add_edge(f"O{depot.id}", f"T{trip.id}")
            else:
                if 0 + travel_time <= trip.start_time:
                    G.add_edge(f"O{depot.id}", f"T{trip.id}")
    for trip1 in instance.trips:
        for trip2 in instance.trips:
            if trip1.id == trip2.id: continue
            travel_time = math.sqrt((trip1.end_point.x - trip2.start_point.x) ** 2 + (trip1.end_point.y - trip2.start_point.y) ** 2)
            if is_tw_instance:
                trip1_duration = trip1.end_time - trip1.start_time
                if trip1.start_time_window[0] + trip1_duration + travel_time <= trip2.start_time_window[1]:
                    G.add_edge(f"T{trip1.id}", f"T{trip2.id}")
            else:
                if trip1.end_time + travel_time <= trip2.start_time:
                    G.add_edge(f"T{trip1.id}", f"T{trip2.id}")
    for trip in instance.trips:
        for depot in instance.depots:
            G.add_edge(f"T{trip.id}", f"D{depot.id}")
    # --- End Standard Arc Generation ---
    
    # --- NEW: Arc Generation (Involving Charging Stations) ---
    for trip in instance.trips:
        trip_node = f"T{trip.id}"
        
        for cs_id, cs_obj in cs_map.items():
            # T -> C arcs: Trip i end point to Charging Station c (Assume feasible for full graph visualization)
            G.add_edge(trip_node, cs_id) 
            
            # C -> T arcs: Charging Station c to Trip j start point
            deadhead_distance = math.sqrt((cs_obj.location.x - trip.start_point.x) ** 2 + (cs_obj.location.y - trip.start_point.y) ** 2)
            if is_tw_instance:
                 travel_time = deadhead_distance / AVG_U_METERS_PER_MINUTE 
                 # Check if earliest departure from CS allows arrival before trip's latest start time
                 if cs_obj.time_window[0] + travel_time <= trip.start_time_window[1]:
                     G.add_edge(cs_id, trip_node)
            else:
                 G.add_edge(cs_id, trip_node) # Assume feasible for fixed-time
    # --- END NEW ARC GENERATION ---
    
    trip_y = _compute_trip_y(instance)

    pos = {}
    max_time = 100 # Default max time

    # Calculate max_time and assign X-position for trip nodes and initial X for CS
    if instance.trips:
        if is_tw_instance:
            max_cs_time = max([cs.time_window[1] for cs in instance.charging_stations] + [0])
            max_time = max(max(t.start_time_window[1] for t in instance.trips), max_cs_time) if instance.trips else 100
            for trip in instance.trips:
                pos[f"T{trip.id}"] = (trip.start_time_window[0], trip_y[f"T{trip.id}"])
            for cs_id, cs_obj in cs_map.items():
                pos[cs_id] = (cs_obj.time_window[0], 0) # Initial X-position (uses earliest time)
        else:
            max_time = max(t.end_time for t in instance.trips) if instance.trips else 100
            for trip in instance.trips:
                pos[f"T{trip.id}"] = (trip.start_time, trip_y[f"T{trip.id}"])
            if cs_nodes:
                cs_x_pos = max_time / 2 
                for cs_id in cs_nodes:
                    pos[cs_id] = (cs_x_pos, 0) # Initial X-position (use central time)
    else:
        max_time = 100

    # --- **FIXED** Layout Logic for Depots and Charging Stations ---
    # Find the range occupied by trip nodes
    if trip_y:
        y_values = list(trip_y.values())
        # Set a buffer zone around the trips
        trip_y_max = max(y_values) + 5
        trip_y_min = min(y_values) - 5
    else:
        # Default range if no trips
        trip_y_max = 5
        trip_y_min = -5

    # 1. Layout Depots BELOW the trips
    n_depots = len(origin_nodes)
    if n_depots > 0:
        depot_y_start = trip_y_min - 5
        depot_y_end = trip_y_min - 5 - (n_depots - 1) * 10 
        y_step_depot = (depot_y_end - depot_y_start) / max(1, n_depots - 1) if n_depots > 1 else 0
        
        for i, node in enumerate(origin_nodes):
            y_pos = depot_y_start + (i * y_step_depot)
            pos[node] = (0, y_pos)

        for i, node in enumerate(dest_nodes):
            y_pos = depot_y_start + (i * y_step_depot)
            pos[node] = (max_time + 50, y_pos)

    # 2. Layout Charging Stations at the TOP MIDDLE
    n_cs = len(cs_nodes)
    if n_cs > 0:
        # **A. Determine the fixed Y-position (Horizontal Rank)**
        # Set them well above the highest trip node (trip_y_max + 5)
        # We use a single Y value to keep them horizontal.
        fixed_cs_y = trip_y_max + 10 # Increase buffer to 10 for safety/clarity

        # **B. Determine the X-position (Middle placement)**
        # Calculate the average X-position of the trip nodes to center the CS nodes horizontally.
        avg_trip_x = sum(pos[t][0] for t in trip_nodes) / len(trip_nodes) if trip_nodes else max_time / 2
        
        # Calculate X positions to spread the CS nodes evenly around the average trip X
        # For example, centered_cs_x_start/end ensures they are centered horizontally.
        cs_width = (n_cs - 1) * 20 # Arbitrary visual spacing width
        
        centered_cs_x_start = avg_trip_x - (cs_width / 2)
        
        for i, node in enumerate(cs_nodes):
            # Calculate an X-position to spread them horizontally (side-by-side)
            x_pos = centered_cs_x_start + (i * 60)
            
            # Use the calculated X position and the fixed Y position
            pos[node] = (x_pos, fixed_cs_y)
            
    # (Plotting commands)
    plt.figure(figsize=(20, 10))
    handle_origin = nx.draw_networkx_nodes(G, pos, nodelist=origin_nodes, node_color='green', node_shape='s')
    handle_dest = nx.draw_networkx_nodes(G, pos, nodelist=dest_nodes, node_color='red', node_shape='s')
    handle_trip = nx.draw_networkx_nodes(G, pos, nodelist=trip_nodes, node_color='skyblue', node_shape='o')
    # --- NEW: Plot Charging Station Nodes ---
    handle_cs = nx.draw_networkx_nodes(G, pos, nodelist=cs_nodes, node_color='gold', node_shape='^')
    # ---
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.2, style='dashed')
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Problem Instance as a Directed Acyclic Graph (DAG) of Feasible Arcs")
    plt.xlabel("Time (minutes)"); plt.ylabel("Visual Separation (no specific meaning)")
    plt.grid(True, axis='x', linestyle=':')
    
    # --- MODIFIED Legend ---
    legend_handles = [handle_origin, handle_dest, handle_trip, handle_cs]
    legend_labels = ['Origin Depots', 'Destination Depots', 'Trips', 'Charging Stations']
    plt.legend(legend_handles, legend_labels,
               bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return trip_y # Return layout for reuse

def save_solution_dag_plot(instance: ProblemInstance, schedules: Dict[str, Any], save_path: str, trip_y: Optional[Dict[str, float]] = None):
    """
    Generates and saves a DAG plot showing only the arcs used in the final solution.
    
    FIXED: Uses the robust layout logic for Depots and Charging Stations.
    """
    G_sol = nx.DiGraph()

    origin_nodes = [f"O{d.id}"for d in instance.depots]
    dest_nodes = [f"D{d.id}"for d in instance.depots]
    trip_nodes = [f"T{t.id}"for t in instance.trips]
    # --- NEW: Charging Station Nodes ---
    cs_nodes = [f"C{i + 1}" for i, cs in enumerate(instance.charging_stations)]
    cs_map = {node_id: cs for node_id, cs in zip(cs_nodes, instance.charging_stations)}
    # ---

    G_sol.add_nodes_from(origin_nodes, type='origin')
    G_sol.add_nodes_from(dest_nodes, type='dest')
    G_sol.add_nodes_from(trip_nodes, type='trip')
    # --- NEW ---
    G_sol.add_nodes_from(cs_nodes, type='charging')
    # ---

    edge_colors = {}
    vehicle_colors = ['orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

    for i, (vehicle_id, route) in enumerate(schedules.items()):
        color = vehicle_colors[i % len(vehicle_colors)]
        for j in range(len(route) - 1):
            u, v = route[j], route[j + 1]
            G_sol.add_edge(u, v)
            edge_colors[(u, v)] = color

    pos = {}
    if trip_y is None:
        trip_y = _compute_trip_y(instance)  # deterministic fallback

    # --- Instance-Type Detection ---
    is_tw_instance = False
    if instance.trips and \
       hasattr(instance.trips[0], 'start_time_window') and \
       instance.trips[0].start_time_window is not None:
        is_tw_instance = True
    # ---

    max_time = 100
    # Calculate max_time and assign X-position for trip nodes and initial X for CS
    if instance.trips:
        if is_tw_instance:
            max_cs_time = max([cs.time_window[1] for cs in instance.charging_stations] + [0])
            max_time = max(max(t.start_time_window[1] for t in instance.trips), max_cs_time) if instance.trips else 100
            for trip in instance.trips:
                pos[f"T{trip.id}"] = (trip.start_time_window[0], trip_y[f"T{trip.id}"])
            for cs_id, cs_obj in cs_map.items():
                pos[cs_id] = (cs_obj.time_window[0], 0) # Initial X-position (uses earliest time)
        else:
            max_time = max(t.end_time for t in instance.trips) if instance.trips else 100
            for trip in instance.trips:
                pos[f"T{trip.id}"] = (trip.start_time, trip_y[f"T{trip.id}"])
            if cs_nodes:
                cs_x_pos = max_time / 2 
                for cs_id in cs_nodes:
                    pos[cs_id] = (cs_x_pos, 0) # Initial X-position (use central time)
    else:
        max_time = 100

    # --- **FIXED** Layout Logic for Depots and Charging Stations (Identical to save_dag_plot) ---
    if trip_y:
        y_values = list(trip_y.values())
        trip_y_max = max(y_values) + 5
        trip_y_min = min(y_values) - 5
    else:
        trip_y_max = 5
        trip_y_min = -5

    # 1. Layout Depots BELOW the trips
    n_depots = len(origin_nodes)
    if n_depots > 0:
        depot_y_start = trip_y_min - 5
        depot_y_end = trip_y_min - 5 - (n_depots - 1) * 10 
        y_step_depot = (depot_y_end - depot_y_start) / max(1, n_depots - 1) if n_depots > 1 else 0
        
        for i, node in enumerate(origin_nodes):
            y_pos = depot_y_start + (i * y_step_depot)
            pos[node] = (0, y_pos)

        for i, node in enumerate(dest_nodes):
            y_pos = depot_y_start + (i * y_step_depot)
            pos[node] = (max_time + 50, y_pos)

    # 2. Layout Charging Stations at the TOP MIDDLE
    n_cs = len(cs_nodes)
    if n_cs > 0:
        # **A. Determine the fixed Y-position (Horizontal Rank)**
        fixed_cs_y = trip_y_max + 10 

        # **B. Determine the X-position (Middle placement)**
        avg_trip_x = sum(pos[t][0] for t in trip_nodes) / len(trip_nodes) if trip_nodes else max_time / 2
        
        # Calculate X positions to spread the CS nodes evenly around the average trip X
        cs_width = (n_cs - 1) * 20
        
        centered_cs_x_start = avg_trip_x - (cs_width / 2)
        
        for i, node in enumerate(cs_nodes):
            # Calculate an X-position to spread them horizontally (side-by-side)
            x_pos = centered_cs_x_start + (i * 60)
            
            # Use the calculated X position and the fixed Y position
            pos[node] = (x_pos, fixed_cs_y)
    
    # (Plotting commands)
    plt.figure(figsize=(20, 10))
    handle_origin = nx.draw_networkx_nodes(G_sol, pos, nodelist=origin_nodes, node_color='green', node_shape='s')
    handle_dest = nx.draw_networkx_nodes(G_sol, pos, nodelist=dest_nodes, node_color='red', node_shape='s')
    handle_trip = nx.draw_networkx_nodes(G_sol, pos, nodelist=trip_nodes, node_color='skyblue', node_shape='o')
    # --- NEW: Plot Charging Station Nodes ---
    handle_cs = nx.draw_networkx_nodes(G_sol, pos, nodelist=cs_nodes, node_color='gold', node_shape='^')
    # ---
    nx.draw_networkx_edges(G_sol, pos, edgelist=edge_colors.keys(), edge_color=edge_colors.values(),
                           arrows=True, width=2.0)
    nx.draw_networkx_labels(G_sol, pos, font_size=8)
    plt.title("Solved Instance as a Directed Acyclic Graph (DAG)")
    plt.xlabel("Time (minutes)"); plt.ylabel("Visual Separation")
    plt.grid(True, axis='x', linestyle=':')

    legend_handles = [handle_origin, handle_dest, handle_trip, handle_cs]
    legend_labels = ['Origin Depots', 'Destination Depots', 'Trips', 'Charging Stations']
    for i, vehicle_id in enumerate(schedules.keys()):
        color = vehicle_colors[i % len(vehicle_colors)]
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2))
        legend_labels.append(f"Route for {vehicle_id}")

    # --- MODIFIED Legend ---
    plt.legend(handles=legend_handles, labels=legend_labels,
               bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(schedules.keys()) + 4,
               fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_solution_plot(instance: ProblemInstance, schedules: Dict[str, Any], save_dir: str):
    """
    Generates and saves a separate geographical plot for each vehicle's route (bus block).
    
    ADAPTED: Now traces paths to and from Charging Station locations.
    """
    # Create mappings from abstract node IDs to physical Point objects
    depot_map = {f"O{d.id}": d.location for d in instance.depots}
    depot_map.update({f"D{d.id}": d.location for d in instance.depots})
    
    # --- NEW: Charging Station Map ---
    cs_map = {f"C{i+1}": cs.location for i, cs in enumerate(instance.charging_stations)}
    # ---

    trips_map = {f"T{t.id}": t for t in instance.trips}
    
    vehicle_colors = ['orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

    for i, (vehicle_id, route) in enumerate(schedules.items()):
        fig, ax = plt.subplots(figsize=(10, 10))
        color = vehicle_colors[i % len(vehicle_colors)]

        # 1. Plot the base map (all relief points, depots, and charging stations)
        relief_x = [p.x for p in instance.relief_points]; relief_y = [p.y for p in instance.relief_points]
        depot_x = [d.location.x for d in instance.depots]; depot_y = [d.location.y for d in instance.depots]
        # --- NEW ---
        cs_x = [cs.location.x for cs in instance.charging_stations]; cs_y = [cs.location.y for cs in instance.charging_stations]
        # ---
        ax.scatter(relief_x, relief_y, c='gray', marker='o', label='Relief Points', alpha=0.3, s=20)
        ax.scatter(depot_x, depot_y, c='black', marker='s', s=120, label='Depots', edgecolors='black')
        # --- NEW ---
        ax.scatter(cs_x, cs_y, c='green', marker='^', s=100, label='Charging Stations', edgecolors='black', zorder=3)
        # ---

        # Add text labels for all points on the base map
        for idx, point in enumerate(instance.relief_points):
            ax.text(point.x + 0.5, point.y + 0.5, str(idx + 1), fontsize=9, color='dimgray')
        for depot in instance.depots:
            ax.text(depot.location.x + 0.5, depot.location.y + 0.5, f"Depot {depot.id}",
                    fontsize=10, color='black', weight='bold')
        # --- NEW ---
        for idx, cs in enumerate(instance.charging_stations):
             ax.text(cs.location.x + 0.5, cs.location.y - 1.0, f"CS {idx + 1}", 
                fontsize=10, color='darkgreen', weight='bold')
        # ---

        # 2. Trace the physical path of the current vehicle
        current_point = depot_map[route[0]]

        for j in range(1, len(route)):
            node_id = route[j]
            prev_node_id = route[j-1]

            if node_id.startswith('T'):
                # Trip: Arriving (Deadhead)
                next_point = trips_map[node_id].start_point
                label_dh = 'Deadhead Travel' if (j == 1 or prev_node_id.startswith('C')) else "" 
                ax.plot([current_point.x, next_point.x], [current_point.y, next_point.y],
                        color=color, linestyle=':', alpha=0.8, label=label_dh)
                current_point = next_point

                # Trip: Service (Revenue)
                next_point = trips_map[node_id].end_point
                label_rev = 'Trip' if j == 1 else ""
                ax.plot([current_point.x, next_point.x], [current_point.y, next_point.y],
                        color=color, marker='.', markersize=8, linestyle='-', linewidth=2.5,
                        label=label_rev)
                current_point = next_point

            # --- NEW: Charging Station Travel ---
            elif node_id.startswith('C'):
                next_point = cs_map[node_id]
                label_charge_dh = 'Deadhead to Charging' if prev_node_id.startswith('T') and j == 1 else ""
                ax.plot([current_point.x, next_point.x], [current_point.y, next_point.y],
                        color='gold', linestyle='--', alpha=1.0, linewidth=2.0, label=label_charge_dh)
                # ax.scatter(next_point.x, next_point.y, c=color, marker='^', s=150, zorder=4)
                current_point = next_point
            # ---

            elif node_id.startswith('D'):
                next_point = depot_map[node_id]
                # End of block travel (Deadhead)
                ax.plot([current_point.x, next_point.x], [current_point.y, next_point.y],
                        color=color, linestyle=':', alpha=0.8)
                current_point = next_point

        # 3. Formatting
        ax.set_title(f"Bus Block for Vehicle {vehicle_id}")
        ax.set_xlabel('X Coordinate'); ax.set_ylabel('Y Coordinate')
        ax.set_xlim(-2, instance.grid_size[0] + 2); ax.set_ylim(-2, instance.grid_size[1] + 2)
        ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.5)

        # Ensure all unique labels are collected for the legend
        handles, labels = ax.get_legend_handles_labels()
        if 'Deadhead to Charging' not in labels and any(n.startswith('C') for n in route):
             handles.append(plt.Line2D([0], [0], color='gold', lw=2, linestyle='--'))
             labels.append('Deadhead to Charging')
        
        # Merge handles and labels to keep only unique ones
        unique_labels_map = dict(zip(labels, handles))
        if 'All Charging Stations' not in unique_labels_map and instance.charging_stations:
             unique_labels_map['All Charging Stations'] = plt.scatter([], [], c='green', marker='^', s=100)

        ax.legend(unique_labels_map.values(), unique_labels_map.keys())

        # 4. Save the individual plot
        filename = f"{vehicle_id}_block.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)