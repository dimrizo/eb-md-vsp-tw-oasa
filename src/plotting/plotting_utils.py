# National Technical University of Athens
# Railways & Transport Lab
# Dimitrios Rizopoulos, Konstantinos Gkiotsalitis

"""
Reusable Plotting Utilities for Vehicle Scheduling Problems

This module provides functions to visualize ProblemInstance objects and
their corresponding solution schedules.

It is designed to be "smart" and can handle two types of ProblemInstance:
1. Carpaneto-style: Trips have 'start_time' and 'end_time' attributes.
2. Desaulniers-style: Trips have 'start_time_window' (a tuple) and
   also 'start_time' and 'end_time' for service duration.
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
    (This function is generic and unchanged)
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
    Shows depots and relief points.
    (This function is generic and unchanged)
    """
    relief_x = [p.x for p in instance.relief_points]
    relief_y = [p.y for p in instance.relief_points]
    depot_x = [d.location.x for d in instance.depots]
    depot_y = [d.location.y for d in instance.depots]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(relief_x, relief_y, c='blue', marker='o', label='Relief Points', alpha=0.6)
    ax.scatter(depot_x, depot_y, c='red', marker='s', s=120, label='Depots', edgecolors='black')

    for i, point in enumerate(instance.relief_points):
        ax.text(point.x + 0.5, point.y + 0.5, str(i + 1), fontsize=9, color='navy')
    for depot in instance.depots:
        ax.text(depot.location.x + 0.5, depot.location.y + 0.5, f"Depot {depot.id}",
                fontsize=10, color='darkred', weight='bold')

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
    
    UPDATED: This function is now "smart" and handles both fixed-time
    and time-window instances.
    
    Returns the computed y-positions for trips, to be reused by the solution plot.
    """
    G = nx.DiGraph()

    origin_nodes = [f"O{d.id}"for d in instance.depots]
    dest_nodes = [f"D{d.id}"for d in instance.depots]
    trip_nodes = [f"T{t.id}"for t in instance.trips]

    G.add_nodes_from(origin_nodes, type='origin')
    G.add_nodes_from(dest_nodes, type='dest')
    G.add_nodes_from(trip_nodes, type='trip')

    # --- **CORRECTED** Smart Instance-Type Detection ---
    is_tw_instance = False
    if instance.trips and \
       hasattr(instance.trips[0], 'start_time_window') and \
       instance.trips[0].start_time_window is not None:
        is_tw_instance = True
    # ---

    for depot in instance.depots:
        for trip in instance.trips:
            travel_time = math.sqrt((depot.location.x - trip.start_point.x) ** 2 + (depot.location.y - trip.start_point.y) ** 2)
            
            # --- Smart Edge Logic (Unchanged, but now relies on correct detection) ---
            if is_tw_instance:
                # Use Time Window (Desaulniers-style)
                if 0 + travel_time <= trip.start_time_window[1]:
                    G.add_edge(f"O{depot.id}", f"T{trip.id}")
            else:
                # Use Fixed Start Time (Carpaneto-style)
                if 0 + travel_time <= trip.start_time:
                    G.add_edge(f"O{depot.id}", f"T{trip.id}")
            # ---

    for trip1 in instance.trips:
        for trip2 in instance.trips:
            if trip1.id == trip2.id: continue
            travel_time = math.sqrt((trip1.end_point.x - trip2.start_point.x) ** 2 + (trip1.end_point.y - trip2.start_point.y) ** 2)
            
            # --- Smart Edge Logic (Unchanged, but now relies on correct detection) ---
            if is_tw_instance:
                # Use Time Window (Desaulniers-style)
                trip1_duration = trip1.end_time - trip1.start_time
                if trip1.start_time_window[0] + trip1_duration + travel_time <= trip2.start_time_window[1]:
                    G.add_edge(f"T{trip1.id}", f"T{trip2.id}")
            else:
                # Use Fixed Start Time (Carpaneto-style)
                if trip1.end_time + travel_time <= trip2.start_time:
                    G.add_edge(f"T{trip1.id}", f"T{trip2.id}")
            # ---

    for trip in instance.trips:
        for depot in instance.depots:
            G.add_edge(f"T{trip.id}", f"D{depot.id}")

    trip_y = _compute_trip_y(instance) # Use deterministic Y-layout

    pos = {}
    
    # --- Smart Layout Logic (Unchanged, but now relies on correct detection) ---
    if instance.trips:
        if is_tw_instance:
            max_time = max(t.start_time_window[1] for t in instance.trips)
            for trip in instance.trips:
                pos[f"T{trip.id}"] = (trip.start_time_window[0], trip_y[f"T{trip.id}"])
        else:
            max_time = max(t.end_time for t in instance.trips)
            for trip in instance.trips:
                pos[f"T{trip.id}"] = (trip.start_time, trip_y[f"T{trip.id}"])
    else:
        max_time = 1
    # ---

    # --- NEW: Calculate Y-positions for depots ---
    if trip_y:
        y_values = list(trip_y.values())
        min_y = min(y_values) if y_values else 0
        max_y = max(y_values) if y_values else 10 # Default height if no trips
        
        # Widen the span if all trips are at the same y-level (or only one trip)
        if min_y == max_y:
            min_y -= 5
            max_y += 5
    else:
        # No trips, just space depots by 10
        min_y = 0
        n_depots_check = len(origin_nodes)
        max_y = (n_depots_check - 1) * 10 if n_depots_check > 1 else 10

    n_depots = len(origin_nodes)
    if n_depots > 1:
        # Spread depots evenly across the (min_y, max_y) range
        y_step = (max_y - min_y) / (n_depots - 1)
        start_y = min_y
    else:
        # Center the single depot
        y_step = 0
        start_y = (min_y + max_y) / 2
    # --- END NEW ---
    
    # --- MODIFIED: Use calculated Y-positions ---
    for i, node in enumerate(origin_nodes):
        y_pos = start_y + (i * y_step)
        pos[node] = (0, y_pos)

    for i, node in enumerate(dest_nodes):
        y_pos = start_y + (i * y_step)
        pos[node] = (max_time + 50, y_pos)
    # --- END MODIFIED ---
    
    # (Plotting commands are generic and unchanged)
    plt.figure(figsize=(20, 10))
    handle_origin = nx.draw_networkx_nodes(G, pos, nodelist=origin_nodes, node_color='green', node_shape='s')
    handle_dest = nx.draw_networkx_nodes(G, pos, nodelist=dest_nodes, node_color='red', node_shape='s')
    handle_trip = nx.draw_networkx_nodes(G, pos, nodelist=trip_nodes, node_color='skyblue', node_shape='o')
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.2, style='dashed')
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Problem Instance as a Directed Acyclic Graph (DAG) of Feasible Arcs")
    plt.xlabel("Time (minutes)"); plt.ylabel("Visual Separation (no specific meaning)")
    plt.grid(True, axis='x', linestyle=':')
    plt.legend([handle_origin, handle_dest, handle_trip], ['Origin Depots', 'Destination Depots', 'Trips'],
               bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return trip_y # Return layout for reuse

def save_solution_dag_plot(instance: ProblemInstance, schedules: Dict[str, Any], save_path: str, trip_y: Optional[Dict[str, float]] = None):
    """
    Generates and saves a DAG plot showing only the arcs used in the final solution.
    
    UPDATED: This function is now "smart" and handles both fixed-time
    and time-window instances.
    
    'schedules' is a dict mapping vehicle_id to its route (a list of node IDs).
    'trip_y' is the optional layout from save_dag_plot to ensure consistency.
    """
    G_sol = nx.DiGraph()

    origin_nodes = [f"O{d.id}"for d in instance.depots]
    dest_nodes = [f"D{d.id}"for d in instance.depots]
    trip_nodes = [f"T{t.id}"for t in instance.trips]

    G_sol.add_nodes_from(origin_nodes, type='origin')
    G_sol.add_nodes_from(dest_nodes, type='dest')
    G_sol.add_nodes_from(trip_nodes, type='trip')

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

    # --- **CORRECTED** Smart Instance-Type Detection ---
    is_tw_instance = False
    if instance.trips and \
       hasattr(instance.trips[0], 'start_time_window') and \
       instance.trips[0].start_time_window is not None:
        is_tw_instance = True
    # ---

    # --- Smart Layout Logic (Unchanged, but now relies on correct detection) ---
    if instance.trips:
        if is_tw_instance:
            max_time = max(t.start_time_window[1] for t in instance.trips)
            for trip in instance.trips:
                pos[f"T{trip.id}"] = (trip.start_time_window[0], trip_y[f"T{trip.id}"])
        else:
            max_time = max(t.end_time for t in instance.trips)
            for trip in instance.trips:
                pos[f"T{trip.id}"] = (trip.start_time, trip_y[f"T{trip.id}"])
    else:
        max_time = 1
    # ---

    # --- NEW: Calculate Y-positions for depots ---
    if trip_y:
        y_values = list(trip_y.values())
        min_y = min(y_values) if y_values else 0
        max_y = max(y_values) if y_values else 10 # Default height if no trips
        
        # Widen the span if all trips are at the same y-level (or only one trip)
        if min_y == max_y:
            min_y -= 5
            max_y += 5
    else:
        # No trips, just space depots by 10
        min_y = 0
        n_depots_check = len(origin_nodes)
        max_y = (n_depots_check - 1) * 10 if n_depots_check > 1 else 10

    n_depots = len(origin_nodes)
    if n_depots > 1:
        # Spread depots evenly across the (min_y, max_y) range
        y_step = (max_y - min_y) / (n_depots - 1)
        start_y = min_y
    else:
        # Center the single depot
        y_step = 0
        start_y = (min_y + max_y) / 2
    # --- END NEW ---
    
    # --- MODIFIED: Use calculated Y-positions ---
    for i, node in enumerate(origin_nodes):
        y_pos = start_y + (i * y_step)
        pos[node] = (0, y_pos)

    for i, node in enumerate(dest_nodes):
        y_pos = start_y + (i * y_step)
        pos[node] = (max_time + 50, y_pos)
    # --- END MODIFIED ---
    
    # (Plotting commands are generic and unchanged)
    plt.figure(figsize=(20, 10))
    handle_origin = nx.draw_networkx_nodes(G_sol, pos, nodelist=origin_nodes, node_color='green', node_shape='s')
    handle_dest = nx.draw_networkx_nodes(G_sol, pos, nodelist=dest_nodes, node_color='red', node_shape='s')
    handle_trip = nx.draw_networkx_nodes(G_sol, pos, nodelist=trip_nodes, node_color='skyblue', node_shape='o')
    nx.draw_networkx_edges(G_sol, pos, edgelist=edge_colors.keys(), edge_color=edge_colors.values(),
                           arrows=True, width=2.0)
    nx.draw_networkx_labels(G_sol, pos, font_size=8)
    plt.title("Solved Instance as a Directed Acyclic Graph (DAG)")
    plt.xlabel("Time (minutes)"); plt.ylabel("Visual Separation")
    plt.grid(True, axis='x', linestyle=':')

    legend_handles = [handle_origin, handle_dest, handle_trip]
    legend_labels = ['Origin Depots', 'Destination Depots', 'Trips']
    for i, vehicle_id in enumerate(schedules.keys()):
        color = vehicle_colors[i % len(vehicle_colors)]
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2))
        legend_labels.append(f"Route for {vehicle_id}")

    plt.legend(handles=legend_handles, labels=legend_labels,
               bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(schedules.keys()) + 3,
               fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_solution_plot(instance: ProblemInstance, schedules: Dict[str, Any], save_dir: str):
    """
    Generates and saves a separate geographical plot for each vehicle's route (bus block).
    (This function is generic and unchanged)
    """
    # Create mappings from abstract node IDs to physical Point objects
    depot_map = {f"O{d.id}": d.location for d in instance.depots}
    depot_map.update({f"D{d.id}": d.location for d in instance.depots})

    trips_map = {f"T{t.id}": t for t in instance.trips}

    vehicle_colors = ['orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

    for i, (vehicle_id, route) in enumerate(schedules.items()):
        fig, ax = plt.subplots(figsize=(10, 10))
        color = vehicle_colors[i % len(vehicle_colors)]

        # 1. Plot the base map (all relief points and depots)
        relief_x = [p.x for p in instance.relief_points]; relief_y = [p.y for p in instance.relief_points]
        depot_x = [d.location.x for d in instance.depots]; depot_y = [d.location.y for d in instance.depots]
        ax.scatter(relief_x, relief_y, c='gray', marker='o', label='All Relief Points', alpha=0.3, s=20)
        ax.scatter(depot_x, depot_y, c='black', marker='s', s=120, label='All Depots', edgecolors='black')

        # Add text labels for all points on the base map
        for idx, point in enumerate(instance.relief_points):
            ax.text(point.x + 0.5, point.y + 0.5, str(idx + 1), fontsize=9, color='dimgray')
        for depot in instance.depots:
            ax.text(depot.location.x + 0.5, depot.location.y + 0.5, f"Depot {depot.id}",
                    fontsize=10, color='black', weight='bold')

        # 2. Trace the physical path of the current vehicle
        current_point = depot_map[route[0]]

        for j in range(1, len(route)):
            node_id = route[j]

            if node_id.startswith('T'):
                next_point = trips_map[node_id].start_point
                ax.plot([current_point.x, next_point.x], [current_point.y, next_point.y],
                        color=color, linestyle=':', alpha=0.8, label='Deadhead Travel' if j == 1 else "")
                current_point = next_point

                next_point = trips_map[node_id].end_point
                ax.plot([current_point.x, next_point.x], [current_point.y, next_point.y],
                        color=color, marker='.', markersize=8, linestyle='-', linewidth=2.5,
                        label='Revenue Trip' if j == 1 else "")
                current_point = next_point

            elif node_id.startswith('D'):
                next_point = depot_map[node_id]
                ax.plot([current_point.x, next_point.x], [current_point.y, next_point.y],
                        color=color, linestyle=':', alpha=0.8)

        # 3. Formatting
        ax.set_title(f"Bus Block for Vehicle {vehicle_id}")
        ax.set_xlabel('X Coordinate'); ax.set_ylabel('Y Coordinate')
        ax.set_xlim(-2, instance.grid_size[0] + 2); ax.set_ylim(-2, instance.grid_size[1] + 2)
        ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

        # 4. Save the individual plot
        filename = f"{vehicle_id}_block.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)