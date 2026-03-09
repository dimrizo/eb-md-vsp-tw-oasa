"""
Stateful rolling-horizon orchestrator.

Runs clusters sequentially and propagates fleet state using
solver-produced bus_end_states_*.json files.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os, json, glob, datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional
from gurobipy import GRB
import time

import shutil

from models import eb_md_vsp_tw_solver_with_vi as solver

# ==================== CONFIG ====================
PROJECT_ROOT = os.path.normpath(os.path.join(solver.project_root, ".."))

CLUSTERS_ROOT = os.path.join(PROJECT_ROOT, "output", "clusters_synthetic", "D2_S3_C30_c_trips.txt")
ORCHESTRATED_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "orchestrated_runs")
BUS_END_STATES_DIR = os.path.join(PROJECT_ROOT, "output", "bus_end_states")

# ==================== DATA ====================

@dataclass(frozen=True)
class ClusterInfo:
    name: str
    instance_txt: str
    n_trips: int

# ==================== HELPERS ====================

def vehicle_ids_from_instance(inst) -> List[str]:
    ids = []
    for depot in inst.depots:
        for i in range(1, depot.vehicle_count + 1):
            ids.append(f"D{depot.id}_V{i}")
    return ids

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def reset_bus_end_states_dir(dir_path: str) -> None:
    if os.path.isdir(dir_path):
        for f in glob.glob(os.path.join(dir_path, "bus_end_states_*.json")):
            os.remove(f)
    else:
        os.makedirs(dir_path, exist_ok=True)

def latest_bus_end_states_file(folder: str) -> Optional[str]:
    files = glob.glob(os.path.join(folder, "bus_end_states_*.json"))
    return max(files, key=os.path.getmtime) if files else None

def read_bus_end_states(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def update_fleet_from_end_states(
    vids: List[str],
    prev_avail: List[float],
    prev_soc: List[float],
    prev_state: List[int],
    end_states: dict
) -> Tuple[List[float], List[float], List[int]]:

    idx = {v: i for i, v in enumerate(vids)}

    avail = list(prev_avail)
    soc = list(prev_soc)

    for vid, st in end_states.items():
        i = idx.get(vid)
        if i is None:
            continue
        if "arrival_time" in st:
            avail[i] = float(st["arrival_time"])
        if "soc" in st:
            soc[i] = float(st["soc"])

    state = list(prev_state)

    for vid in end_states:
        i = idx.get(vid)
        if i is not None:
            state[i] = 0

    return avail, soc, state

def discover_clusters(root: str) -> List[ClusterInfo]:
    clusters = []

    for d in sorted(os.listdir(root)):
        cdir = os.path.join(root, d)
        if not os.path.isdir(cdir):
            continue

        inst_path = os.path.join(cdir, "instance.txt")
        if not os.path.isfile(inst_path):
            continue

        # Read header only to get T
        with open(inst_path) as f:
            header = f.readline().strip().split()
            T = int(float(header[1]))

        clusters.append(
            ClusterInfo(
                name=d,
                instance_txt=inst_path,
                n_trips=T
            )
        )

    if not clusters:
        raise RuntimeError(f"No TXT clusters found in {root}")

    return clusters

def print_cluster_summary(clusters: List[ClusterInfo]) -> None:
    print("\nClusters loaded from TXT instances:")
    header = f"{'Cluster':<12} {'Trips':>6}"
    print(header)
    print("-" * len(header))
    for c in clusters:
        print(f"{c.name:<12} {c.n_trips:>6d}")

def build_batches(clusters: List[ClusterInfo]) -> List[List[ClusterInfo]]:
    return [[c] for c in clusters]

# run single cluster is used from another file
def run_single_cluster(cluster_instance_path: str):
    """
    Runs the solver for exactly ONE cluster instance
    and returns the solver stats dictionary.
    """

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    inst = solver.load_instance_from_txt(cluster_instance_path)

    out_dir = os.path.join(ORCHESTRATED_OUTPUT_ROOT, f"single_cluster_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    (
        _,
        _schedules,
        _,
        stats,
        _,
        sol_status,
        _
    ) = solver.solve_md_vsp_tw_from_instance(
        inst,
        output_dir=out_dir,
        bus_end_states_dir=BUS_END_STATES_DIR,
        run_timestamp=ts
    )

    if sol_status == GRB.INFEASIBLE:
        raise RuntimeError(f"Cluster {cluster_instance_path} infeasible")

    return stats

# ==================== ORCHESTRATOR ====================
def run_stateful_rolling_horizon() -> None:

    reset_bus_end_states_dir(BUS_END_STATES_DIR)

    def reset_previous_runs(*paths: str) -> None:
        for p in paths:
            if os.path.isdir(p):
                shutil.rmtree(p)
            os.makedirs(p, exist_ok=True)

    # ==================== OUTER ITERATION ====================

    clusters = discover_clusters(CLUSTERS_ROOT)

    # Load first cluster to infer fleet structure
    first_inst = solver.load_instance_from_txt(clusters[0].instance_txt)

    # Total fleet size K is in the TXT header
    K = first_inst.meta["K"]

    # Assume equal split across depots (synthetic case)
    # You can refine this later if depots are heterogeneous
    num_depots = len(first_inst.depots)
    base = K // num_depots
    rem = K % num_depots

    buses_per_depot = [
        base + (1 if i < rem else 0)
        for i in range(num_depots)
    ]

    print("\n" + "=" * 80)
    print(f"Starting rolling horizon with buses_per_depot = {buses_per_depot}")
    print("=" * 80)

    # fresh start every attempt
    reset_previous_runs(
        BUS_END_STATES_DIR,
        ORCHESTRATED_OUTPUT_ROOT,
    )

    ensure_dirs(ORCHESTRATED_OUTPUT_ROOT, BUS_END_STATES_DIR)

    if not clusters:
        raise RuntimeError("No clusters found")

    print_cluster_summary(clusters)
    batches = build_batches(clusters)

    master_log = []
    infeasible = False
    total_obj_function = 0

    # ==================== ROLLING HORIZON ====================
    try:
        for b_idx, batch in enumerate(batches):

            print(f"\n=== Batch {b_idx:02d} | {[c.name for c in batch]} ===")

            for c in batch:

                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                run_dir = os.path.join(
                    ORCHESTRATED_OUTPUT_ROOT,
                    f"batch_{b_idx:02d}_{c.name}_{ts}"
                )
                ensure_dirs(run_dir)

                inst = solver.load_instance_from_txt(c.instance_txt)

                # ---- solve ----
                result = solver.solve_md_vsp_tw_from_instance(
                            inst,
                            output_dir=run_dir,
                            bus_end_states_dir=BUS_END_STATES_DIR,
                            run_timestamp=ts
                        )

                (
                    _,
                    schedules,
                    _,
                    _stats,
                    _bus_end_states_unused,
                    sol_status,
                    obj_fun_value
                ) = result

                latest_file = latest_bus_end_states_file(BUS_END_STATES_DIR)
                if latest_file is None:
                    raise RuntimeError("No bus_end_states JSON found after solve")

                if result is None or len(result) < 7:
                    raise RuntimeError("Solver returned invalid result")

                if sol_status == GRB.INFEASIBLE:
                    raise RuntimeError("Solver infeasible.")

                print("\r")

                print(schedules)

                print("\r")

                total_obj_function += obj_fun_value

                master_log.append({
                    "batch": b_idx,
                    "cluster": c.name,
                    "n_trips": c.n_trips,
                })

    except Exception as e:
        print("\n!!! INFEASIBILITY DETECTED !!!")
        print(str(e))
        infeasible = True

    # ==================== SUCCESS OR RETRY ====================
    if not infeasible:
        with open(
            os.path.join(ORCHESTRATED_OUTPUT_ROOT, "master_log.json"), "w"
        ) as f:
            json.dump(
                {
                    "fleet": buses_per_depot,
                    "total_objective": total_obj_function,
                    "runs": master_log,
                },
                f,
                indent=2,
            )

        # final_fleet_size = sum(1 for s in states if s == 0)

        print("\nRolling-horizon orchestration complete.")
        # print(f"Final fleet size: {final_fleet_size}")
        print(f"\nTotal Objective Value: {total_obj_function:.2f}")

        return

if __name__ == "__main__":
    start_time = time.perf_counter()
    run_stateful_rolling_horizon()
    end_time = time.perf_counter()
    elapsed_sec = end_time - start_time

    print(f"\nTotal execution time: {elapsed_sec:.2f} seconds ({elapsed_sec/60:.2f} minutes)")


