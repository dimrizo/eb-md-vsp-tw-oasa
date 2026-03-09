# National Technical University of Athens
# Railways & Transport Lab
# Dimitrios Rizopoulos, Konstantinos Gkiotsalitis

from __future__ import annotations

import os, json, glob, datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional
import csv
from gurobipy import GRB
import time

import eb_md_vsp_tw_solver_with_vi_cluster as solver

# ==================== CONFIG ====================

PROJECT_ROOT = os.path.normpath(os.path.join(solver.project_root, ".."))

CLUSTERS_ROOT = os.path.join(PROJECT_ROOT, "output", "clusters_full")
ORCHESTRATED_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "orchestrated_runs")
BUS_END_STATES_DIR = os.path.join(PROJECT_ROOT, "output", "bus_end_states")

DAY = "monday"
GTFS_FOLDER = os.path.join(PROJECT_ROOT, "input", "gtfs", "oasa_december_2025")
DEPOT_FILE = os.path.join(PROJECT_ROOT, "input", "depots.txt")

STARTING_BUSES_PER_DEPOT = [15, 15]
MAX_TOTAL_BUS_INCREMENTS = 20   # safety cap
INITIAL_AVAILABILITY_TIME = 0.0
INITIAL_SOC = 350.0

MERGE_GAP_MIN = 20.0
MAX_CLUSTERS_PER_BATCH = 1
MAX_TRIPS_PER_BATCH = 20  # or 30, whatever your solver can handle

# ==================== DATA ====================

@dataclass(frozen=True)
class ClusterInfo:
    name: str
    trips_txt: str
    start: float
    end: float
    n_trips: int

# ==================== HELPERS ====================

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def vehicle_ids(STARTING_BUSES_PER_DEPOT: List[int]) -> List[str]:
    ids = []
    for depot_id, n in enumerate(STARTING_BUSES_PER_DEPOT, start=1):
        for i in range(1, n + 1):
            ids.append(f"D{depot_id}_V{i}")
    return ids


def initialize_fleet(
    STARTING_BUSES_PER_DEPOT: List[int]
) -> Tuple[List[float], List[float], List[int]]:
    total = sum(STARTING_BUSES_PER_DEPOT)
    return (
        [INITIAL_AVAILABILITY_TIME] * total,
        [INITIAL_SOC] * total,
        [True] * total
    )

def latest_bus_end_states_file(folder: str) -> Optional[str]:
    files = glob.glob(os.path.join(folder, "bus_end_states_*.json"))
    return max(files, key=os.path.getmtime) if files else None

def update_fleet_from_end_states(
    STARTING_BUSES_PER_DEPOT: List[int],
    prev_avail: List[float],
    prev_soc: List[float],
    prev_state: List[int],
    end_states: dict
) -> Tuple[List[float], List[float], List[int]]:
    vids = vehicle_ids(STARTING_BUSES_PER_DEPOT)
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
    summary_path = os.path.join(root, "clusters_summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing {summary_path}")

    clusters: List[ClusterInfo] = []

    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"cluster_id", "num_trips", "time_start", "time_end"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"clusters_summary.csv missing columns {missing}. "
                f"Found {reader.fieldnames}"
            )

        for row in reader:
            cid = row["cluster_id"].strip()
            name = f"cluster_{cid}"

            trips_txt = os.path.join(root, name, "trips.txt")
            if not os.path.isfile(trips_txt):
                raise FileNotFoundError(trips_txt)

            clusters.append(
                ClusterInfo(
                    name=name,
                    trips_txt=trips_txt,
                    start=float(row["time_start"]),
                    end=float(row["time_end"]),
                    n_trips=int(row["num_trips"]),
                )
            )

    return sorted(clusters, key=lambda c: c.start)

def print_cluster_summary(clusters: List[ClusterInfo]) -> None:
    print("\nClusters loaded from clusters_summary.csv:")
    header = f"{'Cluster':<10} {'Trips':>6} {'Start':>8} {'End':>8} {'Span':>8}"
    print(header)
    print("-" * len(header))

    for c in clusters:
        span = c.end - c.start
        print(
            f"{c.name:<10} "
            f"{c.n_trips:>6d} "
            f"{c.start:>8.1f} "
            f"{c.end:>8.1f} "
            f"{span:>8.1f}"
        )

def build_batches(clusters: List[ClusterInfo]) -> List[List[ClusterInfo]]:
    batches = []
    cur = []
    cur_end = None
    cur_trips = 0

    for c in clusters:
        if not cur:
            cur = [c]
            cur_end = c.end
            cur_trips = c.n_trips
            continue

        close = c.start <= cur_end + MERGE_GAP_MIN
        size_ok = cur_trips + c.n_trips <= MAX_TRIPS_PER_BATCH

        if close and size_ok and len(cur) < MAX_CLUSTERS_PER_BATCH:
            cur.append(c)
            cur_end = max(cur_end, c.end)
            cur_trips += c.n_trips
        else:
            batches.append(cur)
            cur = [c]
            cur_end = c.end
            cur_trips = c.n_trips

    if cur:
        batches.append(cur)

    return batches

def print_fleet_table(STARTING_BUSES_PER_DEPOT, avail, soc, states) -> None:

    vids = vehicle_ids(STARTING_BUSES_PER_DEPOT)

    header = f"{'Vehicle':<8} {'Depot':<6} {'Avail_T':>8} {'SoC':>8} {'Fresh':>6}"
    print("\nVehicle state before solve:")
    print(header)
    print("-" * len(header))

    for vid, t, s, st in zip(vids, avail, soc, states):
        depot = vid.split("_")[0]
        print(f"{vid:<8} {depot:<6} {t:8.1f} {s:8.1f} {st:6d}")

# ==================== ORCHESTRATOR ====================

def run_stateful_rolling_horizon() -> None:

    import shutil

    def reset_previous_runs(*paths: str) -> None:
        for p in paths:
            if os.path.isdir(p):
                shutil.rmtree(p)
            os.makedirs(p, exist_ok=True)

    # ==================== OUTER ITERATION ====================

    buses_per_depot = STARTING_BUSES_PER_DEPOT.copy()
    depot_idx = 0  # round-robin increment
    attempts = 0

    while True:

        print("\n" + "=" * 80)
        print(f"Starting rolling horizon with buses_per_depot = {buses_per_depot}")
        print("=" * 80)

        attempts += 1
        if attempts > MAX_TOTAL_BUS_INCREMENTS:
            raise RuntimeError(
                f"Exceeded max bus increments. Last tried fleet = {buses_per_depot}"
            )

        # fresh start every attempt
        reset_previous_runs(
            BUS_END_STATES_DIR,
            ORCHESTRATED_OUTPUT_ROOT,
        )

        ensure_dirs(ORCHESTRATED_OUTPUT_ROOT, BUS_END_STATES_DIR)

        clusters = discover_clusters(CLUSTERS_ROOT)
        if not clusters:
            raise RuntimeError("No clusters found")

        print_cluster_summary(clusters)
        batches = build_batches(clusters)

        avail, soc, states = initialize_fleet(buses_per_depot)

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

                    solver.output_dir = run_dir
                    solver.bus_end_states_output_dir = BUS_END_STATES_DIR
                    solver.timestamp = ts

                    print(
                        f"Solving {c.name} | trips={c.n_trips} | "
                        f"avail[min,max]=({min(avail):.1f},{max(avail):.1f}) "
                        f"soc[min,max]=({min(soc):.1f},{max(soc):.1f}) | "
                        f"cluster_span=({c.start:.1f},{c.end:.1f})"
                    )

                    print_fleet_table(buses_per_depot, avail, soc, states)

                    inst = solver.load_instance_from_gtfs_cluster(
                        c.trips_txt,
                        DAY,
                        GTFS_FOLDER,
                        DEPOT_FILE,
                        buses_per_depot,
                        avail,
                        soc,
                        1,
                        0,
                        states
                    )

                    # ---- solve ----
                    result = solver.solve_md_vsp_tw_from_instance(inst)

                    if result is None or len(result) < 4:
                        raise RuntimeError("Solver returned invalid result")

                    _, schedules, _, end_states, sol_status, obj_fun_value = result
                    
                    if sol_status == GRB.INFEASIBLE:
                        raise RuntimeError("Solver infeasible.)")

                    avail, soc, states = update_fleet_from_end_states(
                        buses_per_depot, avail, soc, states, end_states
                    )

                    print("\r")

                    print(schedules)

                    print("\r")

                    total_obj_function += obj_fun_value

                    master_log.append({
                        "batch": b_idx,
                        "cluster": c.name,
                        "span": [c.start, c.end],
                    })
                
                print_fleet_table(buses_per_depot, avail, soc, states)

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

            final_fleet_size = sum(1 for s in states if s == 0)

            print("\nRolling-horizon orchestration complete.")
            print(f"Final fleet size: {final_fleet_size}")
            print(f"\nTotal Objective Value: {total_obj_function:.2f}")

            return

        # ---- increase fleet and retry ----
        depot_idx = depot_idx % len(buses_per_depot)
        buses_per_depot[depot_idx] += 1
        print(
            f"\nRetrying with +1 bus at depot {depot_idx + 1} "
            f"→ new fleet {buses_per_depot}"
        )
        depot_idx += 1

if __name__ == "__main__":
    start_time = time.perf_counter()
    run_stateful_rolling_horizon()
    end_time = time.perf_counter()
    elapsed_sec = end_time - start_time

    print(f"\nTotal execution time: {elapsed_sec:.2f} seconds ({elapsed_sec/60:.2f} minutes)")


