# National Technical University of Athens
# Railways & Transport Lab
# Dimitrios Rizopoulos, Konstantinos Gkiotsalitis

"""
Stateful rolling-horizon orchestrator with selective exact cluster merging.

This file deliberately removes the old RNI neighborhood machinery:
- no local trip shifts,
- no adjacent swaps,
- no cross-vehicle relocation,
- no short-duty absorption,
- no route-pair merge,
- no block-compression repair.

Physical idea
-------------
1) Solve the original clusters sequentially with the exact MIP.
2) Preserve the physical end state of every bus after each solved block.
3) After the rolling-horizon solution is built, test only neighboring cluster-block merges.
4) A candidate merge solves the neighboring clusters together as one exact MIP from the
   fleet state before the first block in the pair.
5) If the merge changes fleet states, all downstream blocks are replayed/re-solved exactly.
6) Accept a merge only if the full recomputed schedule objective improves.
7) Never insert trips from a later block into an earlier block while pretending the later
   block still exists. Accepted windows become explicit merged blocks.

This is not Simulated Annealing and not RNI. It is a selective exact cluster-merging
post-improvement for rolling horizon.

Fixed version:
- preserves exact/global trip-ID mappings, and
- safely handles ROLLING_HORIZON_TIME_LIMIT_SEC = None during selective merging.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from gurobipy import GRB

import eb_md_vsp_tw_solver_with_vi_cluster as solver

# ==================== CONFIG ====================

PROJECT_ROOT = os.path.normpath(os.path.join(solver.project_root, ".."))

CLUSTERS_ROOT = os.path.join(PROJECT_ROOT, "output", "clusters")
ORCHESTRATED_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "orchestrated_runs")
# Internal solve/candidate artifacts are kept outside orchestrated_runs so that
# orchestrated_runs can contain only the final accepted solution in the same
# structure as the original rolling-horizon orchestrator.
WORKING_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "orchestrated_runs_work")
BUS_END_STATES_DIR = os.path.join(PROJECT_ROOT, "output", "bus_end_states")

DAY = "monday"
GTFS_FOLDER = os.path.join(PROJECT_ROOT, "input", "gtfs", "oasa_third_results_section")
DEPOT_FILE = os.path.join(PROJECT_ROOT, "input", "depots.txt")

STARTING_BUSES_PER_DEPOT = [5, 5]
MAX_TOTAL_BUS_INCREMENTS = 20
INITIAL_AVAILABILITY_TIME = 0.0
INITIAL_SOC = 350.0

MERGE_GAP_MIN = 20.0
MAX_CLUSTERS_PER_BATCH = 1
MAX_TRIPS_PER_BATCH = 20

# ==================== TIME LIMIT CONFIG ====================

GUROBI_TIME_LIMIT_SEC = 18000
ROLLING_HORIZON_TIME_LIMIT_SEC = None

# If True, a Gurobi TIME_LIMIT solution with an incumbent is accepted.
ACCEPT_TIME_LIMIT_INCUMBENTS = True

# If True, every block must be proven optimal.
# For computational complexity runs with time limits, this should usually be False.
REQUIRE_OPTIMAL_BLOCK_SOLVES = False

# ==================== SELECTIVE EXACT MERGING CONFIG ====================

APPLY_SELECTIVE_EXACT_CLUSTER_MERGING = False
ACCEPT_FINAL_MERGED_SOLUTION_ONLY_IF_IMPROVES_RH = True

# Only adjacent current blocks are tested: block i + block i+1.
SELECTIVE_MERGE_MAX_PASSES = 3
SELECTIVE_MERGE_MAX_TRIPS_PER_WINDOW = 10
SELECTIVE_MERGE_ACCEPTANCE_EPS = 1e-6
SELECTIVE_MERGE_FIRST_IMPROVEMENT = False

# Prevent the post-improvement from silently becoming the full exact model.
SELECTIVE_MERGE_FORBID_FULL_INSTANCE = True

# Correct physical accounting: if a merge is accepted or tested, downstream blocks must be
# re-solved from the new bus states caused by the candidate merged block.
SELECTIVE_MERGE_REPLAY_DOWNSTREAM = True

# Keep solver call aligned with the existing rolling-horizon code.
NUMBER_OF_CS_PER_DEPOT = 1


# ==================== DATA ====================

@dataclass(frozen=True)
class ClusterInfo:
    name: str
    trips_txt: str
    start: float
    end: float
    n_trips: int


@dataclass
class ScheduleBlock:
    """One physical block in the current rolling-horizon/merged schedule."""

    name: str
    cluster_names: List[str]
    trips_txt: str
    n_trips: int
    span: Tuple[float, float]
    pre_avail: List[float]
    pre_soc: List[float]
    pre_states: List[int]
    post_avail: List[float]
    post_soc: List[float]
    post_states: List[int]
    schedules: Dict[str, List[str]]
    end_states: dict
    objective: float
    run_dir: str
    method: str
    instance_meta: dict = field(default_factory=dict)


@dataclass
class MergeCandidateResult:
    accepted_candidate_name: str
    blocks: List[ScheduleBlock]
    total_objective: float
    improvement: float
    merge_run_dir: str
    diagnostic: dict


# ==================== BASIC HELPERS ====================


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _slug(text: str, max_chars: int = 18) -> str:
    """Make compact, Windows-path-safe labels for output folders."""
    cleaned = []
    for ch in str(text):
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {"_", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return (slug[:max_chars] or "block")


def _method_tag(method: str) -> str:
    mapping = {
        "rolling_horizon_exact": "rh",
        "exact_merged_window": "merge",
        "downstream_replay_after_merge": "replay",
    }
    return mapping.get(method, _slug(method, 10))


def _compact_run_name(method: str, block_name: str, timestamp: str) -> str:
    """Shorten nested run folders to avoid Windows MAX_PATH failures."""
    digest = hashlib.md5(f"{method}|{block_name}|{timestamp}".encode("utf-8")).hexdigest()[:6]
    return f"{_method_tag(method)}_{_slug(block_name, 16)}_{digest}_{timestamp}"


def reset_previous_runs(*paths: str) -> None:
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)


def vehicle_ids(buses_per_depot: List[int]) -> List[str]:
    ids: List[str] = []
    for depot_id, n in enumerate(buses_per_depot, start=1):
        for i in range(1, n + 1):
            ids.append(f"D{depot_id}_V{i}")
    return ids


def initialize_fleet(buses_per_depot: List[int]) -> Tuple[List[float], List[float], List[int]]:
    total = sum(buses_per_depot)
    return (
        [INITIAL_AVAILABILITY_TIME] * total,
        [INITIAL_SOC] * total,
        [1] * total,  # 1 means fresh/unused, matching the existing code's convention.
    )


def update_fleet_from_end_states(
    buses_per_depot: List[int],
    prev_avail: List[float],
    prev_soc: List[float],
    prev_state: List[int],
    end_states: dict,
) -> Tuple[List[float], List[float], List[int]]:
    vids = vehicle_ids(buses_per_depot)
    idx = {v: i for i, v in enumerate(vids)}

    avail = list(prev_avail)
    soc = list(prev_soc)
    state = list(prev_state)

    for vid, st in end_states.items():
        i = idx.get(vid)
        if i is None:
            continue
        if "arrival_time" in st:
            avail[i] = float(st["arrival_time"])
        if "soc" in st:
            soc[i] = float(st["soc"])
        state[i] = 0  # 0 means this physical bus has been used before.

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
                f"clusters_summary.csv missing columns {missing}. Found {reader.fieldnames}"
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
        print(
            f"{c.name:<10} {c.n_trips:>6d} {c.start:>8.1f} "
            f"{c.end:>8.1f} {c.end - c.start:>8.1f}"
        )


def build_batches(clusters: List[ClusterInfo]) -> List[List[ClusterInfo]]:
    batches: List[List[ClusterInfo]] = []
    cur: List[ClusterInfo] = []
    cur_end: Optional[float] = None
    cur_trips = 0

    for c in clusters:
        if not cur:
            cur = [c]
            cur_end = c.end
            cur_trips = c.n_trips
            continue

        close = c.start <= float(cur_end) + MERGE_GAP_MIN
        size_ok = cur_trips + c.n_trips <= MAX_TRIPS_PER_BATCH

        if close and size_ok and len(cur) < MAX_CLUSTERS_PER_BATCH:
            cur.append(c)
            cur_end = max(float(cur_end), c.end)
            cur_trips += c.n_trips
        else:
            batches.append(cur)
            cur = [c]
            cur_end = c.end
            cur_trips = c.n_trips

    if cur:
        batches.append(cur)

    return batches


def print_fleet_table(buses_per_depot: List[int], avail: List[float], soc: List[float], states: List[int]) -> None:
    vids = vehicle_ids(buses_per_depot)
    header = f"{'Vehicle':<8} {'Depot':<6} {'Avail_T':>8} {'SoC':>8} {'Fresh':>6}"
    print("\nVehicle state before solve:")
    print(header)
    print("-" * len(header))

    for vid, t, s, st in zip(vids, avail, soc, states):
        depot = vid.split("_")[0]
        print(f"{vid:<8} {depot:<6} {t:8.1f} {s:8.1f} {st:6d}")


def read_trip_ids_from_txt(path: str) -> List[str]:
    """Read a trips.txt file robustly; expected main format is a `trip_id` column."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    header = [h.strip() for h in rows[0]]
    data_rows = rows[1:]
    if "trip_id" in header:
        idx = header.index("trip_id")
    else:
        idx = 0
        data_rows = rows

    trip_ids: List[str] = []
    for row in data_rows:
        if not row or idx >= len(row):
            continue
        trip_id = row[idx].strip()
        if trip_id:
            trip_ids.append(trip_id)
    return trip_ids



def read_trip_identity_rows(path: str) -> List[dict]:
    """
    Read authoritative trip identities from a cluster/merged trips file.

    Required columns:
      - trip_id: original GTFS trip identifier
      - exact_trip_id: authoritative full-instance label, e.g. T65

    The exact_trip_id column must be created by the clustering export from the
    original Trip.id before the trips are split into clusters.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"trip_id", "exact_trip_id"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"{path} is missing columns {sorted(missing)}. "
                "Regenerate the clusters with exact_trip_id preserved."
            )

        rows = []
        for row in reader:
            gtfs_id = str(row.get("trip_id", "")).strip()
            exact_id = str(row.get("exact_trip_id", "")).strip()
            if not gtfs_id or not exact_id:
                continue
            if not re.fullmatch(r"T[1-9]\d*", exact_id):
                raise ValueError(
                    f"Invalid exact_trip_id {exact_id!r} in {path}; "
                    "expected labels such as T1 or T208."
                )
            rows.append({
                "trip_id": gtfs_id,
                "exact_trip_id": exact_id,
            })

    if not rows:
        raise ValueError(f"No trip identities found in {path}")

    return rows


def build_local_to_exact_trip_map(inst, trips_txt: str) -> Dict[str, str]:
    """
    Map the solver-local labels T1..Tn to the exact model's global labels.

    The cluster solver already stores the original GTFS identity as
    trip.gtfs_trip_id. We join that identity to exact_trip_id from trips_txt.
    """
    exact_by_gtfs = {
        row["trip_id"]: row["exact_trip_id"]
        for row in read_trip_identity_rows(trips_txt)
    }

    local_to_exact: Dict[str, str] = {}
    for trip in inst.trips:
        gtfs_id = str(getattr(trip, "gtfs_trip_id", "")).strip()
        if not gtfs_id:
            raise ValueError(
                f"Local trip T{trip.id} has no gtfs_trip_id metadata."
            )
        if gtfs_id not in exact_by_gtfs:
            raise ValueError(
                f"GTFS trip {gtfs_id!r} from the loaded instance is absent "
                f"from {trips_txt}."
            )

        local_id = f"T{trip.id}"
        exact_id = exact_by_gtfs[gtfs_id]
        local_to_exact[local_id] = exact_id

    if len(local_to_exact) != len(inst.trips):
        raise ValueError(
            "Could not build a complete local-to-exact trip mapping."
        )

    if len(set(local_to_exact.values())) != len(local_to_exact):
        raise ValueError(
            f"Duplicate exact trip IDs detected in mapping for {trips_txt}."
        )

    return local_to_exact


def write_trip_map_into_solution_json(
    solution_vars_path: str,
    trip_id_map: Dict[str, str],
) -> None:
    """Embed trip_id_map where synthesize_blocks can consume it."""
    data = _read_json_if_nonempty(solution_vars_path)
    if not data:
        raise RuntimeError(
            f"Cannot add trip_id_map because {solution_vars_path} is missing "
            "or empty."
        )

    data["trip_id_map"] = dict(trip_id_map)

    with open(solution_vars_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def write_merged_trips_file(blocks: List[ScheduleBlock], output_dir: str, label: str) -> str:
    """
    Merge input trip files while preserving authoritative exact_trip_id values.
    """
    ensure_dirs(output_dir)
    merged_path = os.path.join(output_dir, f"{label}_trips.txt")

    exact_by_gtfs: Dict[str, str] = {}
    ordered_gtfs_ids: List[str] = []

    for block in blocks:
        for row in read_trip_identity_rows(block.trips_txt):
            gtfs_id = row["trip_id"]
            exact_id = row["exact_trip_id"]

            previous = exact_by_gtfs.get(gtfs_id)
            if previous is not None and previous != exact_id:
                raise ValueError(
                    f"Conflicting exact IDs for GTFS trip {gtfs_id}: "
                    f"{previous} versus {exact_id}."
                )

            if previous is None:
                exact_by_gtfs[gtfs_id] = exact_id
                ordered_gtfs_ids.append(gtfs_id)

    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trip_id", "exact_trip_id"],
        )
        writer.writeheader()
        for gtfs_id in ordered_gtfs_ids:
            writer.writerow({
                "trip_id": gtfs_id,
                "exact_trip_id": exact_by_gtfs[gtfs_id],
            })

    return merged_path


def json_safe_end_states(end_states: dict) -> dict:
    return json.loads(json.dumps(end_states, default=str))


def _read_json_if_nonempty(path: str) -> dict:
    """Read a JSON file only if it exists and contains a non-empty dict."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and data else {}
    except Exception:
        return {}


def _solution_key(*parts: str) -> str:
    """Match the solver's tuple-key serialization in solution_variables.json."""
    return "(" + ",".join(str(part) for part in parts) + ")"


def _derive_end_states_from_solution_variables(
    *,
    schedules: Dict[str, List[str]],
    solution_vars_path: str,
) -> dict:
    """
    Fallback state extractor.

    Some solver versions successfully write solution_variables.json but do not create
    bus_end_states_<timestamp>.json. For downstream rolling-horizon replay we only
    need, for each used bus, the final depot, final arrival time and final SoC.
    Those values are already present in T and E_pre for the final D# node.
    """
    solution_vars = _read_json_if_nonempty(solution_vars_path)
    if not solution_vars:
        return {}

    t_values = solution_vars.get("T", {}) or {}
    e_pre_values = solution_vars.get("E_pre", {}) or {}
    e_bar_values = solution_vars.get("E_bar", {}) or {}

    derived = {}
    for vehicle, route in (schedules or {}).items():
        if not route:
            continue
        final_node = None
        for node_id in reversed(route):
            if isinstance(node_id, str) and node_id.startswith("D"):
                final_node = node_id
                break
        if final_node is None:
            continue

        key = _solution_key(vehicle, final_node)
        arrival_time = float(t_values.get(key, 0.0))
        soc = float(e_pre_values.get(key, e_bar_values.get(key, 0.0)))

        try:
            depot_id = int(final_node[1:])
        except Exception:
            depot_id = None

        derived[vehicle] = {
            "depot_id": depot_id,
            "arrival_time": round(arrival_time, 4),
            "soc": round(soc, 4),
        }

    return derived


def _normalise_end_states(
    *,
    returned_end_states,
    schedules: Dict[str, List[str]],
    expected_end_states_path: str,
    solution_vars_path: str,
) -> dict:
    """
    Use the solver-returned end_states if available; otherwise recover from files.

    This removes the fragile dependency on bus_end_states_<timestamp>.json.
    The JSON side-file is useful, but it must not be required for candidate repair
    evaluation because the solver already returns enough information.
    """
    if isinstance(returned_end_states, dict) and returned_end_states:
        return returned_end_states

    if isinstance(returned_end_states, str):
        data = _read_json_if_nonempty(returned_end_states)
        if data:
            return data

    data = _read_json_if_nonempty(expected_end_states_path)
    if data:
        return data

    data = _derive_end_states_from_solution_variables(
        schedules=schedules,
        solution_vars_path=solution_vars_path,
    )
    if data:
        return data

    raise RuntimeError(
        "Could not obtain bus end states from the solver return value, "
        "bus_end_states JSON, or solution_variables.json."
    )


# ==================== EXACT SOLVE WRAPPER ====================

def solve_exact_block(
    *,
    block_name: str,
    cluster_names: List[str],
    trips_txt: str,
    n_trips: int,
    span: Tuple[float, float],
    buses_per_depot: List[int],
    pre_avail: List[float],
    pre_soc: List[float],
    pre_states: List[int],
    output_root: str,
    method: str,
    require_optimal: bool = True,
    time_limit_sec: float = None,
) -> ScheduleBlock:
    """Solve one physical block exactly and return its full state transition."""
    # Use compact timestamps and run folders. On Windows, deeply nested descriptive
    # paths can exceed the legacy MAX_PATH limit and make the solver fail when it
    # tries to write bus_end_states_*.json, even though the optimization succeeded.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(output_root, _compact_run_name(method, block_name, ts))
    ensure_dirs(run_dir)

    solver.output_dir = run_dir
    solver.bus_end_states_output_dir = os.path.join(run_dir, "bus_end_states")
    solver.timestamp = ts
    ensure_dirs(solver.output_dir, solver.bus_end_states_output_dir)

    expected_end_states_path = os.path.join(
        solver.bus_end_states_output_dir,
        f"bus_end_states_{ts}.json",
    )
    solution_vars_path = os.path.join(solver.output_dir, "solution_variables.json")

    # Some solver/reporting versions try to read this side-file even when they
    # failed to write it. Pre-create a harmless empty dict so a missing-file
    # exception cannot kill an otherwise optimal repair solve. If the solver
    # writes real end states, it will overwrite this file.
    with open(expected_end_states_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

    print(
        f"Solving {block_name} | trips={n_trips} | method={method} | "
        f"avail[min,max]=({min(pre_avail):.1f},{max(pre_avail):.1f}) | "
        f"soc[min,max]=({min(pre_soc):.1f},{max(pre_soc):.1f}) | "
        f"span=({span[0]:.1f},{span[1]:.1f})"
    )
    print_fleet_table(buses_per_depot, pre_avail, pre_soc, pre_states)

    inst = solver.load_instance_from_gtfs_cluster(
        trips_txt,
        DAY,
        GTFS_FOLDER,
        DEPOT_FILE,
        buses_per_depot,
        list(pre_avail),
        list(pre_soc),
        NUMBER_OF_CS_PER_DEPOT,
        0,
        list(pre_states),
    )

    # Preserve the exact/full-instance trip numbering before solving.
    local_to_exact_trip_map = build_local_to_exact_trip_map(
        inst,
        trips_txt,
    )

    effective_time_limit = time_limit_sec
    if effective_time_limit is None:
        effective_time_limit = GUROBI_TIME_LIMIT_SEC

    solver.GUROBI_TIME_LIMIT_SEC = effective_time_limit
    solver.ACCEPT_TIME_LIMIT_INCUMBENTS = ACCEPT_TIME_LIMIT_INCUMBENTS

    try:
        result = solver.solve_md_vsp_tw_from_instance(
            inst,
            time_limit_sec=effective_time_limit,
            accept_time_limit_incumbent=ACCEPT_TIME_LIMIT_INCUMBENTS,
        )
    except TypeError:
        result = solver.solve_md_vsp_tw_from_instance(inst)
    if result is None or len(result) < 6:
        raise RuntimeError(f"Solver returned invalid result for {block_name}")

    _, schedules, _, end_states, sol_status, obj_fun_value = result

    # The solver has now written solution_variables.json. Add the authoritative
    # local T# -> exact/global T# mapping for downstream synthesis.
    write_trip_map_into_solution_json(
        solution_vars_path,
        local_to_exact_trip_map,
    )

    if sol_status == GRB.INFEASIBLE:
        raise RuntimeError(f"Solver infeasible for {block_name}")

    has_schedule = bool(schedules)

    if require_optimal and sol_status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Solver did not prove optimality for {block_name}; status={sol_status}"
        )

    if not require_optimal and not has_schedule:
        raise RuntimeError(
            f"Solver did not return a feasible schedule for {block_name}; status={sol_status}"
        )

    end_states = _normalise_end_states(
        returned_end_states=end_states,
        schedules=schedules,
        expected_end_states_path=expected_end_states_path,
        solution_vars_path=solution_vars_path,
    )

    # Keep the expected side-file populated for audit/debugging.
    with open(expected_end_states_path, "w", encoding="utf-8") as f:
        json.dump(json_safe_end_states(end_states), f, indent=4)

    post_avail, post_soc, post_states = update_fleet_from_end_states(
        buses_per_depot,
        pre_avail,
        pre_soc,
        pre_states,
        end_states,
    )

    return ScheduleBlock(
        name=block_name,
        cluster_names=list(cluster_names),
        trips_txt=trips_txt,
        n_trips=int(n_trips),
        span=(float(span[0]), float(span[1])),
        pre_avail=list(pre_avail),
        pre_soc=list(pre_soc),
        pre_states=list(pre_states),
        post_avail=post_avail,
        post_soc=post_soc,
        post_states=post_states,
        schedules=schedules,
        end_states=end_states,
        objective=float(obj_fun_value),
        run_dir=run_dir,
        method=method,
        instance_meta=dict(getattr(inst, "meta", {}) or {}),
    )


# ==================== SELECTIVE EXACT CLUSTER MERGING ====================


def total_objective(blocks: List[ScheduleBlock]) -> float:
    return sum(float(block.objective) for block in blocks)


def total_original_cluster_count(blocks: List[ScheduleBlock]) -> int:
    names = set()
    for block in blocks:
        names.update(block.cluster_names)
    return len(names)


def _remaining_time_or_none(remaining_time_callback) -> Optional[float]:
    """
    Evaluate an optional whole-process time-limit callback safely.

    A callback may legitimately return None when no global rolling-horizon
    deadline is configured. In that case, selective merging continues and each
    individual solver call uses GUROBI_TIME_LIMIT_SEC.
    """
    if remaining_time_callback is None:
        return None

    remaining = remaining_time_callback()
    if remaining is None:
        return None

    try:
        remaining_value = float(remaining)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "remaining_time_callback must return a number or None; "
            f"received {remaining!r}."
        ) from exc

    return max(0.0, remaining_value)


def replay_downstream_blocks(
    *,
    downstream_blocks: List[ScheduleBlock],
    start_avail: List[float],
    start_soc: List[float],
    start_states: List[int],
    buses_per_depot: List[int],
    output_root: str,
    method_label: str,
    remaining_time_callback=None,
) -> List[ScheduleBlock]:
    """Re-solve downstream blocks after a candidate merge changes the incoming bus states."""
    replayed: List[ScheduleBlock] = []
    avail = list(start_avail)
    soc = list(start_soc)
    states = list(start_states)

    for old_block in downstream_blocks:
        remaining = _remaining_time_or_none(remaining_time_callback)
        if remaining is not None and remaining <= 0.0:
            raise TimeoutError(
                "Rolling-horizon whole-process time limit reached during downstream replay."
            )

        new_block = solve_exact_block(
            block_name=old_block.name,
            cluster_names=old_block.cluster_names,
            trips_txt=old_block.trips_txt,
            n_trips=old_block.n_trips,
            span=old_block.span,
            buses_per_depot=buses_per_depot,
            pre_avail=avail,
            pre_soc=soc,
            pre_states=states,
            output_root=output_root,
            method=method_label,
            require_optimal=REQUIRE_OPTIMAL_BLOCK_SOLVES,
            time_limit_sec=remaining,
        )
        replayed.append(new_block)
        avail = new_block.post_avail
        soc = new_block.post_soc
        states = new_block.post_states

    return replayed


def build_candidate_merge(
    *,
    blocks: List[ScheduleBlock],
    merge_index: int,
    buses_per_depot: List[int],
    original_cluster_count: int,
    pass_idx: int,
    candidate_idx: int,
    output_root: str,
    remaining_time_callback=None,
) -> Optional[MergeCandidateResult]:
    """Try merging blocks[merge_index] and blocks[merge_index+1] exactly."""
    left = blocks[merge_index]
    right = blocks[merge_index + 1]
    merged_cluster_names = left.cluster_names + right.cluster_names
    merged_n_trips = left.n_trips + right.n_trips
    merged_span = (min(left.span[0], right.span[0]), max(left.span[1], right.span[1]))
    merged_name = "+".join(merged_cluster_names)

    diagnostic = {
        "pass": pass_idx,
        "candidate": candidate_idx,
        "merge_index": merge_index,
        "left_block": left.name,
        "right_block": right.name,
        "merged_name": merged_name,
        "merged_n_trips": merged_n_trips,
        "skipped": False,
        "skip_reason": None,
    }

    if merged_n_trips > SELECTIVE_MERGE_MAX_TRIPS_PER_WINDOW:
        diagnostic["skipped"] = True
        diagnostic["skip_reason"] = (
            f"merged window has {merged_n_trips} trips, cap is "
            f"{SELECTIVE_MERGE_MAX_TRIPS_PER_WINDOW}"
        )
        return None

    if (
        SELECTIVE_MERGE_FORBID_FULL_INSTANCE
        and len(set(merged_cluster_names)) >= original_cluster_count
    ):
        diagnostic["skipped"] = True
        diagnostic["skip_reason"] = "merged window would contain all original clusters"
        return None

    candidate_dir = os.path.join(
        output_root,
        f"p{pass_idx:02d}",
        f"c{candidate_idx:02d}_m{merge_index}_{merge_index + 1}",
    )
    ensure_dirs(candidate_dir)

    merged_trips_txt = write_merged_trips_file(
        [left, right],
        candidate_dir,
        f"merged_{merge_index}_{merge_index + 1}",
    )

    remaining = _remaining_time_or_none(remaining_time_callback)
    if remaining is not None and remaining <= 0.0:
        raise TimeoutError(
            "Rolling-horizon whole-process time limit reached during candidate merge."
        )

    merged_block = solve_exact_block(
        block_name=merged_name,
        cluster_names=merged_cluster_names,
        trips_txt=merged_trips_txt,
        n_trips=merged_n_trips,
        span=merged_span,
        buses_per_depot=buses_per_depot,
        pre_avail=left.pre_avail,
        pre_soc=left.pre_soc,
        pre_states=left.pre_states,
        output_root=candidate_dir,
        method="exact_merged_window",
        require_optimal=REQUIRE_OPTIMAL_BLOCK_SOLVES,
        time_limit_sec=remaining,
    )

    prefix = blocks[:merge_index]
    suffix = blocks[merge_index + 2 :]

    if SELECTIVE_MERGE_REPLAY_DOWNSTREAM and suffix:
        replayed_suffix = replay_downstream_blocks(
            downstream_blocks=suffix,
            start_avail=merged_block.post_avail,
            start_soc=merged_block.post_soc,
            start_states=merged_block.post_states,
            buses_per_depot=buses_per_depot,
            output_root=candidate_dir,
            method_label="downstream_replay_after_merge",
            remaining_time_callback=remaining_time_callback,
        )
    else:
        replayed_suffix = suffix

    candidate_blocks = prefix + [merged_block] + replayed_suffix
    old_total = total_objective(blocks)
    new_total = total_objective(candidate_blocks)
    improvement = old_total - new_total

    diagnostic.update(
        {
            "old_total_objective": old_total,
            "new_total_objective": new_total,
            "improvement": improvement,
            "merged_block_objective": merged_block.objective,
            "candidate_dir": candidate_dir,
            "replayed_downstream_blocks": [block.name for block in replayed_suffix],
        }
    )

    return MergeCandidateResult(
        accepted_candidate_name=merged_name,
        blocks=candidate_blocks,
        total_objective=new_total,
        improvement=improvement,
        merge_run_dir=candidate_dir,
        diagnostic=diagnostic,
    )


def run_selective_exact_cluster_merging(
    *,
    initial_blocks: List[ScheduleBlock],
    buses_per_depot: List[int],
    output_root: str,
    remaining_time_callback=None,
) -> Tuple[List[ScheduleBlock], dict]:
    """Run best-improvement selective adjacent exact cluster merging."""
    ensure_dirs(output_root)

    current_blocks = list(initial_blocks)
    initial_objective = total_objective(current_blocks)
    current_objective = initial_objective
    original_cluster_count = total_original_cluster_count(initial_blocks)

    diagnostics: List[dict] = []
    accepted_merges: List[dict] = []

    print("\n" + "=" * 80)
    print("Running selective exact cluster merging")
    print("=" * 80)
    print(f"Initial RH objective: {initial_objective:.2f}")
    print(f"Max trips per merged window: {SELECTIVE_MERGE_MAX_TRIPS_PER_WINDOW}")
    print(f"Forbid full-instance merge: {SELECTIVE_MERGE_FORBID_FULL_INSTANCE}")
    print(f"Replay downstream after candidate merge: {SELECTIVE_MERGE_REPLAY_DOWNSTREAM}")

    for pass_idx in range(1, SELECTIVE_MERGE_MAX_PASSES + 1):
        print(f"\nSelective merge pass {pass_idx}/{SELECTIVE_MERGE_MAX_PASSES}")
        best_candidate: Optional[MergeCandidateResult] = None
        candidate_counter = 0

        if len(current_blocks) < 2:
            print("  only one block remains; no adjacent merge is possible")
            break

        for merge_index in range(len(current_blocks) - 1):
            remaining = _remaining_time_or_none(remaining_time_callback)
            if remaining is not None and remaining <= 0.0:
                raise TimeoutError(
                    "Rolling-horizon whole-process time limit reached during selective merging."
                )

            candidate_counter += 1
            left = current_blocks[merge_index]
            right = current_blocks[merge_index + 1]

            print(
                f"  testing merge {merge_index}-{merge_index + 1}: "
                f"{left.name} + {right.name} "
                f"({left.n_trips + right.n_trips} trips)"
            )

            try:
                result = build_candidate_merge(
                    blocks=current_blocks,
                    merge_index=merge_index,
                    buses_per_depot=buses_per_depot,
                    original_cluster_count=original_cluster_count,
                    pass_idx=pass_idx,
                    candidate_idx=candidate_counter,
                    output_root=output_root,
                    remaining_time_callback=remaining_time_callback,
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "pass": pass_idx,
                        "candidate": candidate_counter,
                        "merge_index": merge_index,
                        "left_block": left.name,
                        "right_block": right.name,
                        "skipped": True,
                        "skip_reason": f"exception: {exc}",
                    }
                )
                print(f"    skipped due to exception: {exc}")
                continue

            if result is None:
                # Recreate a concise diagnostic for skipped candidates.
                skip_reason = "trip cap or full-instance rule"
                if left.n_trips + right.n_trips > SELECTIVE_MERGE_MAX_TRIPS_PER_WINDOW:
                    skip_reason = (
                        f"{left.n_trips + right.n_trips} trips exceeds cap "
                        f"{SELECTIVE_MERGE_MAX_TRIPS_PER_WINDOW}"
                    )
                elif (
                    SELECTIVE_MERGE_FORBID_FULL_INSTANCE
                    and len(set(left.cluster_names + right.cluster_names)) >= original_cluster_count
                ):
                    skip_reason = "would merge all original clusters"
                diagnostics.append(
                    {
                        "pass": pass_idx,
                        "candidate": candidate_counter,
                        "merge_index": merge_index,
                        "left_block": left.name,
                        "right_block": right.name,
                        "skipped": True,
                        "skip_reason": skip_reason,
                    }
                )
                print(f"    skipped: {skip_reason}")
                continue

            diagnostics.append(result.diagnostic)
            print(
                f"    candidate objective: {result.total_objective:.2f} | "
                f"improvement: {result.improvement:.2f}"
            )

            if result.improvement > SELECTIVE_MERGE_ACCEPTANCE_EPS:
                if best_candidate is None or result.improvement > best_candidate.improvement:
                    best_candidate = result
                    if SELECTIVE_MERGE_FIRST_IMPROVEMENT:
                        break

        if best_candidate is None:
            print("  no improving adjacent merge found; stopping")
            break

        print(
            f"  accepted merge {best_candidate.accepted_candidate_name}: "
            f"{current_objective:.2f} -> {best_candidate.total_objective:.2f}"
        )
        accepted_merges.append(best_candidate.diagnostic)
        current_blocks = best_candidate.blocks
        current_objective = best_candidate.total_objective

    summary = {
        "applied": True,
        "mode": "selective_exact_cluster_merging",
        "initial_rolling_horizon_objective": initial_objective,
        "final_objective": current_objective,
        "improvement": initial_objective - current_objective,
        "accepted": current_objective < initial_objective - SELECTIVE_MERGE_ACCEPTANCE_EPS,
        "accepted_merges": accepted_merges,
        "diagnostics": diagnostics,
        "final_blocks": [block_to_summary(block) for block in current_blocks],
    }

    summary_path = os.path.join(output_root, "selective_exact_merge_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Selective exact merge summary saved to '{summary_path}'")

    report_path = os.path.join(output_root, "selective_exact_merge_report.txt")
    write_blocks_report(current_blocks, report_path, summary)
    print(f"Selective exact merge report saved to '{report_path}'")

    return current_blocks, summary


def block_to_summary(block: ScheduleBlock) -> dict:
    return {
        "name": block.name,
        "cluster_names": block.cluster_names,
        "trips_txt": block.trips_txt,
        "n_trips": block.n_trips,
        "span": list(block.span),
        "objective": block.objective,
        "method": block.method,
        "run_dir": block.run_dir,
        "end_states": json_safe_end_states(block.end_states),
        "schedules": block.schedules,
    }


def write_blocks_report(blocks: List[ScheduleBlock], path: str, summary: Optional[dict] = None) -> None:
    lines: List[str] = []
    lines.append("SELECTIVE EXACT CLUSTER MERGING REPORT")
    lines.append("=" * 80)
    if summary:
        lines.append(f"Initial RH objective: {summary.get('initial_rolling_horizon_objective'):.2f}")
        lines.append(f"Final objective:      {summary.get('final_objective'):.2f}")
        lines.append(f"Improvement:          {summary.get('improvement'):.2f}")
        lines.append(f"Accepted merges:      {len(summary.get('accepted_merges', []))}")
    lines.append("")
    lines.append("Final physical blocks:")

    for idx, block in enumerate(blocks):
        lines.append("")
        lines.append(
            f"Block {idx}: {block.name} | clusters={block.cluster_names} | "
            f"trips={block.n_trips} | obj={block.objective:.2f} | method={block.method}"
        )
        for vehicle in sorted(block.schedules):
            route = block.schedules.get(vehicle, [])
            if route:
                lines.append(f"  {vehicle}: {' -> '.join(route)}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==================== FINAL OUTPUT MATERIALISATION ====================


def _copytree_overwrite(src: str, dst: str) -> None:
    """Copy a solved block directory into the final orchestrated_runs layout."""
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_final_solution_like_original_orchestrator(
    *,
    final_blocks: List[ScheduleBlock],
    buses_per_depot: List[int],
    final_total_objective: float,
    output_root: str,
    bus_end_states_root: str,
    report_summary: Optional[dict] = None,
) -> List[dict]:
    """
    Replace output/orchestrated_runs with only the final accepted physical blocks.

    The produced layout intentionally matches the original orchestrator:

        output/orchestrated_runs/
            batch_00_cluster_X_<timestamp>/
            batch_01_cluster_Y_<timestamp>/
            master_log.json

        output/bus_end_states/
            bus_end_states_<timestamp>.json

    This is the layout expected by downstream visualisation scripts that read the
    final solution from orchestrated_runs rather than from diagnostic merge folders.
    """
    reset_previous_runs(output_root, bus_end_states_root)
    ensure_dirs(output_root, bus_end_states_root)

    master_runs: List[dict] = []

    for idx, block in enumerate(final_blocks):
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        folder_cluster_name = _slug(block.name, 40)
        final_run_dir = os.path.join(
            output_root,
            f"batch_{idx:02d}_{folder_cluster_name}_{ts}",
        )

        _copytree_overwrite(block.run_dir, final_run_dir)

        # Match the first orchestrator convention: bus end-state JSON files are
        # collected in output/bus_end_states, not hidden inside each run folder.
        end_states_path = os.path.join(
            bus_end_states_root,
            f"bus_end_states_{ts}.json",
        )
        with open(end_states_path, "w", encoding="utf-8") as f:
            json.dump(json_safe_end_states(block.end_states), f, indent=4)

        master_runs.append(
            {
                "batch": idx,
                "cluster": block.name,
                "span": [float(block.span[0]), float(block.span[1])],
            }
        )

    master_payload = {
        "fleet": buses_per_depot,
        "total_objective": final_total_objective,
        "runs": master_runs,
    }

    with open(os.path.join(output_root, "master_log.json"), "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    write_blocks_report(
        final_blocks,
        os.path.join(output_root, "final_schedule_report.txt"),
        report_summary or {
            "initial_rolling_horizon_objective": final_total_objective,
            "final_objective": final_total_objective,
            "improvement": 0.0,
            "accepted_merges": [],
        },
    )

    return master_runs


# ==================== ORCHESTRATOR ====================

def run_stateful_rolling_horizon(time_limit_sec: float = None) -> None:
    if time_limit_sec is None:
        time_limit_sec = ROLLING_HORIZON_TIME_LIMIT_SEC

    rh_start_time = time.perf_counter()
    deadline = None
    if time_limit_sec is not None:
        deadline = rh_start_time + float(time_limit_sec)

    def remaining_time_sec() -> Optional[float]:
        if deadline is None:
            return None
        return max(0.0, deadline - time.perf_counter())

    def check_time_limit(stage: str) -> None:
        remaining = remaining_time_sec()
        if remaining is not None and remaining <= 0.0:
            raise TimeoutError(
                f"Rolling-horizon whole-process time limit reached during {stage}."
            )
    
    buses_per_depot = STARTING_BUSES_PER_DEPOT.copy()
    depot_idx = 0
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

        reset_previous_runs(BUS_END_STATES_DIR, ORCHESTRATED_OUTPUT_ROOT, WORKING_OUTPUT_ROOT)
        ensure_dirs(ORCHESTRATED_OUTPUT_ROOT, BUS_END_STATES_DIR, WORKING_OUTPUT_ROOT)

        clusters = discover_clusters(CLUSTERS_ROOT)
        if not clusters:
            raise RuntimeError("No clusters found")

        print_cluster_summary(clusters)
        batches = build_batches(clusters)

        avail, soc, states = initialize_fleet(buses_per_depot)
        master_log = []
        rolling_blocks: List[ScheduleBlock] = []
        infeasible = False

        try:
            for b_idx, batch in enumerate(batches):
                print(f"\n=== Batch {b_idx:02d} | {[c.name for c in batch]} ===")
                for c in batch:

                    check_time_limit("rolling-horizon block solve")

                    block = solve_exact_block(
                        block_name=c.name,
                        cluster_names=[c.name],
                        trips_txt=c.trips_txt,
                        n_trips=c.n_trips,
                        span=(c.start, c.end),
                        buses_per_depot=buses_per_depot,
                        pre_avail=avail,
                        pre_soc=soc,
                        pre_states=states,
                        output_root=WORKING_OUTPUT_ROOT,
                        method="rolling_horizon_exact",
                        require_optimal=REQUIRE_OPTIMAL_BLOCK_SOLVES,
                        time_limit_sec=remaining_time_sec(),
                    )

                    rolling_blocks.append(block)
                    avail = block.post_avail
                    soc = block.post_soc
                    states = block.post_states

                    print("\nSchedules:")
                    print(block.schedules)
                    print_fleet_table(buses_per_depot, avail, soc, states)

                    master_log.append(
                        {
                            "batch": b_idx,
                            "cluster": c.name,
                            "span": [c.start, c.end],
                            "method": "rolling_horizon_exact",
                            "objective": block.objective,
                            "run_dir": block.run_dir,
                        }
                    )

        except TimeoutError:
            raise

        except Exception as exc:
            print("\n!!! INFEASIBILITY OR SOLVER FAILURE DETECTED !!!")
            print(str(exc))
            infeasible = True

        if not infeasible:
            rolling_horizon_total_objective = total_objective(rolling_blocks)
            final_blocks = rolling_blocks
            final_total_objective = rolling_horizon_total_objective
            selective_merge_summary = None

            if APPLY_SELECTIVE_EXACT_CLUSTER_MERGING:
                merge_root = os.path.join(
                    WORKING_OUTPUT_ROOT,
                    f"selmerge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                ensure_dirs(merge_root)

                try:
                    candidate_blocks, selective_merge_summary = run_selective_exact_cluster_merging(
                        initial_blocks=rolling_blocks,
                        buses_per_depot=buses_per_depot,
                        output_root=merge_root,
                        remaining_time_callback=(
                            remaining_time_sec
                            if deadline is not None
                            else None
                        ),
                    )
                    candidate_total = total_objective(candidate_blocks)

                    accept_candidate = True
                    if ACCEPT_FINAL_MERGED_SOLUTION_ONLY_IF_IMPROVES_RH:
                        accept_candidate = (
                            candidate_total < rolling_horizon_total_objective - SELECTIVE_MERGE_ACCEPTANCE_EPS
                        )

                    if accept_candidate:
                        final_blocks = candidate_blocks
                        final_total_objective = candidate_total
                        print(
                            "Final selective exact merged solution accepted | "
                            f"RH={rolling_horizon_total_objective:.2f} | "
                            f"final={final_total_objective:.2f}"
                        )
                    else:
                        print(
                            "Final selective exact merged solution not accepted | "
                            f"RH={rolling_horizon_total_objective:.2f} | "
                            f"candidate={candidate_total:.2f}"
                        )
                except Exception as exc:
                    print("Selective exact cluster merging failed or was skipped.")
                    print(str(exc))
                    selective_merge_summary = {
                        "applied": True,
                        "accepted": False,
                        "mode": "selective_exact_cluster_merging",
                        "error": str(exc),
                    }

            final_states = final_blocks[-1].post_states if final_blocks else states
            final_fleet_size = sum(1 for s in final_states if s == 0)

            final_master_runs = write_final_solution_like_original_orchestrator(
                final_blocks=final_blocks,
                buses_per_depot=buses_per_depot,
                final_total_objective=final_total_objective,
                output_root=ORCHESTRATED_OUTPUT_ROOT,
                bus_end_states_root=BUS_END_STATES_DIR,
                report_summary={
                    "initial_rolling_horizon_objective": rolling_horizon_total_objective,
                    "final_objective": final_total_objective,
                    "improvement": rolling_horizon_total_objective - final_total_objective,
                    "accepted_merges": (
                        selective_merge_summary.get("accepted_merges", [])
                        if isinstance(selective_merge_summary, dict)
                        else []
                    ),
                },
            )

            diagnostic_payload = {
                "fleet": buses_per_depot,
                "rolling_horizon_total_objective": rolling_horizon_total_objective,
                "final_total_objective": final_total_objective,
                "final_fleet_size": final_fleet_size,
                "original_rolling_horizon_runs": master_log,
                "final_runs_written_to_orchestrated_runs": final_master_runs,
                "rolling_horizon_blocks": [block_to_summary(block) for block in rolling_blocks],
                "final_blocks": [block_to_summary(block) for block in final_blocks],
                "selective_exact_cluster_merging": selective_merge_summary,
            }

            ensure_dirs(WORKING_OUTPUT_ROOT)
            with open(os.path.join(WORKING_OUTPUT_ROOT, "diagnostic_master_log.json"), "w", encoding="utf-8") as f:
                json.dump(diagnostic_payload, f, indent=2)

            print("\nRolling-horizon orchestration complete.")
            print(f"Final fleet size: {final_fleet_size}")
            print(f"\nRolling-horizon Total Objective Value: {rolling_horizon_total_objective:.2f}")
            print(f"Final Total Objective Value: {final_total_objective:.2f}")
            print(f"Final solution written in original orchestrator format: {ORCHESTRATED_OUTPUT_ROOT}")
            return

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
    print(f"\nTotal execution time: {elapsed_sec:.2f} seconds ({elapsed_sec / 60:.2f} minutes)")
