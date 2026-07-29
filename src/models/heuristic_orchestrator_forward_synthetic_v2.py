"""
Stateful rolling-horizon orchestrator for synthetic TXT clusters
with selective exact cluster merging.

This is the synthetic-pipeline counterpart of heuristic_orchestrator_forward_oasa_v2.py.
It keeps the original synthetic data flow:
- clusters are folders containing instance.txt files,
- each instance is loaded with solver.load_instance_from_txt(...),
- the synthetic exact solver is called with output_dir, bus_end_states_dir, and run_timestamp.

The main v2 addition is selective exact cluster merging:
1) solve synthetic clusters sequentially with the exact MIP,
2) preserve physical bus end states after each block,
3) test adjacent block merges by building a merged synthetic instance.txt,
4) if a merge changes bus states, replay downstream blocks from the candidate end state,
5) accept only if the full recomputed objective improves.

The code intentionally avoids the old RNI/SA neighborhood machinery. This is not SA.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from gurobipy import GRB

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import eb_md_vsp_tw_solver_with_vi as solver

# ==================== CONFIG ====================

PROJECT_ROOT = os.path.normpath(os.path.join(solver.project_root, ".."))

# Same synthetic root used by the original synthetic orchestrator.
# Expected layout:
#   output/clusters_synthetic/D2_S3_C30_c_trips.txt/
#       cluster_0/instance.txt
#       cluster_1/instance.txt
#       ...
CLUSTERS_ROOT = os.path.join(
    PROJECT_ROOT,
    "output",
    "clusters_synthetic",
    "D2_S3_C30_e_trips.txt",
)

ORCHESTRATED_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "orchestrated_runs")
WORKING_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "orchestrated_runs_work")
BUS_END_STATES_DIR = os.path.join(PROJECT_ROOT, "output", "bus_end_states")

INITIAL_AVAILABILITY_TIME = 0.0

# ==================== SELECTIVE EXACT MERGING CONFIG ====================

APPLY_SELECTIVE_EXACT_CLUSTER_MERGING = True
ACCEPT_FINAL_MERGED_SOLUTION_ONLY_IF_IMPROVES_RH = True

# Only adjacent current blocks are tested: block i + block i+1.
SELECTIVE_MERGE_MAX_PASSES = 2
SELECTIVE_MERGE_MAX_TRIPS_PER_WINDOW = 10
SELECTIVE_MERGE_ACCEPTANCE_EPS = 1e-6
SELECTIVE_MERGE_FIRST_IMPROVEMENT = False

# Prevent the post-improvement from silently becoming the full exact model.
SELECTIVE_MERGE_FORBID_FULL_INSTANCE = True

# Correct physical accounting: after a candidate merge, downstream blocks must be
# re-solved from the candidate's new bus states.
SELECTIVE_MERGE_REPLAY_DOWNSTREAM = True


# ==================== DATA ====================

@dataclass(frozen=True)
class ClusterInfo:
    name: str
    instance_txt: str
    n_trips: int
    start: float = 0.0
    end: float = 0.0


@dataclass
class ScheduleBlock:
    """One physical block in the current rolling-horizon/merged schedule."""

    name: str
    cluster_names: List[str]
    instance_txt: str
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


def reset_previous_runs(*paths: str) -> None:
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
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
    digest = hashlib.md5(f"{method}|{block_name}|{timestamp}".encode("utf-8")).hexdigest()[:6]
    return f"{_method_tag(method)}_{_slug(block_name, 16)}_{digest}_{timestamp}"


def vehicle_ids(depot_ids: List[int], buses_per_depot: List[int]) -> List[str]:
    """Synthetic depots can have IDs like 110, 120, 130, so do not enumerate 1..D."""
    ids: List[str] = []
    for depot_id, n_buses in zip(depot_ids, buses_per_depot):
        for i in range(1, n_buses + 1):
            ids.append(f"D{depot_id}_V{i}")
    return ids


def fleet_from_instance(inst) -> Tuple[List[int], List[int], List[float], List[float], List[int]]:
    """Infer the synthetic fleet ordering directly from the loaded TXT instance."""
    depot_ids = [int(depot.id) for depot in inst.depots]
    buses_per_depot = [int(depot.vehicle_count) for depot in inst.depots]
    total = sum(buses_per_depot)

    initial_soc = list(inst.meta.get("buses_initial_soc", []))
    if len(initial_soc) != total:
        initial_soc = [float(inst.meta["p_max"])] * total

    initial_avail = list(inst.meta.get("buses_availability_times", []))
    if len(initial_avail) != total:
        initial_avail = [INITIAL_AVAILABILITY_TIME] * total

    initial_states = list(inst.meta.get("buses_state", []))
    if len(initial_states) != total:
        initial_states = [1] * total

    return depot_ids, buses_per_depot, initial_avail, initial_soc, initial_states


def update_fleet_from_end_states(
    depot_ids: List[int],
    buses_per_depot: List[int],
    prev_avail: List[float],
    prev_soc: List[float],
    prev_state: List[int],
    end_states: dict,
) -> Tuple[List[float], List[float], List[int]]:
    vids = vehicle_ids(depot_ids, buses_per_depot)
    idx = {v: i for i, v in enumerate(vids)}

    avail = list(prev_avail)
    soc = list(prev_soc)
    state = list(prev_state)

    for vid, st in (end_states or {}).items():
        i = idx.get(vid)
        if i is None:
            continue
        if "arrival_time" in st:
            avail[i] = float(st["arrival_time"])
        if "soc" in st:
            soc[i] = float(st["soc"])
        state[i] = 0  # 0 means this physical bus has now been used.

    return avail, soc, state


def print_fleet_table(
    depot_ids: List[int],
    buses_per_depot: List[int],
    avail: List[float],
    soc: List[float],
    states: List[int],
) -> None:
    vids = vehicle_ids(depot_ids, buses_per_depot)
    header = f"{'Vehicle':<10} {'Depot':<8} {'Avail_T':>8} {'SoC':>8} {'Fresh':>6}"
    print("\nVehicle state before solve:")
    print(header)
    print("-" * len(header))

    for vid, t, s, st in zip(vids, avail, soc, states):
        depot = vid.split("_")[0]
        print(f"{vid:<10} {depot:<8} {t:8.1f} {s:8.1f} {st:6d}")


# ==================== SYNTHETIC INSTANCE HELPERS ====================


def _read_tsv_rows(path: str) -> List[List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter="\t"))


def _write_tsv_rows(path: str, rows: List[List[str]]) -> None:
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def _split_synthetic_instance_rows(path: str) -> Tuple[List[str], List[List[str]], List[List[str]], List[List[str]]]:
    """Return header, depot rows, trip rows, charger rows from a synthetic instance.txt."""
    rows = _read_tsv_rows(path)
    if not rows:
        raise ValueError(f"Empty synthetic instance: {path}")

    header = list(rows[0])
    header_float = [float(x) for x in header]
    k = int(header_float[0])
    t_trips = int(header_float[1])
    f_chargers = int(header_float[2])

    body = rows[1:]
    n_depot_rows = 2 * k
    if len(body) < n_depot_rows + t_trips:
        raise ValueError(f"Malformed synthetic instance: {path}")

    depot_rows = body[:n_depot_rows]
    trip_rows = body[n_depot_rows:n_depot_rows + t_trips]
    charger_start = n_depot_rows + t_trips
    charger_rows = body[charger_start:charger_start + f_chargers]
    return header, depot_rows, trip_rows, charger_rows


def _trip_span_from_rows(trip_rows: List[List[str]]) -> Tuple[float, float]:
    starts: List[float] = []
    ends: List[float] = []
    for row in trip_rows:
        if len(row) < 7:
            continue
        starts.append(float(row[5]))
        ends.append(float(row[6]))
    if not starts:
        return 0.0, 0.0
    return min(starts), max(ends)


def discover_clusters(root: str) -> List[ClusterInfo]:
    clusters: List[ClusterInfo] = []

    for d in sorted(os.listdir(root)):
        cdir = os.path.join(root, d)
        if not os.path.isdir(cdir):
            continue

        inst_path = os.path.join(cdir, "instance.txt")
        if not os.path.isfile(inst_path):
            continue

        _, _, trip_rows, _ = _split_synthetic_instance_rows(inst_path)
        span = _trip_span_from_rows(trip_rows)

        clusters.append(
            ClusterInfo(
                name=d,
                instance_txt=inst_path,
                n_trips=len(trip_rows),
                start=span[0],
                end=span[1],
            )
        )

    if not clusters:
        raise RuntimeError(f"No TXT clusters found in {root}")

    return sorted(clusters, key=lambda c: (c.start, c.name))


def print_cluster_summary(clusters: List[ClusterInfo]) -> None:
    print("\nClusters loaded from TXT instances:")
    header = f"{'Cluster':<12} {'Trips':>6} {'Start':>8} {'End':>8}"
    print(header)
    print("-" * len(header))
    for c in clusters:
        print(f"{c.name:<12} {c.n_trips:>6d} {c.start:>8.1f} {c.end:>8.1f}")


def build_batches(clusters: List[ClusterInfo]) -> List[List[ClusterInfo]]:
    # Preserve the original synthetic orchestrator behaviour: one cluster per batch.
    return [[c] for c in clusters]


def write_merged_instance_file(blocks: List[ScheduleBlock], output_dir: str, label: str) -> str:
    """
    Build a synthetic merged instance.txt by concatenating trip rows.

    The synthetic format is:
      header
      K origin rows
      K destination rows
      T trip rows
      F charger rows

    For a merge, the fleet/depot/charger information is copied from the first block,
    and the trip rows are concatenated from all merged blocks. Duplicate trip IDs are
    ignored defensively, although normal cluster partitions should not contain them.
    """
    ensure_dirs(output_dir)
    merged_path = os.path.join(output_dir, f"{label}_instance.txt")

    first_header, first_depot_rows, _, first_charger_rows = _split_synthetic_instance_rows(blocks[0].instance_txt)

    merged_trip_rows: List[List[str]] = []
    seen_trip_ids = set()
    for block in blocks:
        header, depot_rows, trip_rows, charger_rows = _split_synthetic_instance_rows(block.instance_txt)

        if int(float(header[0])) != int(float(first_header[0])):
            raise ValueError("Cannot merge synthetic blocks with different K values.")
        if int(float(header[2])) != int(float(first_header[2])):
            raise ValueError("Cannot merge synthetic blocks with different F values.")
        if depot_rows != first_depot_rows:
            raise ValueError("Cannot merge synthetic blocks with different depot rows.")
        if charger_rows != first_charger_rows:
            raise ValueError("Cannot merge synthetic blocks with different charger rows.")

        for row in trip_rows:
            trip_id = row[0].strip() if row else ""
            if trip_id and trip_id not in seen_trip_ids:
                seen_trip_ids.add(trip_id)
                merged_trip_rows.append(row)

    merged_header = list(first_header)
    merged_header[1] = str(len(merged_trip_rows))
    merged_rows = [merged_header] + first_depot_rows + merged_trip_rows + first_charger_rows
    _write_tsv_rows(merged_path, merged_rows)
    return merged_path


def json_safe_end_states(end_states: dict) -> dict:
    return json.loads(json.dumps(end_states, default=str))


def _read_json_if_nonempty(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and data else {}
    except Exception:
        return {}


def _solution_key(*parts: str) -> str:
    return "(" + ",".join(str(part) for part in parts) + ")"


def _derive_end_states_from_solution_variables(
    *,
    schedules: Dict[str, List[str]],
    solution_vars_path: str,
) -> dict:
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


def _inject_fleet_state_into_instance(
    inst,
    pre_avail: List[float],
    pre_soc: List[float],
    pre_states: List[int],
) -> None:
    """
    Synthetic load_instance_from_txt reads the latest bus_end_states side file.
    For v2 merge testing and downstream replay we need explicit physical states,
    so override the loaded metadata immediately after loading.
    """
    inst.meta.update(
        {
            "buses_availability_times": list(pre_avail),
            "buses_initial_soc": list(pre_soc),
            "buses_state": list(pre_states),
            "bus_end_states_source": "orchestrator_explicit_state",
        }
    )


# ==================== EXACT SOLVE WRAPPER ====================


def solve_exact_block(
    *,
    block_name: str,
    cluster_names: List[str],
    instance_txt: str,
    n_trips: int,
    span: Tuple[float, float],
    depot_ids: List[int],
    buses_per_depot: List[int],
    pre_avail: List[float],
    pre_soc: List[float],
    pre_states: List[int],
    output_root: str,
    method: str,
    require_optimal: bool = True,
) -> ScheduleBlock:
    """Solve one synthetic physical block exactly and return its state transition."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(output_root, _compact_run_name(method, block_name, ts))
    ensure_dirs(run_dir)

    block_bus_end_states_dir = os.path.join(run_dir, "bus_end_states")
    ensure_dirs(block_bus_end_states_dir)

    expected_end_states_path = os.path.join(
        block_bus_end_states_dir,
        f"bus_end_states_{ts}.json",
    )
    solution_vars_path = os.path.join(run_dir, "solution_variables.json")

    with open(expected_end_states_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

    print(
        f"Solving {block_name} | trips={n_trips} | method={method} | "
        f"avail[min,max]=({min(pre_avail):.1f},{max(pre_avail):.1f}) | "
        f"soc[min,max]=({min(pre_soc):.1f},{max(pre_soc):.1f}) | "
        f"span=({span[0]:.1f},{span[1]:.1f})"
    )
    print_fleet_table(depot_ids, buses_per_depot, pre_avail, pre_soc, pre_states)

    inst = solver.load_instance_from_txt(instance_txt)
    _inject_fleet_state_into_instance(inst, pre_avail, pre_soc, pre_states)

    result = solver.solve_md_vsp_tw_from_instance(
        inst,
        output_dir=run_dir,
        bus_end_states_dir=block_bus_end_states_dir,
        run_timestamp=ts,
    )

    if result is None or len(result) < 7:
        raise RuntimeError(f"Solver returned invalid result for {block_name}")

    _, schedules, _, stats, returned_end_states, sol_status, obj_fun_value = result

    if sol_status == GRB.INFEASIBLE:
        raise RuntimeError(f"Solver infeasible for {block_name}")
    if require_optimal and sol_status != GRB.OPTIMAL:
        raise RuntimeError(f"Solver did not prove optimality for {block_name}; status={sol_status}")

    end_states = _normalise_end_states(
        returned_end_states=returned_end_states,
        schedules=schedules,
        expected_end_states_path=expected_end_states_path,
        solution_vars_path=solution_vars_path,
    )

    with open(expected_end_states_path, "w", encoding="utf-8") as f:
        json.dump(json_safe_end_states(end_states), f, indent=4)

    post_avail, post_soc, post_states = update_fleet_from_end_states(
        depot_ids,
        buses_per_depot,
        pre_avail,
        pre_soc,
        pre_states,
        end_states,
    )

    return ScheduleBlock(
        name=block_name,
        cluster_names=list(cluster_names),
        instance_txt=instance_txt,
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


# run single cluster is used from another file
def run_single_cluster(cluster_instance_path: str):
    """Runs the solver for exactly one synthetic cluster and returns solver stats."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    inst = solver.load_instance_from_txt(cluster_instance_path)

    out_dir = os.path.join(ORCHESTRATED_OUTPUT_ROOT, f"single_cluster_{ts}")
    ensure_dirs(out_dir, BUS_END_STATES_DIR)

    result = solver.solve_md_vsp_tw_from_instance(
        inst,
        output_dir=out_dir,
        bus_end_states_dir=BUS_END_STATES_DIR,
        run_timestamp=ts,
    )

    if result is None or len(result) < 7:
        raise RuntimeError("Solver returned invalid result")

    _, _, _, stats, _, sol_status, _ = result
    if sol_status == GRB.INFEASIBLE:
        raise RuntimeError(f"Cluster {cluster_instance_path} infeasible")

    return stats


# ==================== SELECTIVE EXACT CLUSTER MERGING ====================


def total_objective(blocks: List[ScheduleBlock]) -> float:
    return sum(float(block.objective) for block in blocks)


def total_original_cluster_count(blocks: List[ScheduleBlock]) -> int:
    names = set()
    for block in blocks:
        names.update(block.cluster_names)
    return len(names)


def replay_downstream_blocks(
    *,
    downstream_blocks: List[ScheduleBlock],
    start_avail: List[float],
    start_soc: List[float],
    start_states: List[int],
    depot_ids: List[int],
    buses_per_depot: List[int],
    output_root: str,
    method_label: str,
) -> List[ScheduleBlock]:
    """Re-solve downstream blocks after a candidate merge changes bus states."""
    replayed: List[ScheduleBlock] = []
    avail = list(start_avail)
    soc = list(start_soc)
    states = list(start_states)

    for old_block in downstream_blocks:
        new_block = solve_exact_block(
            block_name=old_block.name,
            cluster_names=old_block.cluster_names,
            instance_txt=old_block.instance_txt,
            n_trips=old_block.n_trips,
            span=old_block.span,
            depot_ids=depot_ids,
            buses_per_depot=buses_per_depot,
            pre_avail=avail,
            pre_soc=soc,
            pre_states=states,
            output_root=output_root,
            method=method_label,
            require_optimal=True,
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
    depot_ids: List[int],
    buses_per_depot: List[int],
    original_cluster_count: int,
    pass_idx: int,
    candidate_idx: int,
    output_root: str,
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

    merged_instance_txt = write_merged_instance_file(
        [left, right],
        candidate_dir,
        f"merged_{merge_index}_{merge_index + 1}",
    )

    merged_block = solve_exact_block(
        block_name=merged_name,
        cluster_names=merged_cluster_names,
        instance_txt=merged_instance_txt,
        n_trips=merged_n_trips,
        span=merged_span,
        depot_ids=depot_ids,
        buses_per_depot=buses_per_depot,
        pre_avail=left.pre_avail,
        pre_soc=left.pre_soc,
        pre_states=left.pre_states,
        output_root=candidate_dir,
        method="exact_merged_window",
        require_optimal=True,
    )

    prefix = blocks[:merge_index]
    suffix = blocks[merge_index + 2:]

    if SELECTIVE_MERGE_REPLAY_DOWNSTREAM and suffix:
        replayed_suffix = replay_downstream_blocks(
            downstream_blocks=suffix,
            start_avail=merged_block.post_avail,
            start_soc=merged_block.post_soc,
            start_states=merged_block.post_states,
            depot_ids=depot_ids,
            buses_per_depot=buses_per_depot,
            output_root=candidate_dir,
            method_label="downstream_replay_after_merge",
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
    depot_ids: List[int],
    buses_per_depot: List[int],
    output_root: str,
) -> Tuple[List[ScheduleBlock], dict]:
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
                    depot_ids=depot_ids,
                    buses_per_depot=buses_per_depot,
                    original_cluster_count=original_cluster_count,
                    pass_idx=pass_idx,
                    candidate_idx=candidate_counter,
                    output_root=output_root,
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
        "mode": "selective_exact_cluster_merging_synthetic",
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
        "instance_txt": block.instance_txt,
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
    lines.append("SELECTIVE EXACT CLUSTER MERGING REPORT - SYNTHETIC")
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
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_final_solution_like_original_orchestrator(
    *,
    final_blocks: List[ScheduleBlock],
    depot_ids: List[int],
    buses_per_depot: List[int],
    final_total_objective: float,
    output_root: str,
    bus_end_states_root: str,
    report_summary: Optional[dict] = None,
) -> List[dict]:
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
                "n_trips": block.n_trips,
                "span": [float(block.span[0]), float(block.span[1])],
                "method": block.method,
                "objective": block.objective,
                "run_dir": final_run_dir,
            }
        )

    master_payload = {
        "fleet": {
            "depot_ids": depot_ids,
            "buses_per_depot": buses_per_depot,
            "vehicle_ids": vehicle_ids(depot_ids, buses_per_depot),
        },
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


def run_stateful_rolling_horizon() -> None:
    clusters = discover_clusters(CLUSTERS_ROOT)
    if not clusters:
        raise RuntimeError("No clusters found")

    # Load first cluster to infer the synthetic fleet structure and initial SoC.
    first_inst = solver.load_instance_from_txt(clusters[0].instance_txt)
    depot_ids, buses_per_depot, avail, soc, states = fleet_from_instance(first_inst)

    print("\n" + "=" * 80)
    print(f"Starting rolling horizon with synthetic fleet = {vehicle_ids(depot_ids, buses_per_depot)}")
    print("=" * 80)

    reset_previous_runs(BUS_END_STATES_DIR, ORCHESTRATED_OUTPUT_ROOT, WORKING_OUTPUT_ROOT)
    ensure_dirs(ORCHESTRATED_OUTPUT_ROOT, BUS_END_STATES_DIR, WORKING_OUTPUT_ROOT)

    print_cluster_summary(clusters)
    batches = build_batches(clusters)

    master_log = []
    rolling_blocks: List[ScheduleBlock] = []
    infeasible = False

    try:
        for b_idx, batch in enumerate(batches):
            print(f"\n=== Batch {b_idx:02d} | {[c.name for c in batch]} ===")

            for c in batch:
                block = solve_exact_block(
                    block_name=c.name,
                    cluster_names=[c.name],
                    instance_txt=c.instance_txt,
                    n_trips=c.n_trips,
                    span=(c.start, c.end),
                    depot_ids=depot_ids,
                    buses_per_depot=buses_per_depot,
                    pre_avail=avail,
                    pre_soc=soc,
                    pre_states=states,
                    output_root=WORKING_OUTPUT_ROOT,
                    method="rolling_horizon_exact",
                    require_optimal=True,
                )

                rolling_blocks.append(block)
                avail = block.post_avail
                soc = block.post_soc
                states = block.post_states

                print("\nSchedules:")
                print(block.schedules)
                print_fleet_table(depot_ids, buses_per_depot, avail, soc, states)

                master_log.append(
                    {
                        "batch": b_idx,
                        "cluster": c.name,
                        "n_trips": c.n_trips,
                        "span": [c.start, c.end],
                        "method": "rolling_horizon_exact",
                        "objective": block.objective,
                        "run_dir": block.run_dir,
                    }
                )

    except Exception as exc:
        print("\n!!! INFEASIBILITY OR SOLVER FAILURE DETECTED !!!")
        print(str(exc))
        infeasible = True

    if infeasible:
        raise RuntimeError("Synthetic rolling-horizon orchestration failed.")

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
                depot_ids=depot_ids,
                buses_per_depot=buses_per_depot,
                output_root=merge_root,
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
                "mode": "selective_exact_cluster_merging_synthetic",
                "error": str(exc),
            }

    final_states = final_blocks[-1].post_states if final_blocks else states
    final_fleet_size = sum(1 for s in final_states if s == 0)

    final_master_runs = write_final_solution_like_original_orchestrator(
        final_blocks=final_blocks,
        depot_ids=depot_ids,
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
        "fleet": {
            "depot_ids": depot_ids,
            "buses_per_depot": buses_per_depot,
            "vehicle_ids": vehicle_ids(depot_ids, buses_per_depot),
        },
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


if __name__ == "__main__":
    start_time = time.perf_counter()
    run_stateful_rolling_horizon()
    end_time = time.perf_counter()
    elapsed_sec = end_time - start_time
    print(f"\nTotal execution time: {elapsed_sec:.2f} seconds ({elapsed_sec / 60:.2f} minutes)")
