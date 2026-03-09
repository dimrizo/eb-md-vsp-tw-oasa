#!/usr/bin/env python3
"""parameter_search_dbscan_v3.py

Parameter search for your DBSCAN + chain-feasible distance using the *module's*
own clustering pipeline.

What this version changes vs v2:
  • Defaults to DBSCAN_clustering_final_v4.py
  • Calls `mod.cluster_trips(trips)` each trial so results match the module logic
    (including route-aware penalty and oversize split-by-time, if enabled there).
  • Metrics are computed from the returned cluster dictionary, not raw DBSCAN labels.

Why? Because humans (and papers) tend to evaluate the thing they actually run.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import contextlib
import io



# -----------------------------
# Module loading
# -----------------------------

def load_module(path: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module not found: {path}")

    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# -----------------------------
# Data + metrics
# -----------------------------

@dataclass
class Params:
    SPATIAL_EPS_METERS: float
    TEMPORAL_EPS_MIN: float
    DBSCAN_EPS: float
    MIN_TRIPS_PER_CLUSTER: int
    MAX_TRIPS_PER_CLUSTER: int
    SPACE_WEIGHT: float
    TIME_WEIGHT: float
    REQUIRE_NONNEGATIVE_GAP: bool
    START_TIME_WINDOW_MIN: float


@dataclass
class Metrics:
    n_trips: int
    n_clusters: int
    n_singletons: int
    n_non_singletons: int
    mean_non_singleton_size: float
    median_non_singleton_size: float
    std_non_singleton_size: float
    pct_non_singletons_in_15_25: float
    max_cluster_size: int
    min_non_singleton_size: int


def build_trips_once(
    mod,
    gtfs_folder_path: str,
    route_ids: List[int],
    day: str,
    first_trip_index: int,
    max_trips_per_route_per_direction: int,
):
    """Load trips once using the module's helper functions."""

    trips_txt_path = os.path.join(gtfs_folder_path, "trips.txt")
    if not os.path.exists(trips_txt_path):
        raise FileNotFoundError(f"trips.txt not found: {trips_txt_path}")

    trips_txt_df = pd.read_csv(trips_txt_path)
    trips_txt_df["trip_id"] = trips_txt_df["trip_id"].astype(str)
    trips_txt_df.set_index("trip_id", inplace=True, drop=False)

    trip_id_to_start_time = mod.load_trip_start_times_from_stop_times(gtfs_folder_path)
    trip_id_to_end_time = mod.load_trip_end_times_from_stop_times(gtfs_folder_path)

    trips = mod.build_trips_for_routes(
        route_ids=route_ids,
        day=day,
        gtfs_folder_path=gtfs_folder_path,
        trips_txt_df=trips_txt_df,
        trip_id_to_start_time=trip_id_to_start_time,
        trip_id_to_end_time=trip_id_to_end_time,
    )

    trips = mod.prune_trips_for_clustering(
        trips,
        max_trips_per_route_per_direction=max_trips_per_route_per_direction,
        first_trip_index=first_trip_index,
    )

    return trips


def compute_metrics_from_clusters(clusters: Dict[int, List]) -> Metrics:
    sizes = [len(v) for v in clusters.values()]
    n_trips = int(sum(sizes))
    n_clusters = int(len(sizes))

    n_singletons = int(sum(1 for s in sizes if s == 1))
    non_singletons = [s for s in sizes if s >= 2]
    n_non_singletons = int(len(non_singletons))

    if n_non_singletons:
        mean_sz = float(np.mean(non_singletons))
        median_sz = float(np.median(non_singletons))
        std_sz = float(np.std(non_singletons))
        pct_15_25 = float(np.mean([(15 <= s <= 25) for s in non_singletons]))
        min_non = int(min(non_singletons))
        max_sz = int(max(sizes))
    else:
        mean_sz = median_sz = std_sz = 0.0
        pct_15_25 = 0.0
        min_non = 0
        max_sz = int(max(sizes)) if sizes else 0

    return Metrics(
        n_trips=n_trips,
        n_clusters=n_clusters,
        n_singletons=n_singletons,
        n_non_singletons=n_non_singletons,
        mean_non_singleton_size=mean_sz,
        median_non_singleton_size=median_sz,
        std_non_singleton_size=std_sz,
        pct_non_singletons_in_15_25=pct_15_25,
        max_cluster_size=max_sz,
        min_non_singleton_size=min_non,
    )


def score_config(p: Params, m: Metrics) -> float:
    """Lower is better.

    Same scoring idea as v2: steer toward fewer singletons and median size ~20,
    and penalize oversize spill (after module's own post-processing).
    """

    target_singletons = 8
    singleton_pen = abs(m.n_singletons - target_singletons) / max(1.0, target_singletons)

    target_med = 20.0
    size_pen = abs(m.median_non_singleton_size - target_med) / max(1.0, target_med)

    overflow = max(0, m.max_cluster_size - int(p.MAX_TRIPS_PER_CLUSTER))
    overflow_pen = overflow / max(1.0, float(p.MAX_TRIPS_PER_CLUSTER))

    return (0.35 * singleton_pen) + (0.5 * size_pen) + (0.15 * overflow_pen)


# -----------------------------
# Search space
# -----------------------------

def sample_params(rng: random.Random) -> Params:
    spatial = rng.uniform(50.0, 600.0)
    temporal = rng.uniform(10.0, 70.0)

    db_eps = rng.uniform(0.45, 1.05)
    min_samples = rng.choice([3, 4, 5])

    max_cluster = 20

    space_w = rng.uniform(0.4, 1.2)
    time_w = rng.uniform(0.4, 1.2)

    nonneg = rng.choice([True, False])

    start_win = rng.choice([240.0, 300.0, 360.0, 480.0, 540.0])

    return Params(
        SPATIAL_EPS_METERS=spatial,
        TEMPORAL_EPS_MIN=temporal,
        DBSCAN_EPS=db_eps,
        MIN_TRIPS_PER_CLUSTER=int(min_samples),
        MAX_TRIPS_PER_CLUSTER=int(max_cluster),
        SPACE_WEIGHT=space_w,
        TIME_WEIGHT=time_w,
        REQUIRE_NONNEGATIVE_GAP=bool(nonneg),
        START_TIME_WINDOW_MIN=float(start_win),
    )


_BOUNDS = {
    "SPATIAL_EPS_METERS": (50.0, 600.0),
    "TEMPORAL_EPS_MIN": (10.0, 70.0),
    "DBSCAN_EPS": (0.45, 1.05),
    "SPACE_WEIGHT": (0.4, 1.2),
    "TIME_WEIGHT": (0.4, 1.2),
}

_CHOICES = {
    "MIN_TRIPS_PER_CLUSTER": [3, 4, 5],
    "START_TIME_WINDOW_MIN": [240.0, 300.0, 360.0, 480.0, 540.0],
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _mutate_choice_near(value, choices: List, rng: random.Random) -> Any:
    if value not in choices:
        idx = int(np.argmin([abs(float(c) - float(value)) for c in choices]))
    else:
        idx = choices.index(value)
    step = rng.choice([-1, 0, 0, 0, 1])
    idx2 = max(0, min(len(choices) - 1, idx + step))
    return choices[idx2]


def mutate_params(elite: Params, rng: random.Random, scale: float) -> Params:
    def mut_cont(name: str, base: float) -> float:
        lo, hi = _BOUNDS[name]
        span = hi - lo
        x = rng.gauss(base, scale * span)
        return float(_clamp(x, lo, hi))

    spatial = mut_cont("SPATIAL_EPS_METERS", float(elite.SPATIAL_EPS_METERS))
    temporal = mut_cont("TEMPORAL_EPS_MIN", float(elite.TEMPORAL_EPS_MIN))
    db_eps = mut_cont("DBSCAN_EPS", float(elite.DBSCAN_EPS))
    space_w = mut_cont("SPACE_WEIGHT", float(elite.SPACE_WEIGHT))
    time_w = mut_cont("TIME_WEIGHT", float(elite.TIME_WEIGHT))

    min_samples = elite.MIN_TRIPS_PER_CLUSTER
    start_win = elite.START_TIME_WINDOW_MIN
    if rng.random() < 0.35:
        min_samples = _mutate_choice_near(min_samples, _CHOICES["MIN_TRIPS_PER_CLUSTER"], rng)
    if rng.random() < 0.35:
        start_win = _mutate_choice_near(start_win, _CHOICES["START_TIME_WINDOW_MIN"], rng)

    nonneg = elite.REQUIRE_NONNEGATIVE_GAP
    if rng.random() < 0.15:
        nonneg = not bool(nonneg)

    return Params(
        SPATIAL_EPS_METERS=spatial,
        TEMPORAL_EPS_MIN=temporal,
        DBSCAN_EPS=db_eps,
        MIN_TRIPS_PER_CLUSTER=int(min_samples),
        MAX_TRIPS_PER_CLUSTER=int(elite.MAX_TRIPS_PER_CLUSTER),
        SPACE_WEIGHT=space_w,
        TIME_WEIGHT=time_w,
        REQUIRE_NONNEGATIVE_GAP=bool(nonneg),
        START_TIME_WINDOW_MIN=float(start_win),
    )

def apply_params(mod, p: Params) -> None:
    mod.SPATIAL_EPS_METERS = float(p.SPATIAL_EPS_METERS)
    mod.TEMPORAL_EPS_MIN = float(p.TEMPORAL_EPS_MIN)
    mod.DBSCAN_EPS = float(p.DBSCAN_EPS)
    mod.MIN_TRIPS_PER_CLUSTER = int(p.MIN_TRIPS_PER_CLUSTER)
    mod.MAX_TRIPS_PER_CLUSTER = int(p.MAX_TRIPS_PER_CLUSTER)
    mod.SPACE_WEIGHT = float(p.SPACE_WEIGHT)
    mod.TIME_WEIGHT = float(p.TIME_WEIGHT)
    mod.REQUIRE_NONNEGATIVE_GAP = bool(p.REQUIRE_NONNEGATIVE_GAP)
    mod.START_TIME_WINDOW_MIN = float(p.START_TIME_WINDOW_MIN)

def run_once(mod, trips, p: Params) -> Tuple[Metrics, float]:
    apply_params(mod, p)

    # Silence chatty prints inside the clustering module (DBSCAN label counts, etc.)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        clusters = mod.cluster_trips(trips)

    m = compute_metrics_from_clusters(clusters)
    s = score_config(p, m)
    return m, s


# -----------------------------
# CLI entrypoint
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--module",
        default="src/pre_processing/DBSCAN_clustering_final_v4.py",
        help="Path to DBSCAN_clustering_final_v4.py",
    )
    ap.add_argument("--gtfs", default=None, help="Path to GTFS folder containing trips.txt and stop_times.txt")
    ap.add_argument("--routes", nargs="+", type=int, default=[1033, 1034, 874, 871], help="Route IDs")
    ap.add_argument("--day", default="monday", help="Day string used by extract_gtfs_trips_data")
    ap.add_argument("--iters", type=int, default=500, help="Number of trials per round (or total trials if --rounds=1)")
    ap.add_argument("--rounds", type=int, default=5, help="Number of rounds (learning iterations)")
    ap.add_argument("--elite", type=int, default=10, help="How many best configs from previous round to sample around")
    ap.add_argument("--explore_frac", type=float, default=0.2, help="Fraction of samples per round that stay global-random (0..1)")
    ap.add_argument("--init_scale", type=float, default=0.45, help="Initial mutation scale relative to param ranges (0..1)")
    ap.add_argument("--scale_decay", type=float, default=0.8, help="Multiply mutation scale by this each round (0..1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--topk", type=int, default=10, help="How many best configs to print")
    ap.add_argument("--first_trip_index", type=int, default=None)
    ap.add_argument("--max_trips_per_route_per_direction", type=int, default=None)
    ap.add_argument("--out_csv", default="parameter_search_results.csv", help="CSV output file")
    args = ap.parse_args()

    mod = load_module(args.module)

    if args.gtfs is None:
        gtfs_folder_path = os.path.join(mod.project_root, "..", "input", "gtfs", "oasa_december_2025")
    else:
        gtfs_folder_path = args.gtfs

    first_trip_index = args.first_trip_index if args.first_trip_index is not None else getattr(mod, "FIRST_TRIP_INDEX", 0)
    max_trips_per_route_per_direction = (
        args.max_trips_per_route_per_direction
        if args.max_trips_per_route_per_direction is not None
        else getattr(mod, "MAX_TRIPS_PER_ROUTE_PER_DIRECTION", 26)
    )

    print("Loading trips once...")
    trips = build_trips_once(
        mod,
        gtfs_folder_path=gtfs_folder_path,
        route_ids=args.routes,
        day=args.day,
        first_trip_index=first_trip_index,
        max_trips_per_route_per_direction=max_trips_per_route_per_direction,
    )
    print(f"Trips loaded for search: {len(trips)}")

    rng = random.Random(args.seed)

    rows: List[Dict[str, Any]] = []
    best: List[Tuple[float, Params, Metrics]] = []

    total_trials = int(args.iters) * int(max(1, args.rounds))
    print(f"Search plan: rounds={args.rounds} × iters={args.iters} => total_trials={total_trials}")

    global_i = 0
    mutate_scale = float(max(0.0, args.init_scale))
    prev_round_elites: List[Params] = []

    for r in range(1, int(max(1, args.rounds)) + 1):
        elite_pool = prev_round_elites[: max(1, int(args.elite))] if prev_round_elites else []

        for j in range(1, int(args.iters) + 1):
            global_i += 1

            if (r == 1) or (not elite_pool) or (rng.random() < float(args.explore_frac)):
                p = sample_params(rng)
            else:
                elite = rng.choice(elite_pool)
                p = mutate_params(elite, rng, scale=mutate_scale)

            try:
                m, s = run_once(mod, trips, p)
            except Exception as e:
                m = Metrics(
                    n_trips=len(trips),
                    n_clusters=0,
                    n_singletons=len(trips),
                    n_non_singletons=0,
                    mean_non_singleton_size=0.0,
                    median_non_singleton_size=0.0,
                    std_non_singleton_size=0.0,
                    pct_non_singletons_in_15_25=0.0,
                    max_cluster_size=0,
                    min_non_singleton_size=0,
                )
                s = 1e9
                print(f"[round {r} | {j:04d}/{args.iters} | {global_i:04d}/{total_trials}] FAILED: {e}")

            row = {
                "round": int(r),
                "trial_in_round": int(j),
                "trial_global": int(global_i),
                **asdict(p),
                **asdict(m),
                "score": float(s),
            }
            rows.append(row)

            best.append((s, p, m))
            best.sort(key=lambda x: x[0])
            best = best[: max(args.topk, 25)]

            if global_i % max(1, total_trials // 10) == 0:
                b0 = best[0]
                print(
                    f"[{global_i:04d}/{total_trials}] best score={b0[0]:.6f} | "
                    f"singletons={b0[2].n_singletons} | "
                    f"median_non={b0[2].median_non_singleton_size:.2f} | "
                    f"pct15-25={b0[2].pct_non_singletons_in_15_25:.2f}"
                )

        prev_round_elites = [bp for (_, bp, _) in best[: max(1, int(args.elite))]]
        mutate_scale *= float(args.scale_decay)

    df = pd.DataFrame(rows).sort_values("score", ascending=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote results to: {args.out_csv}")

    print(f"\nTop {args.topk} parameter sets:")
    for rank, (s, p, m) in enumerate(best[: args.topk], start=1):
        print(f"\n#{rank} score={s:.6f}")
        print("  Params:", asdict(p))
        print("  Metrics:", asdict(m))


if __name__ == "__main__":
    main()
