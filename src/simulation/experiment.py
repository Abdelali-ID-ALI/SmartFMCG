"""Compare online dispatch policies on paired synthetic distribution days."""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from math import ceil
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from src.ml.features import TARGET, chronological_split, make_features
from src.ml.model import HistoricalAverage
from src.model_artifacts import load_traffic_model
from src.optimization.fmcg_problem import FMCGProblem, Stop, ZERO, load_fmcg
from src.optimization.fmcg_validator import Tour, validate_fmcg
from src.planning.forecast import Forecasts, build_forecasts
from src.planning.replay import actual_traffic
from src.routing.matrices import cached_matrix, load_locations, matrix_path
from src.routing.osrm_client import OSRMProvider
from src.simulation.metrics import evaluate
from src.simulation.policies import POLICIES
from src.simulation.simulator import Simulator
from src.simulation.state import (AMBIENT, CHILLED, Congestion, Order, Scenario, ServiceDelay,
                                  Settings, UrgentOrder, plus, within)

REFERENCE = Path("results/experiments/predict_then_optimize")
PLAN_SHA256 = "39b73656727d4744ea9517505e22c71c720311a6688efa2d7f3cb5c303b4c31e"
SOURCE = Path("data/synthetic")
PAIR_METRICS = ("otif", "regular_otif", "urgent_otif", "realized_cost_mad", "operating_cost_mad",
                "delay_min", "distance_km", "route_changes", "replans_applied")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference():
    phase6 = json.loads((REFERENCE / "manifest.json").read_text())
    phase5 = json.loads(Path("results/experiments/traffic_ml/manifest.json").read_text())
    model = load_traffic_model()
    if (model.family, model.target_mode) != ("linear", "ratio"):
        raise ValueError("Expected the phase-five linear ratio model")
    config = yaml.safe_load(Path("configs/development.yaml").read_text())
    provider = OSRMProvider(config["osrm_endpoint"])
    locations = load_locations(SOURCE)
    cache = Path(config["matrix_cache"])
    for key, path in {"matrix_sha256": matrix_path(provider, locations, cache),
                      "traffic_sha256": SOURCE / "traffic_observations.csv"}.items():
        if digest(path) != phase6[key]:
            raise ValueError(f"Phase-six reference mismatch: {path}")
    matrix = cached_matrix(provider, locations, cache, cache_only=True)
    problem = load_fmcg(matrix, SOURCE)
    traffic = pd.read_csv(SOURCE / "traffic_observations.csv")
    train = chronological_split(traffic, phase5["config"]["training_days"], phase5["config"]["validation_days"])["train"]
    history = HistoricalAverage.fit(make_features(train, model.zones), train[TARGET])
    plan_path = REFERENCE / "plans.json"
    if digest(plan_path) != PLAN_SHA256:
        raise ValueError("Expected the saved plans from phase-six checkpoint 0a83bd0")
    reference_plans = json.loads(plan_path.read_text())
    node = {stop.id: i for i, stop in enumerate(problem.stops, 1)}
    vehicle = {truck.id: i for i, truck in enumerate(problem.vehicles)}
    tours = {
        date.fromisoformat(row["day"]): tuple(Tour(vehicle[t["vehicle"]], tuple(node[key] for key in t["stops"]),
                                                  t["departure_s"], tuple(t["service_starts_s"]), t["return_s"])
                                              for t in row["tours"])
        for row in reference_plans if row["strategy"] == "buffered"
    }
    orders = pd.read_csv(SOURCE / "orders.csv")
    products = dict(zip(orders.customer_id, orders.product_type, strict=True))
    priority = frozenset(orders.loc[orders.priority == "high", "customer_id"])
    return phase6, matrix, problem, model, history, tours, products, priority


def beverage_load(crates: int) -> tuple[int, int, int, int]:
    return 12000 * crates, 32 * crates, crates, 0


def allocate_reserves(problem: FMCGProblem, tours: tuple[Tour, ...], crates: int):
    # Each active vehicle draws up to 'crates' from depot stock for one compatible product.
    # Use the free capacity at every static stop, not only at departure.
    report = validate_fmcg(problem, tours)
    reserves = [(ZERO, ZERO)] * len(problem.vehicles)
    for tour, path in zip(tours, report.tours, strict=True):
        vehicle = problem.vehicles[tour.vehicle]
        for count in range(crates, -1, -1):
            spare = beverage_load(count)
            if all(within(plus(state.onboard, spare), vehicle.capacity) for state in path.loads):
                reserves[tour.vehicle] = (ZERO, spare) if vehicle.refrigerated else (spare, ZERO)
                break
    return tuple(reserves)


def realized_grids(base: FMCGProblem, forecasts: Forecasts, day: date, seed: int, ids, zones):
    actual, _ = actual_traffic(forecasts.arcs, forecasts.static, day, seed, ids, zones)
    grids = [[list(row) for row in base.travel_s] for _ in range(24)]
    for hour, table in enumerate(grids):
        for (i, j), minutes in zip(forecasts.arcs, actual[hour], strict=True):
            table[i][j] = ceil(minutes * 60)
    return tuple(tuple(map(tuple, grid)) for grid in grids)


def generate_scenario(problem: FMCGProblem, tours: tuple[Tour, ...], actual, day: date,
                      seed: int, regime: dict, ids, zones, config: dict) -> Scenario:
    rng = np.random.default_rng(np.random.SeedSequence([seed, day.toordinal(), 7]))
    timetable = sorted((start, node) for tour in tours
                       for start, node in zip(tour.service_starts_s, tour.stops, strict=True))
    lower, upper = timetable[len(timetable) // 5][0], timetable[len(timetable) // 2][0]
    traffic_at = int(rng.integers(lower, upper + 1))
    zone = zones[ids[int(rng.choice([node for _, node in timetable]))]]
    nodes = frozenset(i for i, key in enumerate(ids) if zones[key] == zone)
    traffic = Congestion(traffic_at, traffic_at + regime["traffic_duration_s"], nodes, regime["traffic_factor"])
    service_start, service_node = timetable[int(rng.integers(len(timetable) // 4, 3 * len(timetable) // 4))]
    delay_at = max(problem.depot_window[0], service_start + int(rng.integers(-180, 301)))
    low, high = regime["extra_service_s"]
    delay = ServiceDelay(delay_at, problem.stops[service_node - 1].id, int(rng.integers(low, high + 1)))

    available_s = int(rng.integers(lower, upper + 1))
    eligible = [node for _, node in timetable if problem.stops[node - 1].window[1] >= available_s + 900]
    node = int(rng.choice(eligible))
    site = problem.stops[node - 1]
    cold = bool(rng.random() < .35)
    low, high = regime["urgent_crates"]
    demand = beverage_load(int(rng.integers(low, high + 1)))
    # A closed opening interval is avoided even for sites that open late.
    available_s = max(available_s, site.window[0] - regime["urgent_window_s"] + config["urgent_service_s"])
    window = max(site.window[0], available_s), min(site.window[1], available_s + regime["urgent_window_s"])
    stop = Stop("U01", demand, ZERO, window, config["urgent_service_s"],
                cold, site.no_heavy, config["urgent_penalty_cents"])
    urgent = UrgentOrder(available_s, Order(stop, node, CHILLED if cold else AMBIENT, priority=True))
    return Scenario(tuple(sorted((traffic, delay, urgent), key=lambda e: e.at_s)), actual)


def paired_results(frame: pd.DataFrame, draws: int, bootstrap_seed: int):
    keys = ["day", "seed", "regime"]
    metrics = list(PAIR_METRICS)
    base = frame.loc[frame.policy == "never", keys + metrics].set_index(keys)
    paired = []
    for policy in POLICIES:
        if policy == "never":
            continue
        other = frame.loc[frame.policy == policy, keys + metrics].set_index(keys)
        if not other.index.equals(base.index) or not other.index.is_unique:
            raise ValueError("Policies did not use the same paired scenarios")
        paired.append((other - base).reset_index().assign(policy=policy))
    differences = pd.concat(paired, ignore_index=True)
    rng = np.random.default_rng(bootstrap_seed)
    records = []
    for regime in ("all", *frame.regime.unique()):
        subset = differences if regime == "all" else differences[differences.regime == regime]
        days = sorted(subset.day.unique())
        resamples = rng.integers(len(days), size=(draws, len(days)))
        for policy, group in subset.groupby("policy"):
            day_means = group.groupby("day")[metrics].mean().loc[days]
            intervals = np.quantile(day_means.to_numpy()[resamples].mean(axis=1), [.025, .975], axis=0)
            for i, metric in enumerate(metrics):
                records.append({"regime": regime, "policy": policy, "metric": metric,
                                "paired_mean": group[metric].mean(), "paired_sd": group[metric].std(),
                                "ci_low": intervals[0, i], "ci_high": intervals[1, i], "days": len(days),
                                "pairs": len(group)})
    return differences, pd.DataFrame(records)


def run(profile: str, output: Path) -> None:
    started = perf_counter()
    config_path = Path("configs/dynamic_simulation.yaml")
    config = yaml.safe_load(config_path.read_text())
    settings = Settings(**config["dispatch"])
    profile_config = config["profiles"][profile]
    reference, matrix, base, model, history, plans, products, priority = load_reference()
    days = tuple(plans)[:profile_config["days"]]
    if len(days) != profile_config["days"] or not days:
        raise ValueError("Expected available dates in the phase-six checkpoint")
    rows, scenarios, decisions, example = [], [], [], []
    for day in days:
        forecasts = build_forecasts(matrix, day, model, history)
        hours = np.full(len(forecasts.arcs), reference["reference_hour"], dtype=int)
        problem = forecasts.grid(base, "ml", hours, reference["buffer_ratio"])
        tours = plans[day]
        reserves = allocate_reserves(problem, tours, config["reserve_crates"])
        for seed in profile_config["seeds"]:
            actual = realized_grids(base, forecasts, day, seed, matrix.ids, model.zones)
            for regime in profile_config["regimes"]:
                case = {"day": str(day), "seed": seed, "regime": regime}
                scenario = generate_scenario(problem, tours, actual, day, seed, config["regimes"][regime],
                                             matrix.ids, model.zones, config)
                scenarios.append({**case, "starting_reserves": reserves,
                                  "events": [{"type": type(event).__name__, **asdict(event)}
                                             for event in scenario.events]})
                stamps = None
                for name, policy in POLICIES.items():
                    start = perf_counter()
                    sim = Simulator(problem, tours, products, priority, reserves, scenario, settings)
                    while (observation := sim.next_observation()) is not None:
                        sim.act(policy(observation))
                    clock = [(d.time_s, d.events, d.tick) for d in sim.decisions]
                    if stamps is not None and clock != stamps:
                        raise RuntimeError("Decision clocks diverged within a paired scenario")
                    stamps = clock
                    key = {**case, "policy": name}
                    rows.append({**key, **evaluate(sim), "wall_s": perf_counter() - start})
                    decisions.extend({**key, **asdict(d)} for d in sim.decisions if d.requested != "KEEP")
                    if len(scenarios) == 1 and name in {"never", "multi_feature"}:
                        example.extend({**key, **asdict(m)} for m in sim.movements)
    frame = pd.DataFrame(rows)
    differences, intervals = paired_results(frame, config["bootstrap_draws"], config["bootstrap_seed"])
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (("scenario_metrics", frame), ("paired_differences", differences), ("paired_intervals", intervals),
                        ("decisions", pd.DataFrame(decisions)), ("example_movements", pd.DataFrame(example))):
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "scenarios.json").write_text(json.dumps(scenarios, indent=2, default=sorted) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "reference_checkpoint": "0a83bd0", "phase6_plans_sha256": digest(REFERENCE / "plans.json"),
        "matrix_sha256": reference["matrix_sha256"], "model_sha256": digest(Path("models/traffic_model.joblib")),
        "config_sha256": digest(config_path), "profile": profile, "settings": profile_config, "days": list(map(str, days)),
        "scenario_count": len(scenarios), "policy_runs": len(rows), "wall_s": perf_counter() - started,
        "peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
    }, indent=2) + "\n")
    print(frame.groupby("policy")[list(PAIR_METRICS) + ["replan_attempts", "wall_s"]].mean().round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="development", choices=["development", "standard", "research"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = "" if args.profile == "standard" else f"_{args.profile}"
    run(args.profile, args.output or Path(f"results/experiments/dynamic_simulation{suffix}"))


if __name__ == "__main__":
    main()
