"""Use the qualified SmartFMCG components for a bounded interactive scenario."""

from dataclasses import asdict, dataclass, replace
from datetime import date
from functools import lru_cache
from io import BytesIO
from math import ceil
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Callable
import json

import numpy as np
import torch
import yaml

from src.benchmark.evaluation import ARCHIVE, audit_metrics, load_dispatch, policy_runs
from src.benchmark.scenarios import Draw, fingerprint, planning_capacity, traffic_realization
from src.model_artifacts import verified_bytes
from src.optimization.fmcg_greedy import schedule
from src.optimization.fmcg_problem import FMCGProblem, Load, Stop, ZERO, integer_road_grids
from src.optimization.fmcg_solver import solve_fmcg
from src.optimization.fmcg_validator import PlanReport, Tour, validate_fmcg
from src.rl.dqn import DQNEnv, DispatchDQN
from src.rl.dqn_pilot import load_config
from src.rl.environment import Episode
from src.rl.rewards import RewardRates
from src.routing.base_provider import Location, TravelMatrix
from src.scalability.experiment import geography
from src.scalability.prediction import batched_forecasts
from src.scalability.scenarios import design_for, events_for, size_fleet, stops_for
from src.simulation.experiment import beverage_load, load_reference
from src.simulation.simulator import Simulator
from src.simulation.state import Event, Scenario, Settings, VehicleView, plus

SIZES = (12, 30, 50, 100)
TEMPLATES = ("V01", "V02", "V03", "V04", "V05", "V06")
PLANNERS = {"Static OSRM": "static", "Historical average": "historical",
            "Linear regression": "ml", "Robust ML": "ml"}
POLICIES = {"B4": "Fixed robust plan", "B5": "Event-driven", "B8": "Multicriteria",
            "B9": "Tabular Q-Learning", "B10": "Continuous DQN", "B12": "Compact DQN"}
RULES = {"B4": "never", "B5": "always", "B8": "multi_feature"}
WORK = Lock()
COMPACT_PROTOCOL_SHA256 = "58de87b21de13bb223d5e97940353bebbbc9f7c83e1853b9694181deb2b90a60"


@dataclass(frozen=True)
class Inputs:
    customers: int = 12
    seed: int = 317
    max_vehicles: int = 16
    templates: tuple[str, ...] = TEMPLATES
    demand: float = 1.
    reserve: int = 8
    regime: str = "controlled"
    traffic: float = 1.30
    service_min: int = 7
    urgents: int = 1
    urgent_window_min: int = 55
    urgent_crates: tuple[int, int] = (2, 6)

    def __post_init__(self):
        if type(self.customers) is not int or self.customers not in SIZES:
            raise ValueError("Choose a cached layout with 12, 30, 50 or 100 customers.")
        bounds = ((self.seed, 0, 2**32 - 1), (self.max_vehicles, 1, 16), (self.reserve, 0, 12),
                  (self.service_min, 0, 20), (self.urgents, 0, min(self.customers, 10)),
                  (self.urgent_window_min, 30, 90))
        if any(type(v) is not int or not low <= v <= high for v, low, high in bounds):
            raise ValueError("Check the seed, fleet, stock and event bounds.")
        if (not .75 <= self.demand <= 1.20 or not 1 <= self.traffic <= 2.25
                or self.regime not in {"controlled", "stress"}):
            raise ValueError("Demand or disruption severity is outside the demo range.")
        if (not self.templates or len(set(self.templates)) != len(self.templates)
                or not set(self.templates) <= set(TEMPLATES)):
            raise ValueError("Select at least one supported vehicle type.")
        low, high = self.urgent_crates
        if type(low) is not int or type(high) is not int or not 1 <= low <= high <= 16:
            raise ValueError("Urgent order size must be between 1 and 16 crates.")


@dataclass(frozen=True)
class Prepared:
    inputs: Inputs
    locations: tuple[Location, ...]
    problem: FMCGProblem
    reserves: tuple[tuple[Load, Load], ...]
    plans: dict[str, FMCGProblem]
    witness: tuple[Tour, ...]
    products: dict[str, str]
    priority: frozenset[str]
    scenario: Scenario
    settings: Settings
    forecast_hour: int
    margin: float
    case_id: str
    day: date
    wall_s: float


@dataclass(frozen=True)
class Planned:
    strategy: str
    problem: FMCGProblem
    tours: tuple[Tour, ...]
    reserves: tuple[tuple[Load, Load], ...]
    report: PlanReport
    status: str
    detail: str
    budget_s: float
    wall_s: float


@dataclass(frozen=True)
class Frame:
    time_s: int
    vehicles: tuple[VehicleView, ...]
    completed: tuple[str, ...]
    waiting_urgent: tuple[str, ...]
    events: tuple[Event, ...]
    decisions: tuple[dict, ...]
    loss_mad: float | None


@dataclass(frozen=True)
class Run:
    policy: str
    training_seed: int | None
    frames: tuple[Frame, ...]
    metrics: dict
    costs: dict[str, float]
    wall_s: float
    inference_s: float


@lru_cache(maxsize=1)
def reference():
    return load_reference()


@lru_cache(maxsize=4)
def layout(customers: int) -> tuple[tuple[Location, ...], TravelMatrix]:
    _, matrix, *_ = reference()
    config = yaml.safe_load(Path("configs/scalability.yaml").read_text())
    if customers == 12:
        locations, full, _ = geography(30, "osrm", config, matrix)
        n = customers + 1
        locations = locations[:n]
        selected = TravelMatrix(full.ids[:n], full.distance_km[:n, :n].copy(),
                                full.duration_min[:n, :n].copy(), full.provenance)
    else:
        locations, selected, _ = geography(customers, "osrm", config, matrix)
    selected.distance_km.setflags(write=False)
    selected.duration_min.setflags(write=False)
    return locations, selected


def scale_demand(stop: Stop, multiplier: float) -> Stop:
    crates = max(1, round(stop.delivery[2] * multiplier))
    pallets = ceil(crates / 40) if stop.delivery[3] else 0
    service_delta = 60 * (ceil(crates / 6) - ceil(stop.delivery[2] / 6))
    return replace(stop, delivery=(*beverage_load(crates)[:3], pallets), service_s=stop.service_s + service_delta)


def prepare(inputs: Inputs) -> Prepared:
    started = perf_counter()
    ref, reference_matrix, base, model, history, *_ = reference()
    config = yaml.safe_load(Path("configs/scalability.yaml").read_text())
    dispatch = load_config()
    settings = Settings(**dispatch["dispatch"])
    day = date(2026, 9, 14)
    locations, matrix = layout(inputs.customers)
    zones = dict(model.zones)
    anchors = locations[1:len(reference_matrix.ids)]
    for point in locations[len(reference_matrix.ids):]:
        nearest = min(anchors, key=lambda a: (a.latitude - point.latitude)**2
                      + ((a.longitude - point.longitude) * .83)**2)
        zones[point.id] = zones[nearest.id]
    forecasts, _ = batched_forecasts(matrix, day, replace(model, zones=zones), history,
                                     config["prediction_batch_arcs"])
    distances, times = integer_road_grids(matrix)
    stops = stops_for(matrix.ids, base, inputs.seed, config["fleet"]["window_s"])
    if inputs.demand != 1:
        stops = tuple(scale_demand(stop, inputs.demand) for stop in stops)
    if any(stop.refrigerated for stop in stops) and "V06" not in inputs.templates:
        raise ValueError("Cold-chain orders require the refrigerated fleet template (V06).")
    physical = replace(base, stops=stops, distance_m=distances, travel_s=times,
                       vehicles=tuple(v for v in base.vehicles if v.id in inputs.templates))
    hour, margin = ref["reference_hour"], ref["buffer_ratio"]
    hours = np.full(len(forecasts.arcs), hour)
    robust = forecasts.grid(physical, "ml", hours, margin)
    fleet_parameters = {k: v for k, v in config["fleet"].items() if k != "window_s"}
    instance = size_fleet(robust, zones, inputs.reserve, **fleet_parameters)
    if len(instance.problem.vehicles) > inputs.max_vehicles:
        raise ValueError(f"The qualified fleet sizing needs {len(instance.problem.vehicles)} vehicles. "
                         "Increase the fleet limit within 16 vehicles or reduce demand/reserve.")
    plans = {name: replace(forecasts.grid(physical, source, hours, margin if name == "Robust ML" else 0.),
                           vehicles=instance.problem.vehicles) for name, source in PLANNERS.items()}
    design, delays = design_for(config["regimes"][inputs.regime], inputs.customers)
    design = replace(design, demand=inputs.demand, reserve=inputs.reserve, traffic_factor=inputs.traffic,
                     service_extra_s=inputs.service_min * 60, urgents=inputs.urgents,
                     urgent_window_s=inputs.urgent_window_min * 60, urgent_crates=inputs.urgent_crates)
    case_id = fingerprint(asdict(inputs))
    draw = Draw(case_id, day, inputs.seed, inputs.regime, "dashboard", "custom",
                design, config["urgent_penalty_cents"], settings.latency_s)
    actual, _ = traffic_realization(physical, forecasts, draw, matrix.ids, zones, hour)
    events = events_for(physical, draw, matrix.ids, zones, delays if inputs.service_min else 0,
                        config["urgent_service_s"])
    return Prepared(inputs, locations, instance.problem, instance.reserves, plans, instance.witness,
                    instance.products, instance.priority, Scenario(events, actual), settings,
                    hour, margin, case_id, day, perf_counter() - started)


def plan(case: Prepared, strategy: str) -> Planned:
    if strategy not in PLANNERS:
        raise ValueError("Choose a supported planning strategy.")
    started = perf_counter()
    public = case.plans[strategy]
    effective = planning_capacity(public, case.reserves)
    try:
        witness = tuple(schedule(effective, t.vehicle, t.stops) for t in case.witness)
        validate_fmcg(effective, witness)
    except ValueError:
        witness = None
    budget = 0.65 if case.inputs.customers <= 50 else 1.8
    result = solve_fmcg(effective, time_limit_s=budget, initial_tours=witness)
    if result.status in {"feasible", "optimal"}:
        tours, status, detail = result.tours, result.status, result.detail
    elif witness:
        tours, status, detail = witness, "feasible witness", f"Solver status: {result.status}. {result.detail}"
    else:
        raise ValueError(f"Planner status: {result.status}. {result.detail}")
    report = validate_fmcg(effective, tours)
    validate_fmcg(public, tours)
    active = {tour.vehicle for tour in tours}
    reserves = tuple(stock if i in active else (ZERO, ZERO) for i, stock in enumerate(case.reserves))
    return Planned(strategy, public, tours, reserves, report, status, detail, budget, perf_counter() - started)


@lru_cache(maxsize=1)
def frozen_dispatch():
    torch.set_num_threads(2)
    return load_dispatch()


@lru_cache(maxsize=3)
def compact_checkpoint(seed: int):
    manifest = Path("results/experiments/final_benchmark")
    rows = json.loads(verified_bytes(manifest, Path("approximation_protocol.json"),
                                    COMPACT_PROTOCOL_SHA256))["checkpoints"]
    entry = next(row for row in rows if row["training_seed"] == seed)
    checkpoint = verified_bytes(ARCHIVE, Path(entry["path"]), entry["path_sha256"])
    torch.set_num_threads(2)
    return DispatchDQN.load(BytesIO(checkpoint), device="cpu")


def policy_for(name: str, seed: int) -> tuple[Callable, str]:
    if name in RULES:
        return policy_runs(RULES[name], {}, 0)[0][1], "continuous"
    if name not in POLICIES or seed not in {2917, 2927, 2953}:
        raise ValueError("Choose a supported policy and frozen training seed.")
    if name == "B12":
        model = compact_checkpoint(seed)
        return lambda obs, info, public: int(model.predict(obs, deterministic=True)[0]), "compact"
    _, frozen = frozen_dispatch()
    return frozen["tabular" if name == "B9" else "dqn"][seed], "continuous"


def replay(case: Prepared, initial: Planned, policy: str, seed: int = 2917) -> Run:
    started = perf_counter()
    if initial.strategy != "Robust ML":
        raise ValueError("Paired operational replay requires the shared robust ML plan.")
    decide, encoding = policy_for(policy, seed)
    config = load_config()
    rates = RewardRates(**config["reward"])
    env = DQNEnv((Episode(case.case_id, lambda: Simulator(
        initial.problem, initial.tours, case.products, case.priority, initial.reserves,
        case.scenario, case.settings)),), input_mode=encoding, rates=rates,
        solver_budget=config["solver_budget"], probe_budget=config["probe_budget"],
        tight_s=config["tight_slack_s"], max_steps=256)
    obs, info = env.reset(seed=case.inputs.seed)
    sim = env._sim
    frames, journal, total_reward, inference_s = [], {}, 0., 0.
    while True:
        public = env.public_observation
        frames.append(Frame(sim.now_s, public.vehicles if public else sim._views(), tuple(sorted(sim.completed)),
                            public.waiting_urgent if public else sim._waiting_urgent(),
                            tuple(event for event in case.scenario.events if event.at_s <= sim.now_s),
                            tuple(journal.values()), sum(info["costs_mad"].values()) if "costs_mad" in info else None))
        if public is None:
            break
        tick = perf_counter()
        action = int(decide(obs, info, public))
        inference_s += perf_counter() - tick
        obs, reward, _, truncated, info = env.step(action)
        total_reward += reward
        if truncated:
            raise RuntimeError("The bounded interactive replay reached its decision limit.")
        for record in (info["decision"], *info["settled_decisions"]):
            journal[record["decision_index"]] = record
    metrics = env.episode_metrics()
    metrics.update(audit_metrics(sim, metrics, info["costs_mad"], rates, total_reward))
    return Run(policy, seed if policy not in RULES else None, tuple(frames), metrics,
               info["costs_mad"], perf_counter() - started, inference_s)


def cargo_with_reserve(load: Load, reserve: tuple[Load, Load]) -> Load:
    return plus(load, plus(*reserve))
