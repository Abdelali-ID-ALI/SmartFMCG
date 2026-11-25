"""Disjoint synthetic scenario banks derived from the validated buffered tours."""

from dataclasses import dataclass, replace
from datetime import date
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from src.optimization.fmcg_problem import FMCGProblem
from src.optimization.fmcg_validator import Tour
from src.planning.forecast import build_forecasts
from src.simulation.experiment import allocate_reserves, beverage_load, generate_scenario, load_reference, realized_grids
from src.simulation.simulator import Simulator
from src.simulation.state import Scenario, Settings, UrgentOrder, minus

from src.rl.environment import Episode

PARTITIONS = ("train", "validation", "test")
REGIMES = ("moderate", "severe", "extreme")


@dataclass(frozen=True)
class Case:
    key: str
    split: str
    day: date
    seed: int
    regime: str
    problem: FMCGProblem
    tours: tuple[Tour, ...]
    products: dict
    priority: frozenset[str]
    reserves: tuple
    scenario: Scenario

    def episode(self, settings: Settings, urgent_penalty_cents: int | None = None) -> Episode:
        scenario = self.scenario
        if urgent_penalty_cents is not None:
            events = tuple(
                replace(event, order=replace(event.order, stop=replace(event.order.stop,
                                                                      penalty_cents=urgent_penalty_cents)))
                if isinstance(event, UrgentOrder) else event for event in scenario.events
            )
            scenario = replace(scenario, events=events)
        return Episode(self.key, partial(Simulator, self.problem, self.tours, self.products,
                                         self.priority, self.reserves, scenario, settings))


def slice_grid(grid, nodes):
    return tuple(tuple(grid[i][j] for j in nodes) for i in nodes)


def select_instance(problem, tours, count, rng, max_reduction):
    possible = [group for group in combinations(tours, count) if 10 <= sum(len(t.stops) for t in group) <= 20]
    if not possible:
        possible = list(combinations(tours, count))
    chosen = possible[int(rng.integers(len(possible)))]
    nodes = (0, *sorted(node for tour in chosen for node in tour.stops))
    stops = []
    for node in nodes[1:]:
        stop = problem.stops[node - 1]
        reduction = int(rng.integers(min(stop.delivery[2] - 1, max_reduction) + 1))
        weight, volume, crates, _ = minus(stop.delivery, beverage_load(reduction))
        pallets = min(stop.delivery[3], (crates + 39) // 40)
        stops.append(replace(stop, delivery=(weight, volume, crates, pallets)))
    remap = {node: i for i, node in enumerate(nodes)}
    vehicle_indices = tuple(t.vehicle for t in chosen)
    sliced = FMCGProblem(tuple(stops), tuple(problem.vehicles[i] for i in vehicle_indices), problem.depot_window,
                         slice_grid(problem.distance_m, nodes), slice_grid(problem.travel_s, nodes))
    reduced = tuple(replace(tour, vehicle=i, stops=tuple(remap[node] for node in tour.stops), initial_delivery=None)
                    for i, tour in enumerate(chosen))
    return sliced, reduced, nodes


def build_banks(config: dict, profile: str, *, only: tuple[str, ...] = PARTITIONS) -> dict[str, tuple[Case, ...]]:
    reference, matrix, base, model, history, plans, products, priority = load_reference()
    events_config = yaml.safe_load(Path("configs/dynamic_simulation.yaml").read_text())
    events_config.update(urgent_penalty_cents=config["urgent_penalty_cents"])
    settings = config["profiles"][profile]
    splits = config["partitions"]
    days = tuple(plans)
    all_days = [i for split in PARTITIONS for i in splits[split]["day_indices"]]
    namespaces = [splits[split]["namespace"] for split in PARTITIONS]
    if len(set(all_days)) != len(all_days) or len(set(namespaces)) != len(namespaces):
        raise ValueError("Dates and random namespaces must be disjoint")
    banks = {}
    for split, per_day in zip(PARTITIONS, settings["per_day"], strict=True):
        if split not in only:
            continue
        rows = []
        for i in splits[split]["day_indices"]:
            day = days[i]
            forecasts = build_forecasts(matrix, day, model, history)
            full = forecasts.grid(base, "ml", np.full(len(forecasts.arcs), reference["reference_hour"]),
                                  reference["buffer_ratio"])
            for slot in range(per_day):
                regime = REGIMES[(slot + i) % len(REGIMES)]
                name = f"{split}-{day}-{slot}"
                entropy = [splits[split]["namespace"], day.toordinal(), slot]
                if profile == "research":
                    entropy.append(8)
                rng = np.random.default_rng(np.random.SeedSequence(entropy))
                seed = int(rng.integers(2**31))
                problem, tours, nodes = select_instance(full, plans[day], settings["routes"], rng,
                                                         config["regular_crates_reduction"])
                actual = tuple(slice_grid(grid, nodes) for grid in
                               realized_grids(base, forecasts, day, seed, matrix.ids, model.zones))
                ids = tuple(matrix.ids[node] for node in nodes)
                scenario = generate_scenario(problem, tours, actual, day, seed, events_config["regimes"][regime],
                                             ids, model.zones, events_config)
                reserves = allocate_reserves(problem, tours, int(rng.choice(config["reserve_crates"])))
                known_products = {stop.id: products[stop.id] for stop in problem.stops}
                rows.append(Case(name, split, day, seed, regime, problem, tours, known_products,
                                 priority.intersection(known_products), reserves, scenario))
        banks[split] = tuple(rows)
    if len({case.seed for rows in banks.values() for case in rows}) != sum(map(len, banks.values())):
        raise ValueError("Scenario seeds unexpectedly collided")
    return banks
