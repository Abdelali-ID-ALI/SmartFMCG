"""Fresh event draws for the refinement study, on top of validated tours."""

from dataclasses import replace
from pathlib import Path
import json

import numpy as np

from src.optimization.fmcg_problem import ZERO, Stop
from src.rl.scenarios import build_banks
from src.simulation.experiment import beverage_load
from src.simulation.state import AMBIENT, CHILLED, Congestion, Order, ServiceDelay, UrgentOrder

ARCHIVES = ("q_learning", "q_learning_research_stress")


def enrich(case, probability):
    rng = np.random.default_rng(np.random.SeedSequence([case.seed, 8814]))
    events = []
    for event in case.scenario.events:
        if isinstance(event, Congestion):
            at = max(case.problem.depot_window[0], event.at_s + int(rng.integers(-180, 181)))
            severity = 1 + (event.factor - 1) * rng.uniform(.85, 1.15)
            events.append(replace(event, at_s=at, end_s=at + event.end_s - event.at_s, factor=float(severity)))
        elif isinstance(event, ServiceDelay):
            events.append(replace(event, extra_s=max(1, round(event.extra_s * rng.uniform(.85, 1.15)))))
        else:
            events.append(event)
    first = next(e for e in events if isinstance(e, UrgentOrder))
    if rng.random() < probability:
        release = first.at_s + int(rng.integers(120, 1201))
        horizon = int(rng.choice([900, 1800, 2700]))
        sites = [(i, stop) for i, stop in enumerate(case.problem.stops, 1) if i != first.order.node
                 and max(stop.window[0], release) + 180 <= min(stop.window[1], release + horizon)]
        if sites and release <= case.problem.depot_window[1]:
            node, site = sites[int(rng.integers(len(sites)))]
            cold = bool(rng.random() < .35)
            stop = Stop("U02", beverage_load(int(rng.integers(2, 10))), ZERO,
                        (max(site.window[0], release), min(site.window[1], release + horizon)),
                        180, cold, site.no_heavy, first.order.stop.penalty_cents)
            events.append(UrgentOrder(release, Order(stop, node, CHILLED if cold else AMBIENT, priority=True)))
    return replace(case, scenario=replace(case.scenario, events=tuple(sorted(events, key=lambda e: e.at_s))))


def fresh_banks(config, profile, only):
    # Distinct namespaces also separate the development and final experiments.
    offset = {"development": 0, "standard": 100000, "research": 200000}[profile]
    adjusted = {**config, "partitions": {key: {**part, "namespace": part["namespace"] + offset}
                                        for key, part in config["partitions"].items()}}
    archives = Path("results/experiments")
    old = {row["seed"] for experiment in ARCHIVES
           for row in json.loads((archives / experiment / "scenario_manifest.json").read_text())}
    banks = build_banks(adjusted, profile, only=only)
    if old & {case.seed for cases in banks.values() for case in cases}:
        raise ValueError("Refinement scenario collided with a historical generator seed")
    return {key: tuple(enrich(case, config["second_urgent_probability"]) for case in cases)
            for key, cases in banks.items()}
