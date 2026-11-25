"""Attribute phase-seven extreme urgent omissions before training."""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, asdict, replace
from itertools import permutations
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from src.planning.forecast import build_forecasts
from src.simulation.experiment import load_reference, realized_grids
from src.simulation.policies import Action
from src.simulation.replanning import Proposal, project, propose, route_changes
from src.simulation.simulator import Simulator
from src.simulation.state import Congestion, Order, PRODUCTS, Scenario, ServiceDelay, Settings, UrgentOrder, within
from src.optimization.fmcg_problem import Stop
from src.optimization.fmcg_validator import UNITS_PER_CENT, UNITS_PER_MAD
from src.simulation.validation import validate_routes


@dataclass
class Diagnosis:
    day: str
    seed: int
    crates: int
    product: str
    window_s: int
    compatible_stock_crates: int
    opportunities: int = 0
    standalone_forecast_possible: bool = False
    fixed_order_feasible: bool = False
    reordered_feasible: bool = False
    best_slack_s: int | None = None
    permutations_tested: int = 0
    physically_feasible_suffix: bool = False
    first_feasible_s: int | None = None
    best_insertion_delta_at_first_mad: float | None = None
    phase7_search_found_at_first: bool | None = None
    witness_applied: bool | None = None
    witness_served: bool | None = None
    witness_otif: bool | None = None
    cause: str = ""


def restore_events(rows) -> tuple:
    events = []
    for raw in rows:
        row = deepcopy(raw)
        kind = row.pop("type")
        if kind == "Congestion":
            row["nodes"] = frozenset(row["nodes"])
            events.append(Congestion(**row))
        elif kind == "ServiceDelay":
            events.append(ServiceDelay(**row))
        else:
            order = row["order"]
            for field in ("delivery", "pickup", "window"):
                order["stop"][field] = tuple(order["stop"][field])
            order["stop"] = Stop(**order["stop"])
            events.append(UrgentOrder(row["at_s"], Order(**order)))
    return tuple(events)


def diagnose(sim: Simulator, event: UrgentOrder, day: str, seed: int) -> Diagnosis:
    order, stop = event.order, event.order.stop
    product = PRODUCTS.index(order.product)
    compatible = [t.vehicle for t in sim.tours if sim.problem.vehicles[t.vehicle].compatible(stop)]
    stock = max((sim.starting_reserves[i][product][2] for i in compatible), default=0)
    result = Diagnosis(day, seed, stop.delivery[2], order.product, stop.window[1] - stop.window[0], stock)
    supplied = [i for i in compatible if within(stop.delivery, sim.starting_reserves[i][product])]
    if not compatible or not supplied:
        result.cause = "incompatible_active_fleet" if not compatible else "insufficient_reserve"
        return result
    witness = None
    while (obs := sim.next_observation()) is not None:
        if stop.id in obs.waiting_urgent and obs.replan_available:
            result.opportunities += 1
            observed_forecast = sim.dispatch_forecast()
            release = obs.now_s + sim.settings.latency_s
            original = tuple(view.tail for view in obs.vehicles)
            for i in supplied:
                view = obs.vehicles[i]
                if not view.mutable:
                    continue
                start = max(view.anchor_s, release)
                leg = observed_forecast.travel(view.anchor_node, order.node, start)
                if leg is None:
                    continue
                end = max(stop.window[0], start + leg) + stop.service_s
                slack = stop.window[1] - end
                result.best_slack_s = slack if result.best_slack_s is None else max(result.best_slack_s, slack)
                result.standalone_forecast_possible |= slack >= 0

                def feasible(sequence, *, check_windows=True):
                    plan = (*original[:i], sequence, *original[i + 1:])
                    try:
                        validate_routes(observed_forecast, obs.vehicles, plan, release, frozenset({stop.id}),
                                        check_windows=check_windows)
                    except ValueError:
                        return None
                    return plan

                def remember(plan):
                    nonlocal witness
                    if witness is not None:
                        return
                    witness = plan
                    result.first_feasible_s = obs.now_s
                    proposed = propose(observed_forecast, obs.vehicles, (stop.id,), Action.GLOBAL,
                                       release, sim.settings)
                    result.phase7_search_found_at_first = any(stop.id in route for route in proposed.routes)

                for pos in range(len(view.tail) + 1):
                    sequence = (*view.tail[:pos], stop.id, *view.tail[pos:])
                    if not result.physically_feasible_suffix:
                        result.physically_feasible_suffix = feasible(sequence, check_windows=False) is not None
                    if plan := feasible(sequence):
                        result.fixed_order_feasible = result.reordered_feasible = True
                        remember(plan)
                        if obs.now_s == result.first_feasible_s:
                            before = project(observed_forecast, view, view.tail, release)
                            after = project(observed_forecast, view, sequence, release)
                            delay_s = sum(d for _, d in after.delays) - sum(d for _, d in before.delays)
                            overtime_s = after.overtime_s - before.overtime_s
                            delta = (after.cost_units - before.cost_units
                                     + (sim.settings.delay_cents_per_min * delay_s
                                        + sim.settings.overtime_cents_per_min * overtime_s) * (UNITS_PER_CENT // 60)
                                     + (sim.settings.change_cents * route_changes(view.tail, sequence)
                                        - stop.penalty_cents) * UNITS_PER_CENT) / UNITS_PER_MAD
                            current = result.best_insertion_delta_at_first_mad
                            result.best_insertion_delta_at_first_mad = delta if current is None else min(current, delta)
                if not result.reordered_feasible:
                    for route in permutations((*view.tail, stop.id)):
                        result.permutations_tested += 1
                        if plan := feasible(route):
                            result.reordered_feasible = True
                            remember(plan)
                            break
        sim.act(Action.KEEP)
    if result.fixed_order_feasible:
        result.cause = "feasible_insertion_on_static_trajectory"
    elif result.reordered_feasible:
        result.cause = "feasible_after_suffix_reordering"
    elif result.standalone_forecast_possible:
        result.cause = "whole_suffix_constraints_on_static_trajectory"
    else:
        result.cause = "urgent_window_or_vehicle_unavailable_on_static_trajectory"
    if witness is not None:
        fork = replace(sim)
        while (obs := fork.next_observation()) is not None:
            if obs.now_s == result.first_feasible_s:
                fork.act(Action.GLOBAL, planner=lambda *args: Proposal(witness, 0))
            else:
                fork.act(Action.KEEP)
        result.witness_applied = any(d.status == "applied" for d in fork.decisions)
        result.witness_served = stop.id in fork.completed
        result.witness_otif = result.witness_served and fork.completed[stop.id] <= stop.window[1]
    return result


def run(output: Path) -> None:
    data = json.loads(Path("results/experiments/dynamic_simulation_research/scenarios.json").read_text())
    settings = Settings(**yaml.safe_load(Path("configs/dynamic_simulation.yaml").read_text())["dispatch"])
    reference, matrix, base, model, history, plans, products, priority = load_reference()
    rows = []
    for day, tours in plans.items():
        forecasts = build_forecasts(matrix, day, model, history)
        forecast = forecasts.grid(base, "ml", np.full(len(forecasts.arcs), reference["reference_hour"]),
                                  reference["buffer_ratio"])
        for case in (d for d in data if d["regime"] == "extreme" and d["day"] == str(day)):
            events = restore_events(case["events"])
            actual = realized_grids(base, forecasts, day, case["seed"], matrix.ids, model.zones)
            reserves = tuple(tuple(tuple(stock) for stock in vehicle) for vehicle in case["starting_reserves"])
            sim = Simulator(forecast, tours, products, priority, reserves, Scenario(events, actual), settings)
            urgent = next(e for e in events if isinstance(e, UrgentOrder))
            rows.append(asdict(diagnose(sim, urgent, str(day), case["seed"])))
    if len(rows) != 33:
        raise ValueError("Incomplete phase-seven extreme audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(Counter(row["cause"] for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/experiments/q_learning/extreme_audit.csv"))
    run(parser.parse_args().output)
