"""Bounded public probes and continuous dispatch indicators."""

from dataclasses import dataclass
from math import inf

import numpy as np

from src.optimization.fmcg_validator import UNITS_PER_MAD
from src.rl.actions import can_supply, usable_stock
from src.rl.rewards import RewardRates
from src.simulation.replanning import Forecast, project, route_changes
from src.simulation.state import Observation, PRODUCTS, Settings

CARDINALITIES = (7, 3, 2, 2)
CONTINUOUS_NAMES = (
    "eligible", "clock", "waiting", "minimum_best_slack", "mean_best_slack",
    "mean_direct_arrival_slack", "compatible_vehicles", "stocked_vehicles",
    "ambient_free_crates", "chilled_free_crates", "feasible_fraction", "probe_exhausted",
    "estimated_gain", "estimated_cost", "delay_pressure", "priority_risk", "mutable_stops",
    "flexible_vehicles", "congestion", "since_attempt", "since_application", "urgency_coverage",
)


@dataclass(frozen=True)
class Probe:
    minimum_best_slack_s: int | None
    mean_best_slack_s: float | None
    mean_direct_arrival_slack_s: float | None
    compatible_vehicles: int
    stocked_vehicles: int
    free_crates: tuple[int, int]
    inspected: int
    feasible: int
    feasible_orders: int
    best_gain_mad: float | None
    estimated_cost_mad: float
    exhausted: bool
    best_insertion_slack_s: int | None


def probe(obs: Observation, forecast: Forecast, settings: Settings, rates: RewardRates, budget: int) -> Probe:
    """Evaluate individual fixed-order insertions with the public pre-dispatch forecast."""
    if budget < 1:
        raise ValueError("Insertion probe requires a positive limit")
    pending = obs.waiting_urgent
    views = [v for v in obs.vehicles if v.mutable]
    release = obs.now_s + settings.latency_s
    spare = tuple(sum(usable_stock(forecast, view, product)[2] for view in views) for product in PRODUCTS)
    delay_cost = sum(forecast.problem.vehicles[v.index].cents_per_hour
                     * max(0, release - max(obs.now_s, v.anchor_s)) / 360000
                     for v in views if v.phase != "staged")
    expense = rates.attempt_cents / 100 + delay_cost
    compatible, stocked, covered = set(), set(), set()
    slacks, arrivals, candidates = [], [], []
    incumbent = {}
    for key in pending:
        order = forecast.orders[key]
        direct = []
        for view in views:
            if not forecast.problem.vehicles[view.index].compatible(order.stop):
                continue
            compatible.add(view.index)
            if not can_supply(forecast, view, key):
                continue
            stocked.add(view.index)
            clock = max(view.anchor_s, release)
            leg = forecast.travel(view.anchor_node, order.node, clock)
            if leg is not None:
                arrival = clock + leg
                finish = max(arrival, order.stop.window[0]) + forecast.service(key)
                direct.append((order.stop.window[1] - finish, order.stop.window[1] - arrival))
            if obs.replan_available:
                candidates.extend((key, view, pos) for pos in range(len(view.tail) + 1))
        if direct:
            best = max(direct)
            slacks.append(best[0])
            arrivals.append(best[1])

    count = 0
    best_gain = -inf
    best_slack = None
    # Interleave urgent jobs so a bound does not allocate all work to the first request.
    queues = {key: [] for key in pending}
    for key, view, pos in candidates:
        queues[key].append((view, pos))
    numbered = sorted(((position, key, view, pos) for key, entries in queues.items()
                       for position, (view, pos) in enumerate(entries)), key=lambda item: item[0])
    for _, key, view, pos in numbered[:budget]:
        order = forecast.orders[key]
        tail = (*view.tail[:pos], key, *view.tail[pos:])
        result = project(forecast, view, tail, release)
        if result is None or result.overtime_s or any(d for stop, d in result.delays if stop != view.committed):
            continue
        count += 1
        covered.add(key)
        if view.index not in incumbent:
            incumbent[view.index] = project(forecast, view, view.tail, release)
        before = incumbent[view.index]
        saved_delay = sum(d for _, d in before.delays) - sum(d for _, d in result.delays)
        gain = ((before.cost_units - result.cost_units) / UNITS_PER_MAD
                + (saved_delay * rates.delay_cents_per_min + before.overtime_s * rates.overtime_cents_per_min) / 6000
                + (order.stop.penalty_cents - route_changes(view.tail, tail) * rates.change_cents) / 100
                - expense)
        if gain > best_gain:
            clock, node = max(view.anchor_s, release), view.anchor_node
            for stop in tail[:pos + 1]:
                visit = forecast.orders[stop]
                arrive = clock + forecast.travel(node, visit.node, clock)
                clock = max(arrive, visit.stop.window[0]) + forecast.service(stop)
                node = visit.node
            best_gain, best_slack = gain, order.stop.window[1] - clock

    return Probe(min(slacks, default=None), sum(slacks) / len(slacks) if slacks else None,
                 sum(arrivals) / len(arrivals) if arrivals else None,
                 len(compatible), len(stocked), spare, min(len(numbered), budget), count, len(covered),
                 best_gain if count else None, expense, len(numbered) > budget, best_slack)


def tabular(original: np.ndarray, measured: Probe, urgent: int, tight_s: int) -> np.ndarray:
    context, pressure, workload, recent = map(int, original)
    if context and urgent:
        if not measured.stocked_vehicles:
            context = 3
        elif not measured.feasible:
            context = 4
        else:
            comfortable = (measured.best_insertion_slack_s > tight_s and measured.best_gain_mad > 0
                           and measured.feasible_orders == urgent)
            context = 6 if comfortable else 5
    return np.array((context, pressure, workload, recent), dtype=np.int64)


def continuous(obs: Observation, measured: Probe, since_attempt_s: int | None) -> np.ndarray:
    def scaled(value, scale):
        return 0. if value is None else value / (abs(value) + scale)

    tails = [len(view.tail) for view in obs.vehicles if view.mutable]
    return np.array((
        obs.replan_available, scaled(obs.now_s, 43200), scaled(len(obs.waiting_urgent), 2),
        scaled(measured.minimum_best_slack_s, 900), scaled(measured.mean_best_slack_s, 900),
        scaled(measured.mean_direct_arrival_slack_s, 900), scaled(measured.compatible_vehicles, 3),
        scaled(measured.stocked_vehicles, 3), scaled(measured.free_crates[0], 12),
        scaled(measured.free_crates[1], 12), measured.feasible / max(1, measured.inspected),
        measured.exhausted, scaled(measured.best_gain_mad, 80), scaled(measured.estimated_cost_mad, 2),
        scaled(obs.predicted_delay_s, 600), scaled(obs.priority_at_risk, 2), scaled(sum(tails), 12),
        scaled(sum(t > 0 for t in tails), 3), scaled(obs.congestion_factor - 1, .5),
        scaled(since_attempt_s if since_attempt_s is not None else 7200, 900),
        scaled(obs.since_replan_s if obs.since_replan_s is not None else 7200, 900),
        measured.feasible_orders / max(1, len(obs.waiting_urgent)),
    ), dtype=np.float32)
