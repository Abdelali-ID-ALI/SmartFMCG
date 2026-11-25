"""Bounded insertion and relocation on vehicle suffixes with observed forecasts."""

from collections.abc import Mapping
from dataclasses import dataclass

from src.optimization.fmcg_problem import FMCGProblem
from src.optimization.fmcg_validator import UNITS_PER_CENT, operating_cost
from src.simulation.policies import Action
from src.simulation.state import (PRODUCTS, Congestion, Order, Settings, VehicleView, congested_seconds,
                                  minus, plus, within)

Routes = tuple[tuple[str, ...], ...]


def affected(alert: Congestion, origin: int, destination: int, now_s: int) -> bool:
    return (origin != destination and alert.at_s <= now_s < alert.end_s
            and (origin in alert.nodes or destination in alert.nodes))


@dataclass(frozen=True)
class Forecast:
    problem: FMCGProblem
    orders: Mapping[str, Order]
    alerts: tuple[Congestion, ...]
    delays: Mapping[str, int]

    def travel(self, origin: int, destination: int, departure_s: int) -> int | None:
        seconds = self.problem.travel_s[origin][destination]
        factor = max((alert.factor for alert in self.alerts if affected(alert, origin, destination, departure_s)),
                     default=1.)
        return None if seconds is None else congested_seconds(seconds, factor)

    def service(self, key: str) -> int:
        return self.orders[key].stop.service_s + self.delays.get(key, 0)


@dataclass(frozen=True)
class Projection:
    delays: tuple[tuple[str, int], ...]
    overtime_s: int
    cost_units: int


def project(forecast: Forecast, view: VehicleView, tail: tuple[str, ...], release_s: int) -> Projection | None:
    """Screen a suffix, allowing projected lateness when scoring the incumbent."""
    vehicle = forecast.problem.vehicles[view.index]
    clock, origin = max(view.anchor_s, release_s), view.anchor_node
    travel_m = 0
    regular, reserves, collected = view.regular, list(view.reserves), view.collected
    delays = []
    visits = (*((view.committed,) if view.committed else ()), *tail)
    for key in visits:
        order = forecast.orders[key]
        stop = order.stop
        if not vehicle.compatible(stop):
            return None
        if key != view.committed:
            travel = forecast.travel(origin, order.node, clock)
            if travel is None:
                return None
            travel_m += forecast.problem.distance_m[origin][order.node]
            clock = max(clock + travel, stop.window[0]) + forecast.service(key)
            delays.append((key, max(0, clock - stop.window[1])))
        else:
            delays.append((key, max(0, view.anchor_s - stop.window[1])))
        origin = order.node
        if order.owner is None:
            sku = PRODUCTS.index(order.product)
            reserves[sku] = minus(reserves[sku], stop.delivery)
        elif order.owner == view.index:
            regular = minus(regular, stop.delivery)
        else:
            return None
        collected = plus(collected, stop.pickup)
        if min(*regular, *reserves[0], *reserves[1]) < 0 or not within(
            plus(plus(regular, collected), plus(*reserves)), vehicle.capacity
        ):
            return None

    travel = forecast.travel(origin, 0, clock)
    if travel is None:
        return None
    travel_m += forecast.problem.distance_m[origin][0]
    finish = clock + travel
    cost = operating_cost(vehicle, travel_m, finish - max(view.anchor_s, release_s))
    return Projection(tuple(delays), max(0, finish - forecast.problem.operating_window(vehicle)[1]), sum(cost[1:]))


def route_changes(before: tuple[str, ...], after: tuple[str, ...]) -> int:
    """Count inserted or removed visits and changed delivery predecessors."""
    old = dict(zip(before, (None, *before)))
    new = dict(zip(after, (None, *after)))
    return len(old.keys() ^ new.keys()) + sum(old[key] != new[key] for key in old.keys() & new.keys())


@dataclass(frozen=True)
class Proposal:
    routes: Routes
    candidates: int
    solver_calls: int = 0
    solver_status: str = ""


def propose(forecast: Forecast, views: tuple[VehicleView, ...], urgent: tuple[str, ...],
            action: Action, release_s: int, settings: Settings) -> Proposal:
    original = tuple(view.tail for view in views)
    routes = list(original)
    active = [view.index for view in views if view.mutable]
    current = {i: project(forecast, views[i], original[i], release_s) for i in active}
    if any(report is None for report in current.values()):
        raise RuntimeError("Infeasible physical incumbent")
    if not active:
        return Proposal(original, 0)
    if action == Action.LOCAL:
        active = [max(active, key=lambda i: sum(delay for _, delay in current[i].delays)
                      + current[i].overtime_s)]

    def score(index: int, tail: tuple[str, ...], report: Projection) -> int:
        urgency = sum(forecast.orders[key].stop.penalty_cents for key in tail
                      if forecast.orders[key].owner is None)
        delay = sum(seconds for _, seconds in report.delays)
        return (report.cost_units
                + (settings.delay_cents_per_min * delay
                   + settings.overtime_cents_per_min * report.overtime_s) * (UNITS_PER_CENT // 60)
                + (settings.change_cents * route_changes(original[index], tail) - urgency) * UNITS_PER_CENT)

    inspected = 0

    def improvement(index: int, tail: tuple[str, ...]) -> tuple[int, Projection] | None:
        nonlocal inspected
        inspected += 1
        report = project(forecast, views[index], tail, release_s)
        if report is None or report.overtime_s or any(
            delay for key, delay in report.delays if key != views[index].committed
        ):
            return None
        delta = score(index, tail, report) - score(index, routes[index], current[index])
        return (delta, report) if delta < 0 else None

    # Select a feasible position and vehicle for each newly available order.
    for key in urgent:
        best = None
        order = forecast.orders[key]
        for i in active:
            if (not forecast.problem.vehicles[i].compatible(order.stop)
                    or not within(order.stop.delivery, views[i].reserves[PRODUCTS.index(order.product)])):
                continue
            for pos in range(len(routes[i]) + 1):
                if inspected >= settings.candidate_limit:
                    break
                tail = (*routes[i][:pos], key, *routes[i][pos:])
                result = improvement(i, tail)
                if result and (best is None or result[0] < best[0]):
                    best = result[0], result[1], i, tail
        if best:
            _, report, i, tail = best
            routes[i], current[i] = tail, report

    # Best single relocation per selected vehicle, prioritizing forecast pressure.
    for i in sorted(active, key=lambda i: -sum(delay for _, delay in current[i].delays)):
        best = None
        for start, key in enumerate(routes[i]):
            remainder = routes[i][:start] + routes[i][start + 1:]
            for end in range(len(routes[i])):
                if inspected >= settings.candidate_limit:
                    break
                if end == start:
                    continue
                tail = (*remainder[:end], key, *remainder[end:])
                result = improvement(i, tail)
                if result and (best is None or result[0] < best[0]):
                    best = result[0], result[1], tail
        if best:
            _, current[i], routes[i] = best
    return Proposal(tuple(routes), inspected)
