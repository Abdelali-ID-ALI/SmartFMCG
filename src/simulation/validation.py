"""Independent checks for candidate dispatch plans and the realized physical ledger."""

from dataclasses import dataclass
from math import ceil

from src.optimization.fmcg_problem import FMCGProblem, Load, ZERO, nonnegative_integer, valid_load
from src.simulation.replanning import Forecast, Routes
from src.simulation.state import PRODUCTS, Congestion, Order, Scenario, ServiceDelay, VehicleView

Holdings = tuple[Load, Load, Load, Load]


def validate_routes(forecast: Forecast, views: tuple[VehicleView, ...], routes: Routes,
                    release_s: int, eligible: frozenset[str], *, check_windows: bool = True) -> None:
    """Validate coverage and independently replay modified tails from current holdings."""
    if len(routes) != len(views):
        raise ValueError("Expected one suffix per vehicle")
    current = {key for view in views for key in view.tail}
    committed = {view.committed for view in views if view.committed}
    proposed = [key for tail in routes for key in tail]
    if (len(proposed) != len(set(proposed)) or not current <= set(proposed)
            or set(proposed) & committed or not set(proposed) <= (current | eligible)):
        raise ValueError("A candidate duplicated, dropped or invented a visit")

    for view, route in zip(views, routes, strict=True):
        if not view.mutable and route != view.tail:
            raise ValueError("Cannot redirect a vehicle on its return leg")
        if check_windows and route == view.tail:
            continue
        vehicle = forecast.problem.vehicles[view.index]
        visits = ([view.committed] if view.committed else []) + list(route)
        delivered = [[0] * 4 for _ in range(3)]
        picked = [0] * 4
        stocks = (view.regular, *view.reserves)
        node, clock = view.anchor_node, max(release_s, view.anchor_s)
        for key in visits:
            order = forecast.orders[key]
            stop = order.stop
            if not vehicle.compatible(stop) or order.owner not in (None, view.index):
                raise ValueError("Vehicle access, cold chain or assigned stock violated")
            if key != view.committed:
                travel = forecast.travel(node, order.node, clock)
                if travel is None:
                    raise ValueError("Absent link in suffix")
                clock = max(clock + travel, stop.window[0]) + forecast.service(key)
                if check_windows and clock > stop.window[1]:
                    raise ValueError("Predicted customer window violated")
            node = order.node
            compartment = 0 if order.owner is not None else 1 + PRODUCTS.index(order.product)
            for d, capacity in enumerate(vehicle.capacity):
                delivered[compartment][d] += stop.delivery[d]
                picked[d] += stop.pickup[d]
                remaining = [stock[d] - used[d] for stock, used in zip(stocks, delivered, strict=True)]
                if min(remaining) < 0 or sum(remaining) + view.collected[d] + picked[d] > capacity:
                    raise ValueError("Candidate exceeds available stock or vehicle capacity")
        back = forecast.travel(node, 0, clock)
        if back is None or (check_windows and clock + back > forecast.problem.operating_window(vehicle)[1]):
            raise ValueError("Return arc or forecast operating hours violated")


@dataclass(frozen=True)
class Movement:
    vehicle: int
    kind: str
    time_s: int
    node: int
    order_id: str | None
    holdings: Holdings
    distance_m: int
    destination: int | None = None


def validate_movements(problem: FMCGProblem, initial: tuple[Holdings, ...],
                       movements: tuple[Movement, ...], orders: dict[str, Order],
                       releases: dict[str, int], scenario: Scenario | None = None) -> None:
    """Reconcile the journal without calling simulator transitions or candidate search."""
    states = list(initial)
    next_transition = ["depart"] * len(initial)
    nodes = [0] * len(initial)
    destinations = [None] * len(initial)
    commitments = [None] * len(initial)
    started = [0] * len(initial)
    distances = [0] * len(initial)
    seen = set()
    alerts = [event for event in scenario.events if isinstance(event, Congestion)] if scenario else []
    service_events = sorted((e for e in scenario.events if isinstance(e, ServiceDelay)),
                            key=lambda event: event.at_s) if scenario else []
    previous_s = 0
    if len(initial) != len(problem.vehicles):
        raise ValueError("Ledger has the wrong fleet size")
    for vehicle, stock in zip(problem.vehicles, initial, strict=True):
        if len(stock) != 4 or any(not valid_load(load) for load in stock) or any(
            sum(load[d] for load in stock) > limit for d, limit in enumerate(vehicle.capacity)
        ) or (not vehicle.refrigerated and stock[2] != ZERO) or stock[3] != ZERO:
            raise ValueError("Invalid initial cargo")
    for entry in movements:
        i, key = entry.vehicle, entry.order_id
        if not all(map(nonnegative_integer, (entry.time_s, i, entry.node, entry.distance_m))):
            raise ValueError("Invalid ledger timestamp, index or distance")
        if (i >= len(states) or entry.node >= len(problem.travel_s)
                or (key is not None and key not in orders)):
            raise ValueError("Unknown ledger vehicle, location or order")
        if (entry.time_s < previous_s or next_transition[i] == "done" or entry.kind != next_transition[i]
                or (entry.node != nodes[i] and (entry.kind != "arrive" or entry.node != destinations[i]))):
            raise ValueError("Broken vehicle chronology or position")
        previous_s = entry.time_s
        if entry.kind == "depart":
            if not problem.operating_window(problem.vehicles[i])[0] <= entry.time_s <= problem.operating_window(
                problem.vehicles[i]
            )[1]:
                raise ValueError("Departure outside operating hours")
            next_transition[i] = "leg"
        elif entry.kind == "leg":
            if (not nonnegative_integer(entry.destination) or entry.destination >= len(problem.travel_s)
                    or problem.travel_s[entry.node][entry.destination] is None
                    or (key is None and entry.destination != 0)
                    or (key is not None and (entry.destination != orders[key].node
                                             or entry.time_s < releases.get(key, 0)))):
                raise ValueError("Invalid committed destination")
            destinations[i], commitments[i], started[i] = entry.destination, key, entry.time_s
            next_transition[i] = "arrive"
        elif entry.kind == "arrive":
            if scenario is not None:
                departed = started[i]
                origin = nodes[i]
                actual = scenario.actual_s[min(23, departed // 3600)][origin][entry.node]
                if not nonnegative_integer(actual):
                    raise ValueError("Invalid realized travel duration")
                factor = max((a.factor for a in alerts if a.at_s <= departed < a.end_s
                              and origin != entry.node and (origin in a.nodes or entry.node in a.nodes)),
                             default=1.)
                try:
                    expected_arrival = departed + ceil(actual * factor)
                except OverflowError as exc:
                    raise ValueError("Congestion produces a nonfinite realized travel duration") from exc
                if entry.time_s != expected_arrival:
                    raise ValueError("Travel did not use realized conditions at departure")
            if entry.node != destinations[i] or key != commitments[i] or (
                entry.time_s - started[i] < int(problem.travel_s[nodes[i]][entry.node] > 0)
            ):
                raise ValueError("Arrival broke a committed leg")
            distances[i] += problem.distance_m[nodes[i]][entry.node]
            nodes[i] = entry.node
            started[i] = entry.time_s
            next_transition[i] = "unload" if entry.node == 0 else "service_start"
        elif entry.kind == "service_start":
            if key != commitments[i] or entry.time_s != max(started[i], orders[key].stop.window[0]):
                raise ValueError("Service started before opening or at an uncommitted stop")
            started[i] = entry.time_s
            next_transition[i] = "service_end"
        elif entry.kind == "service_end":
            order = orders[key]
            stop = order.stop
            if scenario is not None:
                finish = started[i] + stop.service_s
                for event in service_events:
                    if event.order_id == key and event.at_s < finish:
                        finish += event.extra_s
                if entry.time_s != finish:
                    raise ValueError("Service did not use events released before completion")
            if (key in seen or key != commitments[i] or entry.time_s < started[i] + stop.service_s
                    or not problem.vehicles[i].compatible(stop) or order.owner not in (None, i)):
                raise ValueError("Invalid or duplicated service")
            compartment = 0 if order.owner is not None else 1 + PRODUCTS.index(order.product)
            expected = list(states[i])
            expected[compartment] = tuple(a - b for a, b in zip(expected[compartment], stop.delivery, strict=True))
            expected[3] = tuple(a + b for a, b in zip(expected[3], stop.pickup, strict=True))
            states[i] = tuple(expected)
            seen.add(key)
            next_transition[i] = "leg"
        else:
            if entry.time_s != started[i] or key is not None:
                raise ValueError("Unloading must coincide with the depot return")
            if states[i][0] != ZERO:
                raise ValueError("Regular delivery inventory returned to depot")
            # The depot receives precisely the holdings in the preceding arrival record.
            states[i] = (ZERO, ZERO, ZERO, ZERO)
            next_transition[i] = "done"
        if (states[i] != entry.holdings or distances[i] != entry.distance_m
                or any(not valid_load(load) for load in entry.holdings)
                or any(sum(load[d] for load in entry.holdings) > limit
                       for d, limit in enumerate(problem.vehicles[i].capacity))):
            raise ValueError("Realized inventory, pickup capacity or distance failed reconciliation")
    if any(state not in {"depart", "done"} for state in next_transition) or any(
        next_transition[order.owner] != "done" or key not in seen
        for key, order in orders.items() if order.owner is not None
    ):
        raise ValueError("Incomplete execution ledger")
