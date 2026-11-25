"""Generate heterogeneous, independently checkable scalability instances."""

from dataclasses import dataclass, replace
from math import ceil

import numpy as np

from src.benchmark.scenarios import Design, Draw, Reserve, external_events, planning_capacity
from src.optimization.fmcg_greedy import schedule
from src.optimization.fmcg_problem import FMCGProblem, Stop, ZERO
from src.optimization.fmcg_validator import Tour, route_report, validate_fmcg
from src.simulation.experiment import beverage_load
from src.simulation.state import AMBIENT, CHILLED, ServiceDelay, plus


@dataclass(frozen=True)
class Instance:
    problem: FMCGProblem
    effective: FMCGProblem
    reserves: Reserve
    witness: tuple[Tour, ...]
    products: dict[str, str]
    priority: frozenset[str]


def stops_for(ids: tuple[str, ...], base: FMCGProblem, seed: int, window_s: int) -> tuple[Stop, ...]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1001]))
    stops = []
    for i, key in enumerate(ids[1:]):
        donor = base.stops[i % len(base.stops)]
        delivered = max(1, round(donor.delivery[2] * rng.uniform(.82, 1.08)))
        picked = round(donor.pickup[2] * rng.uniform(.75, 1.20))
        close = min(donor.window[0] + window_s, base.depot_window[1] - 1800)
        cargo = (*beverage_load(delivered)[:3], ceil(delivered / 40) if donor.delivery[3] else 0)
        stops.append(replace(donor, id=key, delivery=cargo, pickup=(picked * 2000, picked * 20, picked, 0),
                             window=(donor.window[0], close), service_s=480 + 60 * ceil(delivered / 6),
                             penalty_cents=None))
    return tuple(stops)


def size_fleet(planned: FMCGProblem, zones: dict[str, str], reserve_crates: int, target_load: float,
               max_route_stops: int, standby_fraction: float) -> Instance:
    """Size routes against service clocks and the full physical load trace."""
    if not 0 < target_load <= 1 or max_route_stops < 1 or reserve_crates < 0 or standby_fraction < 0:
        raise ValueError("Invalid fleet-sizing parameters")
    templates = planned.vehicles
    ready = beverage_load(reserve_crates)
    reserves = []
    fleet = []
    witness = []
    candidates = []
    for vehicle in templates:
        stock = (ZERO, ready) if vehicle.refrigerated else (ready, ZERO)
        reserved = plus(*stock)
        capacity = tuple(cap - extra for cap, extra in zip(vehicle.capacity, reserved, strict=True))
        target = tuple(ceil(cap * target_load) - extra
                       for cap, extra in zip(vehicle.capacity, reserved, strict=True))
        if min(capacity) < 0 or min(target) < 0:
            continue
        effective = replace(planned, vehicles=(replace(vehicle, capacity=capacity),))
        preferred = replace(planned, vehicles=(replace(vehicle, capacity=target),))
        candidates.append((vehicle, stock, effective, preferred))
    order = sorted(range(1, len(planned.stops) + 1), key=lambda i: (
        not planned.stops[i - 1].refrigerated, zones[planned.stops[i - 1].id],
        planned.stops[i - 1].window[1], -planned.stops[i - 1].delivery[0]))
    # Try nearby pending stops first. Each witness route is short by construction.
    while order:
        first, *pending = order
        stop = planned.stops[first - 1]
        best = None
        for vehicle, stock, effective, preferred in candidates:
            if not vehicle.compatible(stop) or vehicle.refrigerated != stop.refrigerated:
                continue
            under_target = all(max(d, p) <= cap for d, p, cap in zip(
                stop.delivery, stop.pickup, preferred.vehicles[0].capacity, strict=True))
            probe = preferred if under_target else effective
            try:
                tour = schedule(probe, 0, (first,))
                route_report(probe, tour)
            except ValueError:
                continue
            for node in pending:
                next_stop = planned.stops[node - 1]
                if len(tour.stops) >= max_route_stops:
                    break
                if next_stop.refrigerated != stop.refrigerated or not vehicle.compatible(next_stop):
                    continue
                try:
                    candidate = schedule(probe, 0, (*tour.stops, node))
                    route_report(probe, candidate)
                except ValueError:
                    continue
                tour = candidate
            score = (-len(tour.stops), vehicle.fixed_cents, vehicle.id)
            if best is None or score < best[0]:
                best = score, vehicle, stock, tour
        if best is None:
            raise ValueError(f"No feasible constructive route for {stop.id}")
        _, vehicle, stock, tour = best
        index = len(fleet)
        fleet.append(replace(vehicle, id=f"{vehicle.id}-{index + 1:04d}"))
        reserves.append(stock)
        witness.append(replace(tour, vehicle=index))
        assigned = set(tour.stops)
        order = [node for node in pending if node not in assigned]
    standby = ceil(len(fleet) * standby_fraction)
    for i in range(standby):
        vehicle = templates[i % len(templates)]
        fleet.append(replace(vehicle, id=f"{vehicle.id}-{len(fleet) + 1:04d}"))
        reserves.append((ZERO, ZERO))
    physical = replace(planned, vehicles=tuple(fleet))
    effective = planning_capacity(physical, tuple(reserves))
    witness = tuple(witness)
    validate_fmcg(effective, witness)
    validate_fmcg(physical, witness)
    return Instance(physical, effective, tuple(reserves), witness,
                    {s.id: CHILLED if s.refrigerated else AMBIENT for s in physical.stops},
                    frozenset(s.id for i, s in enumerate(physical.stops) if i % 5 == 1 or s.refrigerated))


def events_for(problem: FMCGProblem, draw: Draw, ids: tuple[str, ...], zones: dict[str, str],
               service_events: int, urgent_service_s: int):
    """Reuse the phase-ten exogenous draws, adding independently sampled service delays."""
    events = list(external_events(problem, draw, ids, zones, urgent_service_s))
    if service_events > 1:
        rng = np.random.default_rng(np.random.SeedSequence([draw.seed, 1004]))
        delayed = {event.order_id for event in events if isinstance(event, ServiceDelay)}
        others = [stop for stop in problem.stops if stop.id not in delayed]
        for index in rng.choice(len(others), size=min(service_events - 1, len(others)), replace=False):
            stop = others[index]
            at = stop.window[0] + int(rng.integers(0, 2701))
            extra = round(draw.design.service_extra_s * rng.uniform(.85, 1.15))
            events.append(ServiceDelay(at, stop.id, extra))
    return tuple(sorted(events, key=lambda event: event.at_s))


def design_for(regime: dict, customers: int) -> tuple[Design, int]:
    urgents = ceil(customers * regime["urgent_rate"])
    design = Design(1., 1., 0, regime["reserve_crates"], None, regime["traffic_factor"],
                    regime["service_extra_s"], urgents, regime["urgent_window_s"], tuple(regime["urgent_crates"]))
    return design, ceil(customers * regime["service_event_rate"])
