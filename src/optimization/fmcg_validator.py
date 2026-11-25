"""Independent time, inventory and cost replay for the FMCG problem."""

from dataclasses import dataclass

from src.optimization.fmcg_problem import FMCGProblem, Load, Vehicle, ZERO, nonnegative_integer, valid_load

UNITS_PER_CENT = 18000
UNITS_PER_MAD = UNITS_PER_CENT * 100


@dataclass(frozen=True)
class Tour:
    vehicle: int
    stops: tuple[int, ...]
    departure_s: int
    service_starts_s: tuple[int, ...]
    return_s: int
    initial_delivery: Load | None = None


@dataclass(frozen=True)
class FMCGResult:
    status: str
    tours: tuple[Tour, ...]
    elapsed_s: float
    detail: str = ""
    objective_units: int | None = None
    omissions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class VisitLoad:
    remaining_delivery: Load
    collected: Load
    onboard: Load


@dataclass(frozen=True)
class TourReport:
    distance_m: int
    duration_s: int
    waiting_s: int
    fixed_cost: int
    distance_cost: int
    time_cost: int
    loads: tuple[VisitLoad, ...]

    @property
    def cost(self) -> int:
        return self.fixed_cost + self.distance_cost + self.time_cost


@dataclass(frozen=True)
class PlanReport:
    tours: tuple[TourReport, ...]
    unserved: tuple[int, ...]
    penalty: int

    @property
    def cost(self) -> int:
        return sum(tour.cost for tour in self.tours) + self.penalty


def operating_cost(vehicle: Vehicle, distance_m: int, duration_s: int) -> tuple[int, int, int]:
    """Return exact activation, distance and time costs in scaled centimes."""
    return (vehicle.fixed_cents * UNITS_PER_CENT, distance_m * vehicle.cents_per_km * 18,
            duration_s * vehicle.cents_per_hour * 5)


def route_report(problem: FMCGProblem, tour: Tour) -> TourReport:
    """Replay a single active tour and reject every operational violation."""
    if (not nonnegative_integer(tour.vehicle) or tour.vehicle >= len(problem.vehicles)
            or not isinstance(tour.stops, tuple) or not tour.stops):
        raise ValueError("Invalid vehicle or empty active tour")
    if (any(not nonnegative_integer(i) or not 1 <= i <= len(problem.stops) for i in tour.stops)
            or len(set(tour.stops)) != len(tour.stops)):
        raise ValueError("Duplicate or unknown stops")
    vehicle = problem.vehicles[tour.vehicle]
    open_s, close_s = problem.operating_window(vehicle)
    if (not isinstance(tour.service_starts_s, tuple)
            or not all(map(nonnegative_integer, (tour.departure_s, tour.return_s, *tour.service_starts_s)))):
        raise ValueError("Tour times must be nonnegative integer seconds")
    if not open_s <= tour.departure_s <= tour.return_s <= close_s:
        raise ValueError("Depot or vehicle operating hours violated")
    stops = [problem.stops[i - 1] for i in tour.stops]
    if len(tour.service_starts_s) != len(stops):
        raise ValueError("Missing service times")
    remaining = tuple(sum(dimension) for dimension in zip(*(s.delivery for s in stops), strict=True))
    if tour.initial_delivery is not None and (
        not valid_load(tour.initial_delivery) or tour.initial_delivery != remaining
    ):
        raise ValueError("Declared initial delivery does not match assigned orders")
    collected = ZERO
    if any(load > cap for load, cap in zip(remaining, vehicle.capacity, strict=True)):
        raise ValueError("Initial load exceeds capacity")
    trace = [VisitLoad(remaining, collected, remaining)]
    clock, previous, distance, waiting = tour.departure_s, 0, 0, 0
    for node, stop, service_s in zip(tour.stops, stops, tour.service_starts_s, strict=True):
        travel = problem.travel_s[previous][node]
        if travel is None:
            raise ValueError("Unavailable directed arc")
        if not vehicle.compatible(stop):
            raise ValueError(f"Incompatible vehicle for {stop.id}")
        arrival = clock + travel
        clock = service_s + stop.service_s
        if service_s < max(arrival, stop.window[0]) or clock > stop.window[1]:
            raise ValueError(f"Service timing violated for {stop.id}")
        remaining = tuple(a - b for a, b in zip(remaining, stop.delivery, strict=True))
        collected = tuple(a + b for a, b in zip(collected, stop.pickup, strict=True))
        onboard = tuple(a + b for a, b in zip(remaining, collected, strict=True))
        if min(remaining) < 0 or any(load > cap for load, cap in zip(onboard, vehicle.capacity, strict=True)):
            raise ValueError(f"Inventory or pickup capacity violated for {stop.id}")
        trace.append(VisitLoad(remaining, collected, onboard))
        distance += problem.distance_m[previous][node]
        waiting += service_s - arrival
        previous = node
    back = problem.travel_s[previous][0]
    if back is None or tour.return_s != clock + back:
        raise ValueError("Invalid return leg or return time")
    distance += problem.distance_m[previous][0]
    duration = tour.return_s - tour.departure_s
    return TourReport(distance, duration, waiting, *operating_cost(vehicle, distance, duration), tuple(trace))


def validate_fmcg(problem: FMCGProblem, tours: tuple[Tour, ...]) -> PlanReport:
    """Reject duplicate visits and missing mandatory stops, and price optional omissions."""
    reports = tuple(route_report(problem, tour) for tour in tours)
    visited = [node for tour in tours for node in tour.stops]
    if len({tour.vehicle for tour in tours}) != len(tours) or len(set(visited)) != len(visited):
        raise ValueError("Duplicate vehicle or customer in plan")
    unserved = tuple(sorted(set(range(1, len(problem.stops) + 1)).difference(visited)))
    if any(problem.stops[i - 1].penalty_cents is None for i in unserved):
        raise ValueError("Missing mandatory stops")
    penalty = sum(problem.stops[i - 1].penalty_cents for i in unserved) * UNITS_PER_CENT
    return PlanReport(reports, unserved, penalty)
