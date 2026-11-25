"""OR-Tools model for simultaneous delivery/pickup and time windows."""

from collections.abc import Callable
from itertools import permutations, product
from math import isfinite
from time import perf_counter

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.optimization.fmcg_greedy import schedule
from src.optimization.fmcg_problem import FMCGProblem, ZERO
from src.optimization.fmcg_validator import FMCGResult, Tour, UNITS_PER_CENT, route_report, validate_fmcg


def solve_fmcg(problem: FMCGProblem, *, time_limit_s: float = 2.0,
               initial_tours: tuple[Tour, ...] | None = None,
               on_incumbent: Callable[[float], None] | None = None) -> FMCGResult:
    """Return a bounded-search incumbent with independently reconciled costs."""
    if not isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("Expected a finite positive time limit")
    started = perf_counter()
    deadline = started + time_limit_s
    fleet = problem.vehicles
    eligibility = [
        [i for i, vehicle in enumerate(fleet)
         if vehicle.compatible(stop) and (window := problem.operating_window(vehicle))[0] <= window[1]
         and max(window[0], stop.window[0]) + stop.service_s <= min(window[1], stop.window[1])
         and all(max(d, p) <= c for d, p, c in zip(stop.delivery, stop.pickup, vehicle.capacity, strict=True))]
        for stop in problem.stops
    ]
    for stop, eligible in zip(problem.stops, eligibility, strict=True):
        if stop.penalty_cents is None and not eligible:
            return FMCGResult("infeasible", (), perf_counter() - started, f"No compatible capacity/time for {stop.id}")

    manager = pywrapcp.RoutingIndexManager(len(problem.stops) + 1, len(fleet), 0)
    routing = pywrapcp.RoutingModel(manager)
    node = manager.IndexToNode
    solver = routing.solver()
    ends = [routing.End(v) for v in range(len(fleet))]
    starts = [routing.Start(v) for v in range(len(fleet))]
    for i in range(routing.Size()):
        for j in (*range(routing.Size()), *ends):
            if problem.travel_s[node(i)][node(j)] is None:
                routing.NextVar(i).RemoveValue(j)

    service = (0, *(stop.service_s for stop in problem.stops))

    def time_callback(i: int, j: int) -> int:
        return service[node(i)] + (problem.travel_s[node(i)][node(j)] or 0)

    horizon = 86400
    routing.AddDimension(routing.RegisterTransitCallback(time_callback), horizon, horizon, False, "Time")
    time = routing.GetDimensionOrDie("Time")
    for v, vehicle in enumerate(fleet):
        opening, closing = problem.operating_window(vehicle)
        start, end = starts[v], ends[v]
        if opening > closing:
            solver.Add(routing.NextVar(start) == end)
            opening = closing = 0
        time.CumulVar(start).SetRange(opening, closing)
        time.CumulVar(end).SetRange(opening, closing)
        routing.AddVariableMaximizedByFinalizer(time.CumulVar(start))
        routing.AddVariableMinimizedByFinalizer(time.CumulVar(end))

        def distance_cost(i: int, j: int, rate: int = vehicle.cents_per_km) -> int:
            return (problem.distance_m[node(i)][node(j)] or 0) * rate * 18

        routing.SetArcCostEvaluatorOfVehicle(routing.RegisterTransitCallback(distance_cost), v)
        routing.SetFixedCostOfVehicle(vehicle.fixed_cents * UNITS_PER_CENT, v)
        time.SetSpanCostCoefficientForVehicle(vehicle.cents_per_hour * 5, v)

    for stop_number, (stop, eligible) in enumerate(zip(problem.stops, eligibility, strict=True), start=1):
        index = manager.NodeToIndex(stop_number)
        if stop.penalty_cents is not None:
            routing.AddDisjunction([index], stop.penalty_cents * UNITS_PER_CENT)
        if not eligible:
            routing.ActiveVar(index).SetValue(0)
            continue
        for v in range(len(fleet)):
            if v not in eligible:
                routing.VehicleVar(index).RemoveValue(v)
        time.CumulVar(index).SetRange(stop.window[0], stop.window[1] - stop.service_s)

    deliveries = (ZERO, *(stop.delivery for stop in problem.stops))
    pickups = (ZERO, *(stop.pickup for stop in problem.stops))
    onboard_dimensions = []
    for dim in range(len(ZERO)):
        capacities = [v.capacity[dim] for v in fleet]

        def delivery(i: int, dim: int = dim) -> int:
            return deliveries[node(i)][dim]

        def net_change(i: int, dim: int = dim) -> int:
            return pickups[node(i)][dim] - deliveries[node(i)][dim]

        delivered_name, onboard_name = f"Delivered{dim}", f"Onboard{dim}"
        routing.AddDimensionWithVehicleCapacity(routing.RegisterUnaryTransitCallback(delivery),
                                                0, capacities, True, delivered_name)
        routing.AddDimensionWithVehicleCapacity(routing.RegisterUnaryTransitCallback(net_change),
                                                0, capacities, False, onboard_name)
        delivered = routing.GetDimensionOrDie(delivered_name)
        onboard = routing.GetDimensionOrDie(onboard_name)
        onboard_dimensions.append(onboard)
        for start, end in zip(starts, ends, strict=True):
            solver.Add(onboard.CumulVar(start) == delivered.CumulVar(end))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.sat_parameters.num_workers = 1
    # Keep the existing sparse budget split and let very small dense cases
    # use time left after a native search failure.
    small_sparse = (len(problem.stops) <= 5 and len(fleet) <= 4
                    and any(None in row for row in problem.travel_s))
    allow_construction = small_sparse or (initial_tours is None and len(problem.stops) <= 2 and len(fleet) <= 4)
    if on_incumbent is not None:
        routing.AddAtSolutionCallback(lambda: on_incumbent(perf_counter() - started))
    remaining_ms = int((deadline - perf_counter()) * 1000)
    if remaining_ms < 1:
        return FMCGResult("timeout", (), perf_counter() - started, "Budget exhausted during model construction")
    assignment = None
    if initial_tours is not None:
        params.time_limit.FromMilliseconds(remaining_ms)
        validate_fmcg(problem, initial_tours)
        routes = {tour.vehicle: tour.stops for tour in initial_tours}
        routing.CloseModelWithParameters(params)
        assignment = routing.ReadAssignmentFromRoutes([list(routes.get(v, ())) for v in range(len(fleet))], False)
        remaining_ms = int((deadline - perf_counter()) * 1000)
        if remaining_ms < 1:
            return FMCGResult("timeout", (), perf_counter() - started, "Budget exhausted during warm-start import")
        if assignment is None:
            return FMCGResult("no_solution", (), perf_counter() - started, "Warm start rejected by OR-Tools")
    params.time_limit.FromMilliseconds(max(1, remaining_ms // 2 if small_sparse else remaining_ms))
    solution = (routing.SolveWithParameters(params) if assignment is None
                else routing.SolveFromAssignmentWithParameters(assignment, params))
    native = routing_enums_pb2.RoutingSearchStatus.Value.Name(routing.status())
    if solution is None:
        status = {"ROUTING_INFEASIBLE": "infeasible", "ROUTING_FAIL_TIMEOUT": "timeout",
                  "ROUTING_INVALID": "invalid"}.get(native, "no_solution")
        if allow_construction and status in {"timeout", "no_solution"}:
            tours = _bounded_construction(problem, eligibility, deadline)
            if tours is not None:
                report = validate_fmcg(problem, tours)
                omissions = tuple((problem.stops[i - 1].id, "Omitted in bounded construction fallback")
                                  for i in report.unserved)
                return FMCGResult("feasible", tours, perf_counter() - started,
                                  f"{native}; bounded construction fallback", report.cost, omissions)
            if perf_counter() >= deadline:
                status = "timeout"
        return FMCGResult(status, (), perf_counter() - started, native)

    tours = []
    for v, (start, end) in enumerate(zip(starts, ends, strict=True)):
        index, visits, service_starts = solution.Value(routing.NextVar(start)), [], []
        while not routing.IsEnd(index):
            visits.append(node(index))
            service_starts.append(solution.Min(time.CumulVar(index)))
            index = solution.Value(routing.NextVar(index))
        if visits:
            departure = solution.Min(time.CumulVar(start))
            # No artificial waiting is needed at the depot on the last leg.
            returned = service_starts[-1] + service[visits[-1]] + problem.travel_s[visits[-1]][0]
            initial = tuple(solution.Min(dimension.CumulVar(start)) for dimension in onboard_dimensions)
            tours.append(Tour(v, tuple(visits), departure, tuple(service_starts), returned, initial))
    tours = tuple(tours)
    report = validate_fmcg(problem, tours)
    if report.cost != solution.ObjectiveValue():
        raise RuntimeError(f"OR-Tools cost {solution.ObjectiveValue()} differs from validated cost {report.cost}")
    omissions = tuple((problem.stops[i - 1].id,
                       "No individually eligible vehicle" if not eligibility[i - 1]
                       else "Optional omission in the best plan found; global cause undetermined")
                      for i in report.unserved)
    status = "optimal" if native == "ROUTING_OPTIMAL" else "feasible"
    return FMCGResult(status, tours, perf_counter() - started, native, report.cost, omissions)



def _bounded_construction(problem: FMCGProblem, eligibility: list[list[int]],
                          deadline: float) -> tuple[Tour, ...] | None:
    """Find a feasible witness within the original search deadline."""
    omit = len(problem.vehicles)
    choices = [(*eligible, omit) if stop.penalty_cents is not None else tuple(eligible)
               for stop, eligible in zip(problem.stops, eligibility, strict=True)]
    for assignment in product(*choices):
        if perf_counter() >= deadline:
            break
        tours = []
        for vehicle in range(omit):
            nodes = tuple(i for i, assigned in enumerate(assignment, start=1) if assigned == vehicle)
            if not nodes:
                continue
            for ordering in permutations(nodes):
                if perf_counter() >= deadline:
                    return None
                try:
                    candidate = schedule(problem, vehicle, ordering)
                    route_report(problem, candidate)
                except ValueError:
                    continue
                tours.append(candidate)
                break
            else:
                break
        else:
            return tuple(tours)
    return None
