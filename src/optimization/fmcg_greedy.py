"""Cheapest feasible insertion for the same static FMCG constraints."""

from time import perf_counter

from src.optimization.fmcg_problem import FMCGProblem
from src.optimization.fmcg_validator import FMCGResult, Tour, UNITS_PER_CENT, route_report, validate_fmcg


def schedule(problem: FMCGProblem, vehicle: int, stops: tuple[int, ...]) -> Tour:
    """Construct the shortest-span timetable for a fixed sequence."""
    opening, closing = problem.operating_window(problem.vehicles[vehicle])

    def forward(departure: int) -> tuple[tuple[int, ...], int, int]:
        starts = []
        clock, previous, waiting = departure, 0, 0
        for node in stops:
            travel = problem.travel_s[previous][node]
            if travel is None:
                raise ValueError("Unavailable directed arc")
            arrival = clock + travel
            stop = problem.stops[node - 1]
            start = max(arrival, stop.window[0])
            starts.append(start)
            waiting += start - arrival
            clock, previous = start + stop.service_s, node
        back = problem.travel_s[previous][0]
        if back is None:
            raise ValueError("Unavailable return arc")
        return tuple(starts), clock + back, waiting

    _, _, waiting = forward(opening)
    latest, following = closing, 0
    for node in reversed(stops):
        stop = problem.stops[node - 1]
        latest = min(latest - problem.travel_s[node][following], stop.window[1]) - stop.service_s
        following = node
    latest_departure = latest - problem.travel_s[0][following]
    if latest_departure < opening:
        raise ValueError("No timetable meets all service windows and operating hours")
    departure = min(opening + waiting, latest_departure)
    starts, end, _ = forward(departure)
    return Tour(vehicle, stops, departure, starts, end)


def cheapest_insertion(problem: FMCGProblem) -> FMCGResult:
    """Attempt mandatory stops first, then insert optional stops when worth serving."""
    start = perf_counter()
    chosen: dict[int, tuple[Tour, int]] = {}
    omissions = []
    nodes = sorted(range(1, len(problem.stops) + 1),
                   key=lambda i: (problem.stops[i-1].penalty_cents is not None, problem.stops[i-1].window[1]))
    for node in nodes:
        stop = problem.stops[node - 1]
        best = None
        failures: set[str] = set()
        for vehicle in range(len(problem.vehicles)):
            current, cost = chosen.get(vehicle, (None, 0))
            sequence = current.stops if current else ()
            for position in range(len(sequence) + 1):
                try:
                    tour = schedule(problem, vehicle, (*sequence[:position], node, *sequence[position:]))
                    candidate_cost = route_report(problem, tour).cost
                except ValueError as exc:
                    failures.add(str(exc))
                    continue
                delta = candidate_cost - cost
                if best is None or delta < best[0]:
                    best = delta, tour, candidate_cost
        penalty = stop.penalty_cents
        if best is None:
            omissions.append((stop.id, "No feasible insertion: " + "; ".join(sorted(failures))))
        elif penalty is not None and best[0] > penalty * UNITS_PER_CENT:
            omissions.append((stop.id, "Cheapest feasible insertion costs more than the omission penalty"))
        else:
            _, tour, cost = best
            chosen[tour.vehicle] = tour, cost
    tours = tuple(tour for _, (tour, _) in sorted(chosen.items()))
    mandatory = {stop.id for stop in problem.stops if stop.penalty_cents is None}
    if any(identifier in mandatory for identifier, _ in omissions):
        return FMCGResult("no_solution", tours, perf_counter() - start,
                          "Greedy left mandatory stops unserved", omissions=tuple(omissions))
    report = validate_fmcg(problem, tours)
    return FMCGResult("feasible", tours, perf_counter() - start, "Cheapest feasible insertion",
                      report.cost, tuple(omissions))
