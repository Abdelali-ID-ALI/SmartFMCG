"""Deterministic nearest-neighbour construction."""

from time import perf_counter

from src.optimization.problem import CVRP, RoutingResult


def nearest_neighbour(problem: CVRP) -> RoutingResult:
    """Extend each route while capacity and a direct depot return are available."""
    start = perf_counter()
    if reason := problem.impossible_reason():
        return RoutingResult("infeasible", (), perf_counter() - start, reason)
    remaining = set(range(1, len(problem.ids)))
    routes = []
    for _ in range(problem.vehicle_count):
        route = []
        current, capacity = 0, problem.capacity_kg
        while remaining:
            options = [
                (distance, node) for node in remaining
                if problem.demand_kg[node] <= capacity
                and (distance := problem.distance_m[current][node]) is not None
                and problem.distance_m[node][0] is not None
            ]
            if not options:
                break
            _, current = min(options)
            capacity -= problem.demand_kg[current]
            route.append(current)
            remaining.remove(current)
        if route:
            routes.append(tuple(route))
        if not remaining:
            break
    detail = ""
    if remaining:
        ids = ", ".join(problem.ids[node] for node in sorted(remaining))
        detail = f"Greedy exhausted capacity or available extensions; unserved: {ids}"
    return RoutingResult("no_solution" if remaining else "feasible", tuple(routes),
                         perf_counter() - start, detail)
