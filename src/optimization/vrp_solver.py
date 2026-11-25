"""Translate the classical CVRP to OR-Tools RoutingModel."""

from math import isfinite
from time import perf_counter

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.optimization.problem import CVRP, RoutingResult


def solve(problem: CVRP, *, first: str = "PATH_CHEAPEST_ARC",
          search: str = "GUIDED_LOCAL_SEARCH", time_limit_s: float = 1.0) -> RoutingResult:
    """Return the incumbent and the native status within a bounded CPU search."""
    if not isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("Expected a finite positive time limit")
    start = perf_counter()
    if reason := problem.impossible_reason():
        return RoutingResult("infeasible", (), perf_counter() - start, reason)

    manager = pywrapcp.RoutingIndexManager(len(problem.ids), problem.vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)
    distances = problem.distance_m

    def arc_cost(i: int, j: int) -> int:
        value = distances[manager.IndexToNode(i)][manager.IndexToNode(j)]
        return 0 if value is None else value

    # End indices are outside RoutingModel.Size() and share the depot node.
    destinations = [*range(routing.Size()), *(routing.End(v) for v in range(problem.vehicle_count))]
    for i in range(routing.Size()):
        for j in destinations:
            if distances[manager.IndexToNode(i)][manager.IndexToNode(j)] is None:
                routing.NextVar(i).RemoveValue(j)

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(arc_cost))
    demand = routing.RegisterUnaryTransitCallback(lambda i: problem.demand_kg[manager.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(demand, 0, [problem.capacity_kg] * problem.vehicle_count, True, "Weight")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.Value.Value(first)
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.Value.Value(search)
    params.sat_parameters.num_workers = 1
    params.time_limit.FromMilliseconds(max(1, round(1000 * time_limit_s)))
    solution = routing.SolveWithParameters(params)
    native = routing_enums_pb2.RoutingSearchStatus.Value.Name(routing.status())
    if solution is None:
        status = {
            "ROUTING_INFEASIBLE": "infeasible",
            "ROUTING_FAIL_TIMEOUT": "timeout",
            "ROUTING_INVALID": "invalid",
        }.get(native, "no_solution")
        return RoutingResult(status, (), perf_counter() - start, native)

    routes = []
    for vehicle in range(problem.vehicle_count):
        index = solution.Value(routing.NextVar(routing.Start(vehicle)))
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        if route:
            routes.append(tuple(route))
    status = "optimal" if native == "ROUTING_OPTIMAL" else "feasible"
    return RoutingResult(status, tuple(routes), perf_counter() - start, native)
