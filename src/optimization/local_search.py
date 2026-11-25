"""Feasibility-filtered directed 2-opt, swap and relocate search."""

from collections.abc import Iterator
from math import isfinite
from time import perf_counter

from src.optimization.problem import CVRP, Routes, RoutingResult
from src.optimization.solution_validator import validate


def neighbours(routes: Routes, vehicle_count: int) -> Iterator[Routes]:
    """Generate intra-route reversals and swaps and inter-route swaps and relocations."""
    slots = routes + ((),) * (len(routes) < vehicle_count)
    for i, first in enumerate(routes):
        for a in range(len(first)):
            for b in range(a + 1, len(first)):
                yield (*routes[:i], first[:a] + first[a:b+1][::-1] + first[b+1:], *routes[i+1:])
                swapped = list(first)
                swapped[a], swapped[b] = swapped[b], swapped[a]
                yield (*routes[:i], tuple(swapped), *routes[i+1:])
            for j, second in enumerate(slots):
                if i == j:
                    continue
                for b in range(len(second) + 1):
                    candidate = list(slots)
                    candidate[i] = first[:a] + first[a+1:]
                    candidate[j] = second[:b] + (first[a],) + second[b:]
                    yield tuple(route for route in candidate if route)
                    if i < j and b < len(second):
                        candidate[i] = first[:a] + (second[b],) + first[a+1:]
                        candidate[j] = second[:b] + (first[a],) + second[b+1:]
                        yield tuple(route for route in candidate if route)


def improve(problem: CVRP, initial: RoutingResult, time_limit_s: float = 1.0) -> RoutingResult:
    """Apply best feasible improvements until convergence or the deadline."""
    if not isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("Expected a finite positive time limit")
    if initial.status != "feasible":
        return initial
    start = perf_counter()
    deadline = start + time_limit_s
    routes = initial.routes
    cost = validate(problem, routes).distance_m
    if cost is None:
        raise ValueError("Local search requires an independently feasible start")
    while perf_counter() < deadline:
        best_routes, best_cost = routes, cost
        for candidate in neighbours(routes, problem.vehicle_count):
            if perf_counter() >= deadline:
                break
            score = validate(problem, candidate).distance_m
            if score is not None and score < best_cost:
                best_routes, best_cost = candidate, score
        if best_cost == cost:
            break
        routes, cost = best_routes, best_cost
    return RoutingResult("feasible", routes, perf_counter() - start + initial.elapsed_s,
                         "Includes greedy construction time")
