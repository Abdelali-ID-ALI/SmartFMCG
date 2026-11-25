"""Independently evaluate proposed classical CVRP routes."""

from collections import Counter
from dataclasses import dataclass

from src.optimization.problem import CVRP, Routes


@dataclass(frozen=True)
class Validation:
    errors: tuple[str, ...]
    distance_m: int | None
    loads_kg: tuple[int, ...]

    @property
    def feasible(self) -> bool:
        return not self.errors


def validate(problem: CVRP, routes: Routes) -> Validation:
    """Verify every customer, capacity, fleet size, depot legs and total distance."""
    errors = []
    loads = []
    total = 0
    visited: Counter[int] = Counter()
    if len(routes) > problem.vehicle_count:
        errors.append("Too many vehicles used")
    for number, route in enumerate(routes, start=1):
        if not route or any(not 1 <= node < len(problem.ids) for node in route):
            errors.append(f"Invalid stops in route {number}")
            continue
        visited.update(route)
        load = sum(problem.demand_kg[node] for node in route)
        loads.append(load)
        if load > problem.capacity_kg:
            errors.append(f"Capacity exceeded in route {number}")
        path = (0, *route, 0)
        for origin, destination in zip(path, path[1:]):
            distance = problem.distance_m[origin][destination]
            if distance is None:
                errors.append(f"Unavailable arc {problem.ids[origin]} → {problem.ids[destination]}")
            else:
                total += distance
    for node, identifier in enumerate(problem.ids[1:], start=1):
        if visited[node] != 1:
            errors.append(f"{identifier} served {visited[node]} times")
    return Validation(tuple(errors), total if not errors else None, tuple(loads))
