"""Manually check CVRP constraints, directed costs and failure cases."""

from dataclasses import replace

import pytest

from src.optimization.greedy import nearest_neighbour
from src.optimization.local_search import improve
from src.optimization.problem import CVRP, RoutingResult
from src.optimization.solution_validator import validate
from src.optimization.vrp_solver import solve


@pytest.fixture
def small():
    return CVRP(("D", "A", "B"), (0, 6, 6), ((0, 10, 20), (40, 0, 5), (30, 25, 0)), 2, 10)


@pytest.mark.parametrize("algorithm", [nearest_neighbour, solve])
def test_two_vehicles_required(small, algorithm):
    result = algorithm(small)
    checked = validate(small, result.routes)
    assert result.status in {"feasible", "optimal"}
    assert checked.feasible
    assert checked.distance_m == 100
    assert checked.loads_kg == (6, 6)


def test_validator_rejects_overload_and_duplicate(small):
    assert validate(small, ((1, 2),)).errors == ("Capacity exceeded in route 1",)
    assert validate(small, ((1,), (1,))).errors == ("A served 2 times", "B served 0 times")
    assert validate(small, ((0,), (2,))).distance_m is None


@pytest.mark.parametrize("algorithm", [nearest_neighbour, solve])
@pytest.mark.parametrize("fleet,capacity,reason", [(1, 10, "Total demand"), (2, 5, "customer demand")])
def test_capacity_certificate(small, algorithm, fleet, capacity, reason):
    result = algorithm(replace(small, vehicle_count=fleet, capacity_kg=capacity))
    assert result.status == "infeasible"
    assert reason in result.detail


def test_local_search_uses_directed_costs(small):
    problem = replace(small, capacity_kg=12)
    initial = RoutingResult("feasible", ((2, 1),), 0)
    assert validate(problem, initial.routes).distance_m == 85
    result = improve(problem, initial)
    assert result.routes == ((1, 2),)
    assert validate(problem, result.routes).distance_m == 45


def test_solver_prohibits_unavailable_arcs(small):
    # The only complete route is D → B → A → D.
    problem = replace(small, vehicle_count=1, capacity_kg=12,
                      distance_m=((0, None, 20), (40, 0, 5), (None, 25, 0)))
    assert not validate(problem, ((1, 2),)).feasible
    assert nearest_neighbour(problem).status == "no_solution"
    result = solve(problem)
    assert result.routes == ((2, 1),)
    assert validate(problem, result.routes).distance_m == 85


def test_solver_reports_absence_of_solution(small):
    problem = replace(small, distance_m=((0, None, None), (40, 0, 5), (30, 25, 0)))
    result = solve(problem)
    assert result.status in {"no_solution", "infeasible"}
    assert not result.routes
    assert result.detail.startswith("ROUTING_")


def test_rejects_invalid_budget(small):
    with pytest.raises(ValueError, match="time limit"):
        solve(small, time_limit_s=0)
