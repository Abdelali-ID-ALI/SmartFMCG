"""Small independent checks of the heterogeneous pickup-and-delivery model."""

from dataclasses import replace

import pytest

from src.optimization.fmcg_greedy import cheapest_insertion, schedule
from src.optimization.fmcg_problem import FMCGProblem, Stop, Vehicle, ZERO
from src.optimization.fmcg_solver import solve_fmcg
from src.optimization.fmcg_validator import Tour, UNITS_PER_MAD, validate_fmcg


def instance(stops, vehicles=None, depot_window=(0, 1000)):
    size = len(stops) + 1
    distance = tuple(tuple(0 if i == j else 1000 for j in range(size)) for i in range(size))
    duration = tuple(tuple(0 if i == j else 10 for j in range(size)) for i in range(size))
    default = Vehicle("V", (10, 10, 10, 10), (0, 1000), 1000, 200, 3600)
    return FMCGProblem(tuple(stops), (default,) if vehicles is None else vehicles, depot_window, distance, duration)


@pytest.mark.parametrize("dimension", range(4))
def test_post_pickup_overload_forces_reverse_visit(dimension):
    def load(value):
        return tuple(value if i == dimension else 0 for i in range(4))

    a = Stop("A", load(3), load(5), (0, 1000), 0)
    b = Stop("B", load(6), ZERO, (0, 1000), 0)
    problem = instance([a, b])
    with pytest.raises(ValueError, match="pickup capacity"):
        validate_fmcg(problem, (schedule(problem, 0, (1, 2)),))
    result = solve_fmcg(problem, time_limit_s=.15)
    assert result.tours[0].stops == (2, 1)
    trace = validate_fmcg(problem, result.tours).tours[0].loads
    assert [load_state.onboard[dimension] for load_state in trace] == [9, 3, 5]
    assert [load_state.remaining_delivery[dimension] for load_state in trace] == [9, 3, 0]


def test_initial_delivery_cannot_be_hidden_by_negative_net_changes():
    a = Stop("A", (6, 0, 0, 0), ZERO, (0, 1000), 0)
    b = replace(a, id="B")
    problem = instance([a, b])
    with pytest.raises(ValueError, match="Initial load"):
        validate_fmcg(problem, (schedule(problem, 0, (1, 2)),))
    assert solve_fmcg(problem, time_limit_s=.15).status in {"infeasible", "no_solution"}


def test_cold_chain_access_and_vehicle_cost():
    stop = Stop("A", (1, 2, 3, 4), ZERO, (0, 1000), 10, refrigerated=True, no_heavy=True)
    van = Vehicle("V", (10, 10, 10, 10), (0, 1000), 1000, 200, 3600)
    problem = instance([stop], (van, replace(van, id="H", heavy=True, refrigerated=True),
                               replace(van, id="R", refrigerated=True)))
    result = solve_fmcg(problem, time_limit_s=.15)
    assert result.tours[0].vehicle == 2
    report = validate_fmcg(problem, result.tours)
    assert report.cost == result.objective_units
    assert report.cost / UNITS_PER_MAD == pytest.approx(14.3)
    with pytest.raises(ValueError, match="Incompatible"):
        validate_fmcg(problem, (schedule(problem, 1, (1,)),))


@pytest.mark.parametrize("algorithm", [cheapest_insertion, lambda p: solve_fmcg(p, time_limit_s=.15)])
def test_tight_window_requires_waiting_and_preserves_service_completion(algorithm):
    a = Stop("A", ZERO, ZERO, (0, 30), 10)
    b = Stop("B", ZERO, ZERO, (100, 120), 10)
    problem = instance([a, b], depot_window=(0, 120))
    result = algorithm(problem)
    assert tuple(replace(tour, initial_delivery=None) for tour in result.tours) == (Tour(0, (1, 2), 10, (20, 100), 120),)
    report = validate_fmcg(problem, result.tours)
    assert report.tours[0].waiting_s == 60
    assert report.tours[0].duration_s == 110
    with pytest.raises(ValueError, match="Service timing"):
        validate_fmcg(problem, (Tour(0, (1, 2), 11, (21, 100), 120),))
    with pytest.raises(ValueError, match="operating hours"):
        validate_fmcg(problem, (replace(result.tours[0], return_s=121),))
    with pytest.raises(ValueError, match="operating hours"):
        validate_fmcg(replace(problem, vehicles=(replace(problem.vehicles[0], shift=(0, 119)),)),
                      result.tours)


def test_optional_omission_is_priced_and_mandatory_omission_rejected():
    stop = Stop("A", ZERO, ZERO, (0, 1000), 10, penalty_cents=100)
    problem = instance([stop])
    for result in (cheapest_insertion(problem), solve_fmcg(problem, time_limit_s=.15)):
        assert not result.tours
        assert result.omissions[0][0] == "A"
        assert result.objective_units == UNITS_PER_MAD
    with pytest.raises(ValueError, match="Missing mandatory"):
        validate_fmcg(instance([replace(stop, penalty_cents=None)]), ())


def test_no_eligible_vehicle_is_a_hard_failure_for_mandatory_stop():
    stop = Stop("Cold", ZERO, ZERO, (0, 1000), 0, refrigerated=True)
    result = solve_fmcg(instance([stop]), time_limit_s=.15)
    assert result.status == "infeasible"
    assert "Cold" in result.detail
    optional = instance([replace(stop, penalty_cents=500)])
    result = solve_fmcg(optional, time_limit_s=.15)
    assert result.objective_units == 5 * UNITS_PER_MAD
    assert "eligible" in result.omissions[0][1]


def test_heterogeneous_capacity_forces_larger_vehicle():
    stop = Stop("Bulky", (1, 11, 1, 1), ZERO, (0, 1000), 0)
    small = Vehicle("S", (10, 10, 10, 10), (0, 1000), 100, 200, 3600)
    large = replace(small, id="L", capacity=(10, 12, 10, 10), fixed_cents=200)
    problem = instance([stop], (small, large))
    result = solve_fmcg(problem, time_limit_s=.15)
    assert result.tours[0].vehicle == 1
    assert validate_fmcg(problem, result.tours).tours[0].loads[0].onboard == stop.delivery


def test_directed_unavailable_arc_rejected_by_both_solver_and_validator():
    a = Stop("A", ZERO, ZERO, (0, 1000), 0)
    problem = replace(instance([a, replace(a, id="B")]),
                      travel_s=((0, None, 10), (10, 0, 10), (None, 10, 0)),
                      distance_m=((0, None, 1000), (1000, 0, 1000), (None, 1000, 0)))
    with pytest.raises(ValueError, match="Unavailable directed arc"):
        validate_fmcg(problem, (Tour(0, (1, 2), 0, (10, 20), 30),))
    result = solve_fmcg(problem, time_limit_s=.15)
    assert result.tours[0].stops == (2, 1)


@pytest.mark.parametrize("shift,depot", [((0, 29), (0, 1000)), ((0, 1000), (0, 29))])
def test_solver_enforces_depot_and_shift_return_deadlines(shift, depot):
    stop = Stop("A", ZERO, ZERO, (0, 1000), 10)
    vehicle = Vehicle("V", (10, 10, 10, 10), shift, 0, 0, 0)
    problem = instance([stop], (vehicle,), depot_window=depot)
    result = solve_fmcg(problem, time_limit_s=.15)
    assert result.status in {"infeasible", "no_solution"}
    assert not result.tours
