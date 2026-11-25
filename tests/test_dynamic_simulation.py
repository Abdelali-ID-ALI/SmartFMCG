"""Hand-checkable operational transitions and paired online comparisons."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.optimization.fmcg_problem import FMCGProblem, Stop, Vehicle, ZERO
from src.optimization.fmcg_validator import Tour
from src.planning.replay import replay
from src.simulation.experiment import run
from src.simulation.metrics import Costs, evaluate, reward_delta
from src.simulation.policies import Action, always, never
from src.simulation.replanning import Forecast
from src.simulation.simulator import Simulator
from src.simulation.state import (AMBIENT, CHILLED, Congestion, Order, Scenario, ServiceDelay,
                                  Settings, UrgentOrder)
from src.simulation.validation import validate_movements, validate_routes


def fixture(events=(), *, actual=None, reserves=((2, 2, 2, 0), ZERO), settings=None):
    stops = (
        Stop("A", (3, 3, 3, 0), (4, 4, 4, 0), (0, 150), 10),
        Stop("B", (4, 4, 4, 0), ZERO, (0, 150), 10),
    )
    vehicles = (Vehicle("V1", (10, 10, 10, 1), (0, 200), 1000, 200, 3600),
                Vehicle("V2", (10, 10, 10, 1), (0, 200), 1000, 200, 3600, refrigerated=True))
    problem = FMCGProblem(stops, vehicles, (0, 200),
                          ((0, 100, 100), (100, 0, 100), (100, 100, 0)),
                          ((0, 10, 10), (10, 0, 10), (10, 10, 0)))
    tours = (Tour(0, (1, 2), 0, (10, 30), 50),)
    point = Settings(20, 5, 40, 180, 0, 300) if settings is None else settings
    scenario = Scenario(events, (actual or problem.travel_s,) * 24)
    return Simulator(problem, tours, {"A": AMBIENT, "B": AMBIENT}, frozenset({"B"}),
                     (reserves, (ZERO, ZERO)), scenario, point)


def simulate(sim, policy=never):
    observations = []
    while (obs := sim.next_observation()) is not None:
        observations.append(obs)
        sim.act(policy(obs))
    return observations


def urgent(at=21, *, demand=(2, 2, 2, 0), refrigerate=False, deadline=100, node=2):
    order = Order(Stop("U", demand, ZERO, (at, deadline), 1, refrigerate, False, 8000), node,
                  CHILLED if refrigerate else AMBIENT, True)
    return UrgentOrder(at, order)


def test_congestion_during_travel_and_service_delay_preserve_position_and_load():
    sim = fixture((Congestion(5, 65, frozenset({1}), 2.), ServiceDelay(15, "A", 15)))
    observations = {obs.now_s: obs for obs in simulate(sim)}
    at_5 = observations[5].vehicles[0]
    assert (at_5.node, at_5.destination, at_5.committed, at_5.tail, at_5.anchor_s) == (0, 1, "A", ("B",), 20)
    at_15 = observations[15].vehicles[0]
    assert (at_15.phase, at_15.node, at_15.anchor_s) == ("service", 1, 35)
    assert observations[20].completed == frozenset()
    assert sim.completed == {"A": 35, "B": 65}
    arrivals = [(entry.node, entry.time_s) for entry in sim.movements if entry.kind == "arrive"]
    assert arrivals == [(1, 10), (2, 55), (0, 75)]
    first = next(entry for entry in sim.movements if entry.kind == "service_end")
    assert first.holdings == ((4, 4, 4, 0), (2, 2, 2, 0), ZERO, (4, 4, 4, 0))
    assert evaluate(sim)["realized_cost_mad"] == pytest.approx(11.35)


def test_urgent_insertion_preserves_regular_goods_and_reconciles_depot_receipts():
    sim = fixture((urgent(),))
    simulate(sim, always)
    assert sim.completed == {"A": 20, "B": 40, "U": 41}
    applied = next(d for d in sim.decisions if d.status == "applied")
    assert (applied.time_s, applied.applied_s, applied.modeled_latency_s, applied.executed) == (21, 26, 5, "GLOBAL")
    service = [entry for entry in sim.movements if entry.kind == "service_end"]
    assert service[-1].holdings == (ZERO, ZERO, ZERO, (4, 4, 4, 0))
    assert sim.movements[-2].holdings == service[-1].holdings
    assert sim.movements[-1].holdings == (ZERO, ZERO, ZERO, ZERO)
    assert evaluate(sim)["urgent_otif"] == 1


@pytest.mark.parametrize(("demand", "refrigerate"), [
    ((3, 2, 2, 0), False), ((2, 3, 2, 0), False), ((2, 2, 3, 0), False),
    ((2, 2, 2, 1), False), ((2, 2, 2, 0), True),
])
def test_urgent_refusal_is_reported_without_consuming_stock(demand, refrigerate):
    sim = fixture((urgent(demand=demand, refrigerate=refrigerate),))
    simulate(sim, always)
    assert evaluate(sim)["unserved"] == 1
    assert evaluate(sim)["omission_mad"] == 80
    assert set(sim.completed) == {"A", "B"}
    assert sim.movements[-2].holdings[1] == (2, 2, 2, 0)


def test_revalidation_rejects_proposal_when_a_delay_is_revealed_during_computation():
    sim = fixture((urgent(deadline=50), ServiceDelay(23, "B", 30)))
    simulate(sim, always)
    assert sim.completed == {"A": 20, "B": 70}
    assert [(d.time_s, d.status) for d in sim.decisions if d.events] == [
        (21, "invalid_at_application"), (23, "busy")]


def test_computation_dispatch_hold_is_realized_and_measured():
    sim = fixture((Congestion(17, 100, frozenset({2}), 1.),))
    simulate(sim, always)
    assert sim.completed == {"A": 20, "B": 42}
    assert evaluate(sim)["dispatch_hold_s"] == 2
    assert evaluate(sim)["modeled_latency_s"] == 5


def test_policy_cannot_see_future_events_or_hidden_arrival_time():
    fast = fixture((Congestion(5, 20, frozenset({1}), 1.), urgent(80)))
    slow = fixture((Congestion(5, 20, frozenset({1}), 1.),),
                   actual=((0, 15, 10), (10, 0, 10), (10, 10, 0)))
    for sim in (fast, slow):
        t0 = sim.next_observation()
        assert t0.waiting_urgent == () and t0.event_types == ()
        sim.act(Action.KEEP)
    first, second = fast.next_observation(), slow.next_observation()
    assert first == second
    assert first.vehicles[0].anchor_s == 20
    fast.act(Action.KEEP)
    slow.act(Action.KEEP)
    assert simulate(fast)
    assert simulate(slow)
    assert fast.completed["A"] == 20 and slow.completed["A"] == 25


def test_independent_validator_rejects_mutated_routes_and_execution():
    sim = fixture()
    before = sim.next_observation()
    forecast = Forecast(sim.problem, sim.orders, (), {})
    with pytest.raises(ValueError, match="duplicated"):
        validate_routes(forecast, before.vehicles, (("A", "B", "A"), ()), 0, frozenset())
    with pytest.raises(ValueError, match="assigned stock"):
        validate_routes(forecast, (before.vehicles[0], replace(before.vehicles[1], phase="staged")),
                        (("B",), ("A",)), 0, frozenset())
    # Extra reserve would fit at dispatch but overflow after the first collection.
    with pytest.raises(ValueError, match="capacity"):
        fixture(reserves=((3, 3, 3, 0), ZERO))
    sim.act(Action.KEEP)
    simulate(sim)
    initial = (((7, 7, 7, 0), (2, 2, 2, 0), ZERO, ZERO), (ZERO, ZERO, ZERO, ZERO))
    output = list(sim.movements)
    i = next(i for i, entry in enumerate(output) if entry.kind == "arrive")
    output[i] = replace(output[i], time_s=9)
    with pytest.raises(ValueError, match="realized conditions"):
        validate_movements(sim.problem, initial, tuple(output), sim.orders, {}, sim.scenario)
    output = list(sim.movements)
    i = next(i for i, entry in enumerate(output) if entry.kind == "service_end")
    output[i] = replace(output[i], holdings=(ZERO, ZERO, ZERO, ZERO))
    with pytest.raises(ValueError, match="inventory"):
        validate_movements(sim.problem, initial, tuple(output), sim.orders, {}, sim.scenario)


def test_static_no_shock_matches_phase_six_replay():
    import numpy as np

    sim = fixture()
    simulate(sim)
    reference = replay(sim.problem, sim.tours, (), np.empty((24, 0)), np.empty((24, 0)))
    metrics = evaluate(sim)
    for key in ("realized_cost_mad", "distance_km", "otif", "delay_min", "overtime_min"):
        assert metrics[key] == pytest.approx(reference[key])


def test_reward_delta_is_additive_and_counts_omission_once():
    start = Costs(0, 0, 0, 0, 0, 0, 0)
    middle = Costs(10, 1, 3, 0, 1, 0, 1)
    end = Costs(10, 2, 5, 80, 3, 1, 1)
    rates = {"delay_mad_per_min": 2, "overtime_mad_per_min": 3, "change_mad": 4}
    assert reward_delta(start, end, **rates) == -110
    assert reward_delta(start, middle, **rates) + reward_delta(middle, end, **rates) == -110


def test_experiment_replays_same_scenarios_and_operational_results(tmp_path: Path):
    run("development", tmp_path / "first")
    run("development", tmp_path / "second")
    first, second = tmp_path / "first", tmp_path / "second"
    for name in ("scenarios.json", "paired_differences.csv", "paired_intervals.csv", "example_movements.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    data = pd.read_csv(first / "scenario_metrics.csv")
    assert len(data) == 15
    assert data.groupby(["day", "seed", "regime"]).opportunities.nunique().eq(1).all()
    assert data.unserved_mandatory.eq(0).all()
    assert data.loc[data.policy.eq("never"), "urgent_served"].eq(0).all()
    assert data.loc[data.policy.ne("never"), "urgent_served"].sum() > 0


def test_independent_rejection_keeps_dispatch_safe_if_search_returns_a_bad_candidate(monkeypatch):
    from src.simulation import simulator
    from src.simulation.replanning import Proposal

    sim = fixture((urgent(),))
    monkeypatch.setattr(simulator, "propose", lambda *args: Proposal((("U", "U"), ()), 1))
    simulate(sim, always)
    decision = next(d for d in sim.decisions if d.events)
    assert (decision.status, decision.executed) == ("invalid_candidate", "KEEP")
    assert sim.completed == {"A": 20, "B": 40}


def test_service_delay_at_completion_does_not_reopen_the_delivery():
    sim = fixture((ServiceDelay(20, "A", 100),))
    observed = {obs.now_s: obs for obs in simulate(sim)}
    assert "A" in observed[20].completed
    assert sim.completed == {"A": 20, "B": 40}


def test_nonfinite_traffic_and_invalid_chilled_stock_are_rejected():
    with pytest.raises(ValueError, match="congestion"):
        Congestion(5, 20, frozenset({1}), float("nan"))
    with pytest.raises(ValueError, match="realized travel"):
        fixture(actual=((0, float("inf"), 10), (10, 0, 10), (10, 10, 0)))
    with pytest.raises(ValueError, match="initial cargo"):
        fixture(reserves=(ZERO, (1, 1, 1, 0)))
