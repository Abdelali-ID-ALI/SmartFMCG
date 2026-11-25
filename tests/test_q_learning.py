"""Gymnasium contract, causal transitions, suffix safety and tabular learning."""

from dataclasses import replace
from pathlib import Path

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest
import yaml

from src.optimization.fmcg_problem import FMCGProblem, Stop, Vehicle, ZERO
from src.optimization.fmcg_validator import Tour
from src.rl.actions import DispatchAction
from src.rl.environment import DistributionEnv, Episode
from src.rl.q_learning import Parameters, QTable
from src.rl.scenarios import build_banks
from src.simulation.simulator import Simulator
from src.simulation.state import AMBIENT, Congestion, Order, Scenario, ServiceDelay, Settings, UrgentOrder


CARGO = (2, 2, 2, 0)
TIME = ((0, 10, 10), (10, 0, 10), (10, 10, 0))
DISTANCE = ((0, 100, 100), (100, 0, 100), (100, 100, 0))


def miniature(*, delayed=False, cutoff=False, deadline=95, hidden=None, max_steps=None):
    stops = (Stop("A", (3, 3, 3, 0), (4, 4, 4, 0), (0, 150), 10),
             Stop("B", (4, 4, 4, 0), ZERO, (0, 150), 10))
    truck = Vehicle("V1", (10, 10, 10, 1), (0, 160), 1000, 200, 3600)
    problem = FMCGProblem(stops, (truck,), (0, 160), DISTANCE, TIME)
    tours = (Tour(0, (1, 2), 0, (10, 30), 50),)
    urgent = UrgentOrder(21, Order(Stop("U", CARGO, ZERO, (21, deadline), 1, penalty_cents=8000), 2,
                                   AMBIENT, priority=True))
    events = (urgent, ServiceDelay(23, "B", 30)) if delayed else (urgent,)
    if not cutoff:
        events = (Congestion(5, 25, frozenset({1}), 1.), *events)
    settings = Settings(20, 5, 40, 180, 0, 300)
    scenario = Scenario(events, (hidden or TIME,) * 24)
    return DistributionEnv([Episode("tiny", lambda: Simulator(problem, tours, {"A": AMBIENT, "B": AMBIENT},
                                                              frozenset(), ((CARGO, ZERO),), scenario, settings))],
                           max_steps=max_steps)


def rollout(env, urgent_action=DispatchAction.KEEP_PLAN):
    state, info = env.reset(seed=917)
    records, rewards = [], []
    while True:
        action = urgent_action if "urgent_order" in info.get("event_types", ()) else DispatchAction.KEEP_PLAN
        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        records.append((tuple(state), reward, terminated, truncated, info))
        if terminated or truncated:
            return records, rewards


def test_gymnasium_checker_and_replay():
    env = miniature()
    check_env(env, skip_render_check=True)
    first, rewards = rollout(env)
    second, _ = rollout(env)
    assert first == second
    assert len({row[4]["elapsed_s"] for row in first}) > 1
    assert all(env.observation_space.contains(np.array(row[0])) for row in first)
    assert -sum(rewards) == pytest.approx(91.1)
    assert -sum(rewards) == pytest.approx(env.episode_metrics()["objective_mad"])
    assert first[-1][-1]["costs_mad"]["omission_mad"] == 80
    assert all(row[-1]["costs_mad"]["omission_mad"] == 0 for row in first[:-1])
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(0)


def test_three_replanning_choices_are_distinct_computations_and_share_guards():
    for action in tuple(DispatchAction)[1:]:
        env = miniature()
        rows, rewards = rollout(env, action)
        issued = next(info["decision"] for *_, info in rows if info["decision"]["requested"] == action.name)
        assert (issued["requested"], issued["executed"], issued["status"]) == (action.name, action.name, "applied")
        assert issued["modeled_latency_s"] == 5
        assert issued["solver_calls"] == (action == DispatchAction.LOCAL_REOPTIMIZATION)
        assert env.episode_metrics()["urgent_served"] == 1
        assert -sum(rewards) == pytest.approx(env.episode_metrics()["objective_mad"])
    env = miniature(delayed=True, deadline=50)
    rows, _ = rollout(env, DispatchAction.LOCAL_REOPTIMIZATION)
    issued = [info["decision"] for *_, info in rows if info["decision"]["requested"] == "LOCAL_REOPTIMIZATION"]
    assert issued[0]["status"] == "pending"
    assert any(result["fallback_reason"] == "invalid_at_application" for *_, info in rows
               for result in info["settled_decisions"])
    assert env.episode_metrics()["urgent_served"] == 0


def test_no_future_information_from_hidden_travel_or_events():
    standard = miniature()
    hidden = miniature(delayed=True, hidden=((0, 15, 10), (10, 0, 10), (10, 10, 0)))
    first, x = standard.reset(seed=10)
    second, y = hidden.reset(seed=10)
    assert np.array_equal(first, second) and x == y
    a, *left = standard.step(0)
    b, *right = hidden.step(0)
    assert np.array_equal(a, b) and left == right
    assert standard.public_observation.vehicles[0].anchor_s == 20
    assert "urgent_order" not in left[-1]["event_types"]


def test_truncation_keeps_bootstrap_state_and_no_missing_order_penalty():
    env = miniature(max_steps=1)
    env.reset(seed=12)
    state, reward, terminated, truncated, info = env.step(0)
    assert not terminated and truncated
    assert info["now_s"] == 5 and reward < 0 and env.observation_space.contains(state)
    assert info["costs_mad"]["omission_mad"] == 0
    with pytest.raises(RuntimeError, match="complete day"):
        env.episode_metrics()


def test_unavailable_action_has_visible_safe_fallback():
    env = miniature()
    _, info = env.reset()
    while info["action_mask"] != (1, 0, 0, 0):
        _, _, _, _, info = env.step(0)
    _, _, _, _, info = env.step(3)
    assert info["decision"]["fallback_reason"] == "unavailable"
    assert info["decision"]["executed"] == "KEEP_PLAN"
    assert info["decision"]["solver_calls"] == 0


def test_bellman_example_terminal_truncation_and_reload(tmp_path: Path):
    table = QTable(Parameters(.5, .9))
    current, successor, unseen = (1, 0, 1, 0), (3, 1, 0, 1), (2, 0, 0, 0)
    mask = (1, 1, 1, 1)
    table.update(current, 1, 8, successor, mask, terminated=True, elapsed_s=900)
    table.update(successor, 2, 12, successor, mask, terminated=True, elapsed_s=900)
    assert table.update(current, 1, -2, successor, mask, terminated=False, elapsed_s=900) == pytest.approx(3.7)
    assert table.update(current, 1, -2, successor, mask, terminated=True, elapsed_s=900) == pytest.approx(.85)
    assert Parameters(.5, .81).discount(450) == pytest.approx(.9)
    assert table.choose(unseen, mask) == 0
    result = tmp_path / "model.json"
    table.save(result, seed=11)
    reloaded = QTable.load(result)
    assert reloaded.choose(current, mask) == table.choose(current, mask)
    assert reloaded.choose(successor, mask) == 2
    assert np.array_equal(reloaded.counts[current], table.counts[current])


def test_train_validation_test_dates_seeds_and_cases_are_disjoint():
    config = yaml.safe_load(Path("configs/rl_training.yaml").read_text())
    banks = build_banks(config, "development")
    for feature in ("day", "key", "seed"):
        groups = [{getattr(case, feature) for case in cases} for cases in banks.values()]
        assert all(not left & right for i, left in enumerate(groups) for right in groups[i + 1:])
    for case in (case for bank in banks.values() for case in bank):
        sim = case.episode(Settings(**config["dispatch"])).make_simulator()
        assert 10 <= len(sim.problem.stops) <= 20 and len(sim.tours) == 2


def test_cp_sat_finds_a_multi_move_suffix_outside_the_local_heuristic_neighborhood():
    stops = tuple(Stop(key, (1, 1, 1, 0), ZERO, (0, close), 10)
                  for key, close in zip("ABCD", (280, 45, 280, 85), strict=True))
    travel = tuple(tuple(0 if i == j else 10 for j in range(5)) for i in range(5))
    distance = tuple(tuple(0 if i == j else 100 for j in range(5)) for i in range(5))
    truck = Vehicle("V1", (10, 10, 10, 1), (0, 300), 1000, 300, 3600)
    problem = FMCGProblem(stops, (truck,), (0, 300), distance, travel)
    tour = Tour(0, (1, 2, 3, 4), 0, (10, 30, 50, 70), 90)
    scenario = Scenario((ServiceDelay(0, "A", 60), ServiceDelay(0, "C", 60)), (travel,) * 24)
    episode = Episode("multi_move", lambda: Simulator(
        problem, (tour,), {key: AMBIENT for key in "ABCD"}, frozenset(),
        ((ZERO, ZERO),), scenario, Settings(20, 5, 100, 180, 0, 300)
    ))
    for action in (DispatchAction.LOCAL_HEURISTIC, DispatchAction.LOCAL_REOPTIMIZATION,
                   DispatchAction.GLOBAL_REOPTIMIZATION):
        env = DistributionEnv([episode], solver_budget=.02)
        env.reset(seed=1)
        _, _, _, _, info = env.step(action)
        effort = info["decision"]
        if action == DispatchAction.LOCAL_REOPTIMIZATION:
            assert effort["executed"] == action.name and effort["solver_calls"] == 1
        else:
            assert effort["status"] == "no_improvement" and effort["solver_calls"] == 0
