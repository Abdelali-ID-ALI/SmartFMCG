"""Independent checks for the Phase 10 experimental contract."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("stable_baselines3")

from src.benchmark.analysis import complete_cases, paired, pareto, summaries
from src.benchmark.evaluation import audit_metrics, load_dispatch, make_plan, urgent_witnesses
from src.benchmark.experiment import CONFIG, validate_config, verify_fairness
from src.benchmark.scenarios import build_draws, fingerprint, instance, make_case, planning_capacity, traffic_realization
from src.planning.forecast import build_forecasts
from src.rl.dqn import DQNEnv, WIDTH
from src.simulation.experiment import load_reference
from src.simulation.state import UrgentOrder
from test_q_learning import miniature


@pytest.fixture(scope="module")
def protocol():
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def test_configuration_rejects_noncomparable_baseline_and_overlap(protocol):
    validate_config(protocol)
    altered = deepcopy(protocol)
    altered["policies"]["B6"]["planner"] = "static"
    with pytest.raises(ValueError, match="B6"):
        validate_config(altered)
    altered = deepcopy(protocol)
    altered["profiles"]["standard"]["first_day"] = "2026-05-27"
    with pytest.raises(ValueError, match="held out"):
        build_draws(altered, "standard", 20)


def test_scenario_reproduction_and_disjoint_seed_spaces(protocol, reference):
    manifest, matrix, base, model, history, _, _, _ = reference
    draws = build_draws(protocol, "standard", 20)
    dev = build_draws(protocol, "development", 20)
    other_seeds = {row.seed for row in dev}
    assert not other_seeds & {row.seed for row in draws}
    assert len({draw.seed for draw in draws if draw.block == "benchmark"}) == 35
    d = draws[3]
    forecast = build_forecasts(matrix, d.day, model, history)
    actual, errors = traffic_realization(base, forecast, d, matrix.ids, model.zones, manifest["reference_hour"])
    def create():
        return make_case(base, d, protocol, matrix.ids, model.zones, actual, fingerprint(actual), errors)
    left, right = create(), create()
    assert left == right
    assert left.problem.distance_m == base.distance_m and left.scenario.actual_s != (base.travel_s,) * 24
    assert left.scenario_fingerprint != make_case(
        base, replace(d, seed=d.seed + 1), protocol, matrix.ids, model.zones, actual, fingerprint(actual), errors
    ).scenario_fingerprint


def test_sensitivity_changes_one_public_or_external_factor(protocol, reference):
    manifest, matrix, base, model, history, _, _, _ = reference
    draws = [draw for draw in build_draws(protocol, "development", 20) if draw.block == "sensitivity"]
    original = draws[0]
    forecast = build_forecasts(matrix, original.day, model, history)
    actual, errors = traffic_realization(base, forecast, original, matrix.ids, model.zones, manifest["reference_hour"])
    initial = make_case(base, original, protocol, matrix.ids, model.zones, actual, "traffic", errors)
    for d in draws[1:]:
        before, after = asdict(original), asdict(d)
        overrides = protocol["sensitivity"]["variants"][d.variant]["field"]
        for key in ("key", "variant"):
            before.pop(key)
            after.pop(key)
        if overrides == "urgent_penalty_cents":
            assert {k for k in before if before[k] != after[k]} == {"penalty_cents"}
        elif overrides == "latency_s":
            assert {k for k in before if before[k] != after[k]} == {"latency_s"}
        else:
            assert {k for k in before if before[k] != after[k]} == {"design"}
            assert {k for k in before["design"] if before["design"][k] != after["design"][k]} == {overrides}
        candidate = make_case(base, d, protocol, matrix.ids, model.zones, actual, "traffic", errors)
        assert candidate.scenario.actual_s is initial.scenario.actual_s
        assert [s.delivery for s in candidate.problem.stops] == [s.delivery for s in initial.problem.stops]
        if overrides in {"urgent_penalty_cents", "latency_s", "traffic_factor", "service_extra_s"}:
            assert d.planning_key() == original.planning_key()
        first = next(e for e in initial.scenario.events if isinstance(e, UrgentOrder))
        second = next(e for e in candidate.scenario.events if isinstance(e, UrgentOrder))
        assert (first.at_s, first.order.node, first.order.stop.delivery) == (
            second.at_s, second.order.node, second.order.stop.delivery
        )


def test_planners_reserve_same_inventory_and_loaded_plan_roundtrips(protocol, reference):
    manifest, matrix, base, model, history, _, _, _ = reference
    draw = build_draws(protocol, "development", 20)[0]
    physical, reserved = instance(base, draw)
    effective = planning_capacity(physical, reserved)
    cold = next(i for i, v in enumerate(physical.vehicles) if v.refrigerated)
    assert physical.vehicles[cold].capacity[2] - effective.vehicles[cold].capacity[2] == 2 * draw.design.reserve
    forecasts = build_forecasts(matrix, draw.day, model, history)
    args = physical, reserved, forecasts, "greedy", manifest["reference_hour"], manifest["buffer_ratio"], .3
    greedy = make_plan(*args)
    assert greedy.usable and any(t.vehicle == cold for t in greedy.tours)
    stored = greedy.record("sample", "greedy")
    assert make_plan(*args, archived=stored) == greedy
    with pytest.raises(ValueError, match="different planning problem"):
        make_plan(*args, archived={**stored, "planning_problem_sha256": "other"})
    with pytest.raises(ValueError, match="vehicle"):
        instance(base, replace(draw, design=replace(draw.design, unavailable="V06")))


def test_frozen_cpu_actions_use_three_seeds():
    info, policies = load_dispatch()
    assert info["tabular_family"] == "tabular_A"
    obs = np.zeros(WIDTH, dtype=np.float32)
    public = {"original_state": (0, 0, 0, 0), "action_mask": (1, 0, 0, 0)}
    for group in policies.values():
        assert set(group) == {2917, 2927, 2953}
        assert all(policy(obs, public, None) == 0 for policy in group.values())


def test_public_input_has_identical_prefix_with_hidden_traffic_and_event():
    regular = DQNEnv(miniature().episodes, probe_budget=2, tight_s=30)
    changed = DQNEnv(miniature(delayed=True, hidden=((0, 15, 10), (10, 0, 10), (10, 10, 0))).episodes,
                     probe_budget=2, tight_s=30)
    for one, two in ((regular.reset(seed=12), changed.reset(seed=12)),
                     (regular.step(0), changed.step(0))):
        assert np.array_equal(one[0], two[0]) and one[1:] == two[1:]


def test_hand_checked_driving_cost_and_tardiness_is_physically_feasible():
    env = miniature(delayed=True, hidden=((0, 10, 10), (10, 0, 115), (10, 10, 0)))
    env.reset()
    total = 0.
    while True:
        _, reward, ended, _, info = env.step(0)
        total += reward
        if ended:
            break
    metrics = env.episode_metrics()
    report = audit_metrics(env._sim, metrics, info["costs_mad"], env.rates, total)
    assert report["realized_travel_min"] == pytest.approx(135 / 60)
    assert report["mandatory_completion_rate"] == 1 and report["physical_violations"] == 0
    assert report["perfect_schedule"] == 0
    assert metrics["objective_mad"] == pytest.approx(-total)
    with pytest.raises(RuntimeError, match="reconcile"):
        audit_metrics(env._sim, {**metrics, "otif": 1.}, info["costs_mad"], env.rates, total + 1.)


def test_paired_protocol_rejects_clock_plan_or_seed_mismatch(protocol):
    rows = []
    for policy, designs in protocol["policies"].items():
        seeds = (2917, 2927, 2953) if policy in {"B9", "B10"} else (None,)
        for seed in seeds:
            rows.append({"case": "case", "policy": policy, "block": "benchmark", "training_seed": seed,
                         "scenario_sha256": "same", "traffic_sha256": "same", "decision_clock_sha256": "same",
                         "dispatched": True, "planner": designs["planner"], "plan_sha256": designs["planner"]})
    frame = pd.DataFrame(rows)
    verify_fairness(frame, protocol)
    with pytest.raises(ValueError, match="opportunities"):
        verify_fairness(frame.assign(decision_clock_sha256=np.arange(len(frame))), protocol)
    with pytest.raises(ValueError, match="planner"):
        verify_fairness(frame.assign(plan_sha256=np.arange(len(frame))), protocol)
    with pytest.raises(ValueError, match="seed"):
        verify_fairness(frame.drop(frame.index[frame.policy.eq("B10")][0]), protocol)


def test_pairing_with_dates_and_seeds_recovers_constant_delta():
    rows = []
    for day in (1, 2, 3):
        for case in (1, 2):
            for seed in (2917, 2927, 2953):
                for policy, offset in (("L", -20), ("R", 0)):
                    rows.append({"case": f"{day}-{case}", "day": day, "regime": "same",
                                 "training_seed": seed, "policy": policy, "dispatched": True,
                                 "objective_mad": 1600 + day + seed % 6 + case + offset})
    frame = pd.DataFrame(rows)
    raw, intervals = paired(frame, [("L", "R", "example")], 100, 514, ("objective_mad",))
    assert len(raw) == 6
    assert np.allclose(intervals[["difference", "ci_low", "ci_high"]], -20)
    s = summaries(frame, ["policy"], ("objective_mad",))
    assert s.cases.eq(6).all()
    altered = frame.copy()
    altered.loc[0, "dispatched"] = False
    assert complete_cases(altered).case.nunique() == 5


def test_cost_service_dominance_preserves_tradeoffs():
    means = {"X": (10, .8), "Y": (11, .9), "Z": (12, .85)}
    rows = [{"policy": policy, "metric": metric, "mean": value}
            for policy, coordinates in means.items()
            for metric, value in zip(("objective_mad", "otif"), coordinates, strict=True)]
    result = pareto(pd.DataFrame(rows)).set_index("policy")
    assert set(result[result.pareto_loss_otif].index) == {"X", "Y"}
