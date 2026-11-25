"""Initial planning and dispatch evaluation using the validated solver and simulator."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from math import isclose
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import psutil

from src.benchmark.scenarios import Case, fingerprint, planning_capacity
from src.model_artifacts import verified_bytes
from src.optimization.fmcg_greedy import cheapest_insertion
from src.optimization.fmcg_problem import FMCGProblem
from src.optimization.fmcg_solver import solve_fmcg
from src.optimization.fmcg_validator import Tour, UNITS_PER_MAD, validate_fmcg
from src.planning.forecast import Forecasts
from src.rl.dqn import DQNEnv, DispatchDQN
from src.rl.environment import Episode
from src.rl.experiment import random_policy, reference_policy
from src.rl.q_learning import QTable
from src.rl.refined_observations import CARDINALITIES
from src.rl.rewards import RewardRates
from src.simulation.experiment import digest
from src.simulation.simulator import Simulator
from src.simulation.state import Settings, UrgentOrder, within

ARCHIVE = Path("results/experiments/dqn_evaluation")
FROZEN_PROTOCOL_SHA256 = "efb9634c5233837b8a7321f3efff3a55381aae6aa88b30b84bfbf53cd98e447f"
PLANNERS = ("greedy", "static", "historical", "ml", "buffered")
SEEDS = (2917, 2927, 2953)
PROCESS = psutil.Process()
Policy = Callable[[np.ndarray, dict, object], int]


def frozen_protocol() -> dict:
    return json.loads(verified_bytes(ARCHIVE, Path("frozen_protocol.json"), FROZEN_PROTOCOL_SHA256))


@dataclass(frozen=True)
class Plan:
    problem: FMCGProblem
    tours: tuple[Tour, ...]
    status: str
    detail: str
    planning_wall_s: float
    solver_calls: int
    cost_mad: float | None
    omitted_regular: int
    fingerprint: str

    @property
    def usable(self) -> bool:
        return self.status in {"feasible", "optimal"}

    def record(self, key, mode) -> dict:
        return {"planning_key": key, "planner": mode, "status": self.status, "detail": self.detail,
                "planning_wall_s": self.planning_wall_s, "initial_solver_calls": self.solver_calls,
                "planned_cost_mad": self.cost_mad, "planned_regular_omissions": self.omitted_regular,
                "planning_problem_sha256": self.fingerprint, "tours": [asdict(t) for t in self.tours]}


def freeze_approximation(output: Path) -> None:
    main = output / "frozen_protocol.json"
    if (output / "scenario_metrics.csv").exists():
        raise ValueError("Freeze the matched-information comparison before reading benchmark outcomes")
    rows = [row for row in frozen_protocol()["models"] if row["policy"] == "dqn_compact"]
    if tuple(sorted(row["training_seed"] for row in rows)) != SEEDS:
        raise ValueError("Expected the three validation-selected compact checkpoints")
    record = {
        "checkpoints": rows, "benchmark_protocol_sha256": digest(main),
        "source_protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "scope": "All benchmark cases, reuse their B4 plans, all three frozen training seeds",
        "comparisons": [["B12", "B9", "matched_information_neural_vs_tabular"],
                        ["B10", "B12", "continuous_vs_compact_neural"]],
        "wall_limit_s": 85,
    }
    (output / "approximation_protocol.json").write_text(json.dumps(record, indent=2) + "\n")


def load_dispatch() -> tuple[dict, dict[str, dict[int, Policy]]]:
    """Choose and verify frozen models using the previous validation record alone."""
    protocol = frozen_protocol()
    candidates = [row for row in protocol["models"] if row["policy"].startswith("tabular_")]
    families = sorted({row["policy"] for row in candidates})
    validation = {name: float(np.mean([row["validation_loss_mad"] for row in candidates if row["policy"] == name]))
                  for name in families}
    tabular = min(validation, key=lambda name: (validation[name], name))
    dqn = protocol["primary_dqn"]
    if dqn != "dqn_continuous":
        raise ValueError("The archived validation did not select the required continuous DQN")
    metadata = {"tabular_family": tabular, "tabular_validation_mean_mad": validation,
                "dqn_family": dqn, "source_protocol_sha256": FROZEN_PROTOCOL_SHA256,
                "checkpoints": {}}
    policies = {}
    for group, selected in (("tabular", tabular), ("dqn", dqn)):
        rows = sorted((r for r in protocol["models"] if r["policy"] == selected), key=lambda r: r["training_seed"])
        if tuple(r["training_seed"] for r in rows) != SEEDS:
            raise ValueError("Expected all three preselected training seeds")
        policies[group] = {}
        metadata["checkpoints"][group] = rows
        for row in rows:
            relative = Path(row["path"])
            checkpoint = verified_bytes(ARCHIVE, relative, row["path_sha256"])
            if group == "dqn":
                model = DispatchDQN.load(BytesIO(checkpoint), device="cpu")
                if model.device.type != "cpu":
                    raise RuntimeError("DQN checkpoint must run on CPU")
                policy = lambda obs, info, public, model=model: int(model.predict(obs, deterministic=True)[0])
            else:
                field = row["field"]
                table = QTable.from_json(checkpoint,
                                         **({"cardinalities": CARDINALITIES} if field == "refined_state" else {}))
                policy = lambda obs, info, public, table=table, field=field: table.choose(info[field], info["action_mask"])
            policies[group][row["training_seed"]] = policy
    return metadata, policies


def make_plan(physical: FMCGProblem, reserves, forecast: Forecasts, mode: str, hour: int,
              margin: float, budget_s: float, archived: dict | None = None) -> Plan:
    """Keep full physical capacity for replay and deduct the same reserve in planning."""
    start = perf_counter()
    if mode not in PLANNERS:
        raise ValueError("Unknown planning strategy")
    source = {"greedy": "static", "buffered": "ml"}.get(mode, mode)
    hours = np.full(len(forecast.arcs), hour, dtype=int)
    public = forecast.grid(physical, source, hours, margin if mode == "buffered" else 0.)
    effective = planning_capacity(public, reserves)
    signature = fingerprint(asdict(effective))
    if archived is None:
        result = cheapest_insertion(effective) if mode == "greedy" else solve_fmcg(effective, time_limit_s=budget_s)
        tours, status, detail = result.tours, result.status, result.detail
        calls = int(mode != "greedy")
    else:
        if archived["planning_problem_sha256"] != signature:
            raise ValueError("Saved plan belongs to a different planning problem")
        tours = tuple(Tour(t["vehicle"], tuple(t["stops"]), t["departure_s"], tuple(t["service_starts_s"]),
                           t["return_s"], tuple(t["initial_delivery"]) if t["initial_delivery"] is not None else None)
                      for t in archived["tours"])
        status, detail = archived["status"], archived["detail"]
        calls = archived["initial_solver_calls"]
    cost, omissions = None, len(public.stops)
    if status in {"feasible", "optimal"}:
        report = validate_fmcg(effective, tours)
        cost, omissions = report.cost / UNITS_PER_MAD, len(report.unserved)
        active = {tour.vehicle for tour in tours}
        if any(any(map(any, stock)) and index not in active for index, stock in enumerate(reserves)):
            raise RuntimeError("Common reserve carrier must be active in every usable plan")
    seconds = perf_counter() - start if archived is None else archived["planning_wall_s"]
    return Plan(public, tours, status, detail, seconds, calls, cost, omissions, signature)


def realized_travel_seconds(sim: Simulator) -> int:
    """Measure driving from departure of each leg to its recorded arrival."""
    departures, total = {}, 0
    for entry in sim.movements:
        if entry.kind == "leg":
            if entry.vehicle in departures:
                raise RuntimeError("Overlapping driving legs")
            departures[entry.vehicle] = entry.time_s
        elif entry.kind == "arrive":
            total += entry.time_s - departures.pop(entry.vehicle)
    if departures:
        raise RuntimeError("Unclosed driving leg")
    return total


def audit_metrics(sim, metrics: dict, costs: dict, rates: RewardRates, total_reward: float) -> dict:
    """Reconcile independent journal, terminal statistics and incremental reward."""
    completed, orders = sim.completed, sim.orders
    mandatory = {key for key, order in orders.items() if order.stop.penalty_cents is None}
    missing = mandatory - completed.keys()
    urgent = orders.keys() - sim.products.keys()
    on_time = {key for key, time in completed.items() if time <= orders[key].stop.window[1]}
    regular = orders.keys() - urgent
    expected_objective = (metrics["realized_cost_mad"] + metrics["delay_min"] * rates.delay_cents_per_min / 100
                          + metrics["overtime_min"] * rates.overtime_cents_per_min / 100
                          + metrics["route_changes"] * rates.change_cents / 100
                          + metrics["replan_attempts"] * rates.attempt_cents / 100
                          + len(missing) * rates.mandatory_omission_cents / 100)
    checks = [(metrics["objective_mad"], expected_objective), (-total_reward, expected_objective),
              (sum(costs.values()), expected_objective),
              (sum(costs[k] for k in ("fixed_mad", "distance_mad", "time_mad")), metrics["operating_cost_mad"]),
              (costs["omission_mad"], metrics["omission_mad"] + len(missing) * rates.mandatory_omission_cents / 100),
              (metrics["otif"], len(on_time) / len(orders)), (metrics["regular_otif"], len(on_time & regular) / len(regular)),
              (metrics["urgent_served"], len(urgent & completed.keys())),
              (metrics["unserved_mandatory"], len(missing))]
    if not all(isclose(lhs, rhs, rel_tol=0., abs_tol=1e-8) for lhs, rhs in checks):
        raise RuntimeError("Terminal costs or service rates do not reconcile")
    return {"mandatory_total": len(mandatory), "mandatory_completion_rate": 1 - len(missing) / len(mandatory),
            "urgent_service_rate": len(urgent & completed.keys()) / len(urgent) if urgent else None,
            "realized_travel_min": realized_travel_seconds(sim) / 60,
            "physical_violations": 0, "perfect_schedule": metrics["realized_feasible"]}


def policy_runs(dispatch: str, frozen: dict, seed: int):
    if dispatch in frozen:
        return tuple(frozen[dispatch].items())
    policy = random_policy(seed) if dispatch == "random_feasible" else reference_policy(dispatch)
    return ((None, policy),)


def run_day(case: Case, plan: Plan, dispatch: Policy, products, priority, settings: Settings,
            rates: RewardRates, frozen_config: dict, *, input_mode: str = "continuous",
            observe: Callable[[np.ndarray, dict, object, int], None] | None = None
            ) -> tuple[dict, tuple[dict, ...], frozenset[str]]:
    start = perf_counter()
    settings = replace(settings, latency_s=case.draw.latency_s)
    episode = Episode(case.draw.key, lambda: Simulator(plan.problem, plan.tours, products, priority,
                                                       case.reserves, case.scenario, settings))
    env = DQNEnv((episode,), input_mode=input_mode, rates=rates, solver_budget=frozen_config["solver_budget"],
                 probe_budget=frozen_config["probe_budget"], tight_s=frozen_config["tight_slack_s"])
    obs, info = env.reset(seed=case.draw.seed)
    reset_s = perf_counter() - start
    inference_s = total_reward = 0.
    audit_s = 0.
    invalid = steps = eligible = 0
    peak_rss = PROCESS.memory_info().rss
    while True:
        public = env.public_observation
        infer_at = perf_counter()
        action = int(dispatch(obs, info, public))
        inference_s += perf_counter() - infer_at
        if observe is not None:
            observed_at = perf_counter()
            if not env.observation_space.contains(obs):
                raise ValueError("Observation violates the frozen dispatch contract")
            observe(obs, info, public, action)
            audit_s += perf_counter() - observed_at
        eligible += info["action_mask"][1]
        invalid += int(not info["action_mask"][action])
        obs, reward, ended, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if steps % 8 == 0:
            peak_rss = max(peak_rss, PROCESS.memory_info().rss)
        if truncated:
            raise RuntimeError("Benchmark day was truncated")
        if ended:
            break
    sim = env._sim
    metrics = env.episode_metrics()
    costs = audit_metrics(sim, metrics, info["costs_mad"], rates, total_reward)
    clock = tuple((decision.time_s, decision.tick, decision.events) for decision in sim.decisions)
    if len(clock) != steps:
        raise RuntimeError("Decision clock and simulator disagree")
    trace = env.decision_trace()
    wall_s = perf_counter() - start
    return ({**metrics, **costs, "decision_clock_sha256": fingerprint(clock), "replay_wall_s": wall_s,
             "inference_wall_s": inference_s, "decisions": steps, "eligible_decisions": eligible,
             "reset_wall_s": reset_s, "probe_wall_s": env.probe_wall_s,
             "probe_candidates": env.probe_candidates, "probe_exhaustions": env.probe_exhaustions,
             "diagnostic_wall_s": audit_s,
             "invalid_actions": invalid, "route_instability": metrics["route_changes"] / len(case.problem.stops),
             "physical_rss_sample_peak_mib": max(peak_rss, PROCESS.memory_info().rss) / 2**20,
             **{key: info["costs_mad"][key] for key in ("lateness_mad", "overtime_mad", "instability_mad", "attempt_mad")}},
            trace, frozenset(sim.completed.keys() - products.keys()))


def urgent_witnesses(case: Case, served_by: dict[str, set[str]]) -> list[dict]:
    records = []
    for event in case.scenario.events:
        if not isinstance(event, UrgentOrder):
            continue
        order = event.order
        product_index = int(order.stop.refrigerated)
        can_stock = any(v.compatible(order.stop) and within(order.stop.delivery, stock[product_index])
                        for v, stock in zip(case.problem.vehicles, case.reserves, strict=True))
        witnesses = sorted(served_by.get(order.stop.id, ()))
        if witnesses and not can_stock:
            raise RuntimeError("Observed a service impossible under initial stock")
        reason = "witnessed_service" if witnesses else "stock_or_compatibility_impossible" if not can_stock else "undetermined"
        records.append({**case.metadata(), "order": order.stop.id, "product": order.product,
                        "crates": order.stop.delivery[2], "release_s": event.at_s,
                        "deadline_s": order.stop.window[1], "classification": reason,
                        "witness_policies": "|".join(witnesses)})
    return records
