"""Instrument bounded planning and frozen dispatch without altering their decisions."""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
from stable_baselines3.common.save_util import load_from_pkl

from src.benchmark.evaluation import ARCHIVE, Plan, policy_runs, run_day
from src.benchmark.scenarios import Case, fingerprint
from src.optimization.fmcg_greedy import cheapest_insertion
from src.optimization.fmcg_solver import solve_fmcg
from src.optimization.fmcg_validator import UNITS_PER_MAD, validate_fmcg
from src.rl.q_learning import QTable
from src.rl.refined_observations import CARDINALITIES, CONTINUOUS_NAMES
from src.simulation.experiment import digest
from src.optimization.fmcg_problem import ZERO
from src.simulation.state import plus

POLICIES = {"B4": "never", "B5": "always", "B8": "multi_feature", "B9": "tabular", "B10": "dqn"}


def planning_row(instance, mode: str, problem, budget_s=None):
    first = None

    def incumbent(seconds: float):
        nonlocal first
        if first is None:
            first = seconds

    start = perf_counter()
    if mode == "witness":
        tours, status, detail = instance.witness, "feasible", "Independently validated constructive witness"
    elif mode == "greedy":
        result = cheapest_insertion(problem)
        tours, status, detail = result.tours, result.status, result.detail
    else:
        initial = instance.witness if mode == "ortools_warm" else None
        result = solve_fmcg(problem, time_limit_s=budget_s, initial_tours=initial, on_incumbent=incumbent)
        tours, status, detail = result.tours, result.status, result.detail
    elapsed = perf_counter() - start
    usable = status in {"feasible", "optimal"}
    row = {"planner": mode, "budget_s": budget_s, "status": status, "detail": detail, "usable": usable,
           "planning_wall_s": elapsed, "first_incumbent_s": first if mode.startswith("ortools") else
           elapsed if mode == "greedy" and usable else None,
           "first_incumbent_source": "native_callback" if first is not None else
           "greedy_completion" if mode == "greedy" and usable else None,
           "solver_calls": int(mode.startswith("ortools")),
           "mandatory_omissions": sum(s.penalty_cents is None for s in problem.stops)
           - sum(problem.stops[i - 1].penalty_cents is None for t in tours for i in t.stops)}
    if usable:
        report = validate_fmcg(problem, tours)
        physical = instance.problem
        stats = ("grams", "liters", "crates", "pallets")
        max_usage = []
        for tour, path in zip(tours, report.tours, strict=True):
            vehicle = physical.vehicles[tour.vehicle]
            stock = plus(*instance.reserves[tour.vehicle])
            max_usage.append(tuple(max((load.onboard[i] + stock[i]) / cap for load in path.loads)
                                   if cap else 0. for i, cap in enumerate(vehicle.capacity)))
        available_seconds = [b - a for a, b in map(problem.operating_window, problem.vehicles)]
        row.update(cost_mad=report.cost / UNITS_PER_MAD, distance_km=sum(p.distance_m for p in report.tours) / 1000,
                   vehicles_used=len(tours), vehicle_use_rate=len(tours) / len(problem.vehicles),
                   drive_service_wait_hours=sum(p.duration_s for p in report.tours) / 3600,
                   shift_use_rate=sum(p.duration_s for p in report.tours) / sum(available_seconds),
                   planned_otif=(len(problem.stops) - len(report.unserved)) / len(problem.stops),
                   min_window_slack_s=min(problem.stops[i - 1].window[1] - problem.stops[i - 1].service_s - start
                                          for t in tours for i, start in zip(t.stops, t.service_starts_s, strict=True)),
                   max_route_stops=max(len(t.stops) for t in tours),
                   **{f"mean_peak_capacity_{unit}": float(np.mean([x[i] for x in max_usage]))
                      for i, unit in enumerate(stats)})
    return row, tours


def training_support(checkpoints: dict) -> dict:
    result = {}
    for row in checkpoints["checkpoints"]["dqn"]:
        path = ARCHIVE / row["replay"]
        if digest(path) != row["replay_sha256"]:
            raise ValueError("DQN training support changed")
        replay = load_from_pkl(path, 0)
        states = replay.observations[:replay.size(), 0, :]
        if states.shape[1] != len(CONTINUOUS_NAMES) or not np.isfinite(states).all():
            raise ValueError("Archived replay does not match the trained observation schema")
        result["B10", row["training_seed"]] = (states.min(axis=0), states.max(axis=0))
    for row in checkpoints["checkpoints"]["tabular"]:
        field = row["field"]
        table = QTable.load(ARCHIVE / row["path"], **({"cardinalities": CARDINALITIES}
                                                     if field == "refined_state" else {}))
        result["B9", row["training_seed"]] = (field, table.counts)
    return result


def observe_dispatch(case: Case, instance, tours, plan_source, labels, candidates: int, frozen, support,
                     settings, rates, frozen_config):
    active = {tour.vehicle for tour in tours}
    reserves = tuple(stock if i in active else (ZERO, ZERO) for i, stock in enumerate(instance.reserves))
    case = replace(case, reserves=reserves)
    shared_plan = Plan(instance.problem, tours, "feasible", plan_source, 0., 0, None, 0,
                       fingerprint([asdict(tour) for tour in tours]))
    settings = replace(settings, candidate_limit=candidates)
    records, trace, observations = [], [], []
    for label in labels:
        for seed, decide in policy_runs(POLICIES[label], frozen, case.draw.seed):
            observed = []

            def monitor(obs, info, public, action):
                state = tuple(info["refined_state"])
                if any(value < 0 or value >= bound for value, bound in zip(state, CARDINALITIES, strict=True)):
                    raise ValueError("Invalid refined tabular state")
                eligible = bool(info["action_mask"][1])
                outside = ()
                absent, no_known_action = None, None
                if label == "B10":
                    low, high = support[label, seed]
                    outside = tuple(CONTINUOUS_NAMES[i] for i in np.flatnonzero((obs < low) | (obs > high)))
                elif label == "B9":
                    state_field, counts = support[label, seed]
                    prior = counts.get(tuple(info[state_field]))
                    absent = prior is None
                    no_known_action = absent or not any(prior[np.asarray(info["action_mask"], dtype=bool)])
                observed.append({"decision_index": len(observed), "time_s": public.now_s,
                                 "events": "|".join(public.event_types), "eligible": eligible,
                                 "waiting": len(public.waiting_urgent), "action": action,
                                 "max_remaining_route": max((len(v.tail) for v in public.vehicles), default=0),
                                 "remaining_nodes": sum(len(v.tail) for v in public.vehicles),
                                 "tabular_state": str(state), "outside_features": "|".join(outside),
                                 "ood_box": bool(outside) if label == "B10" else None,
                                 "unseen_table_state": absent, "no_trained_available_action": no_known_action,
                                 **{f"feature_{name}": float(value) for name, value in zip(CONTINUOUS_NAMES, obs, strict=True)}})

            metrics, details, _ = run_day(case, shared_plan, decide, instance.products, instance.priority,
                                         settings, rates, frozen_config, observe=monitor)
            keys = {"policy": label, "training_seed": seed, "candidate_limit": candidates}
            effective = [r for r in details if r["attempted"]]
            eligible = [r for r in observed if r["eligible"]]
            records.append({**keys, **metrics, "plan_sha256": shared_plan.fingerprint, "plan_source": plan_source,
                            "simulation_with_probes_wall_s": metrics["replay_wall_s"] - sum(metrics[k] for k in (
                                "planning_wall_s", "inference_wall_s", "diagnostic_wall_s")),
                            "regular_loss_per_customer_mad": metrics["objective_mad"] / len(case.problem.stops),
                            "mean_policy_inference_ms": 1000 * metrics["inference_wall_s"] / len(observed),
                            "replan_p95_ms": float(1000 * np.quantile([r["planning_wall_s"] for r in effective], .95))
                            if effective else None,
                            "candidate_limit_hits": sum(r["search_items"] == candidates and not r["solver_calls"]
                                                        for r in effective),
                            "ood_box_eligible": sum(r["ood_box"] for r in eligible) if label == "B10" else None,
                            "unseen_table_eligible": sum(r["unseen_table_state"] for r in eligible)
                            if label == "B9" else None,
                            "unknown_action_eligible": sum(r["no_trained_available_action"] for r in eligible)
                            if label == "B9" else None,
                            "loaded_reserve_crates": sum(a[2] + b[2] for a, b in reserves)})
            trace.extend({**keys, **r} for r in details)
            observations.extend({**keys, **r} for r in observed)
    if len({row["decision_clock_sha256"] for row in records}) > 1:
        raise ValueError("Paired policies have different exogenous decision opportunities")
    return records, trace, observations
