"""Run the final paired Predict–Optimize–Adapt benchmark without training."""

from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict
from math import isfinite
from functools import lru_cache
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
import json

import pandas as pd
import torch
import yaml

from src.benchmark.evaluation import PLANNERS, SEEDS, freeze_approximation, load_dispatch, make_plan, policy_runs, run_day, urgent_witnesses
from src.benchmark.scenarios import Design, build_draws, fingerprint, make_case, traffic_realization
from src.planning.forecast import build_forecasts
from src.rl.dqn_pilot import load_config as dispatch_config
from src.rl.rewards import RewardRates
from src.simulation.experiment import digest, load_reference
from src.simulation.policies import POLICIES
from src.simulation.state import Settings

CONFIG = Path("configs/final_benchmark.yaml")
ROOT = Path("results/experiments")
ALLOWED_OUTPUTS = ("final_benchmark", "final_benchmark_development", "final_benchmark_research")


def save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=sorted, allow_nan=False) + "\n")


def validate_config(config: dict) -> None:
    required = (("greedy", "never"), ("static", "never"), ("historical", "never"), ("ml", "never"),
                ("buffered", "never"), ("buffered", "always"), ("buffered", "periodic"),
                ("buffered", "threshold"), ("buffered", "multi_feature"), ("buffered", "tabular"),
                ("buffered", "dqn"))
    policies = config["policies"]
    for i, (planner, dispatch) in enumerate(required):
        if policies.get(f"B{i}") != {"planner": planner, "dispatch": dispatch}:
            raise ValueError(f"B{i} violates the frozen benchmark comparison")
    known_dispatch = {*POLICIES, "tabular", "dqn", "random_feasible"}
    if any(p["planner"] not in PLANNERS or p["dispatch"] not in known_dispatch for p in policies.values()):
        raise ValueError("Unknown planner or dispatch policy")
    for profile in config["profiles"].values():
        if (not all(type(profile[name]) is int and profile[name] > 0
                    for name in ("namespace", "dates", "repeats", "sensitivity_dates", "wall_limit_s"))
                or not isfinite(profile["solver_budget_s"]) or profile["solver_budget_s"] <= 0):
            raise ValueError("Invalid runtime profile")
    if len({p["namespace"] for p in config["profiles"].values()}) != len(config["profiles"]):
        raise ValueError("Profile seed namespaces must differ")
    for values in config["regimes"].values():
        design = Design(**values)
        if (any(not isfinite(item) or item <= 0 for item in (design.demand, design.pickup))
                or not isfinite(design.traffic_factor) or design.traffic_factor < 1
                or any(type(item) is not int or item < 0 for item in
                       (design.reserve, design.service_extra_s, design.urgents))
                or not config["urgent_service_s"] <= design.urgent_window_s
                or not 1 <= design.urgent_crates[0] <= design.urgent_crates[1]):
            raise ValueError("Incorrect types or bounds in scenario design")
    sensitivity = config["sensitivity"]
    if (not set(sensitivity["policies"]) <= policies.keys()
            or any(policies[key]["planner"] != "buffered" for key in sensitivity["policies"])):
        raise ValueError("Sensitivity requires frozen dispatch under buffered planning")
    expected = {"urgent_penalty_cents", "latency_s", "reserve", "traffic_factor",
                "service_extra_s", "closing_shift_s", "unavailable"}
    fields = [v["field"] for v in sensitivity["variants"].values()]
    if set(fields) != expected or any(fields.count(field) != 2 for field in expected):
        raise ValueError("Sensitivity requires two levels for each predeclared factor")


def verify_fairness(frame: pd.DataFrame, config: dict) -> None:
    for case, group in frame.groupby("case"):
        ids = config["policies"] if group.block.iloc[0] == "benchmark" else config["sensitivity"]["policies"]
        if set(group.policy) != set(ids):
            raise ValueError(f"Missing configured policies for {case}")
        for policy, rows in group.groupby("policy"):
            kind = config["policies"][policy]["dispatch"]
            seeds = tuple(sorted(rows.training_seed.dropna().astype(int)))
            if kind in {"tabular", "dqn"}:
                if seeds != SEEDS or len(rows) != len(SEEDS):
                    raise ValueError(f"Missing or repeated learned seed in {case}/{policy}")
            elif seeds or len(rows) != 1:
                raise ValueError(f"Repeated classical policy in {case}/{policy}")
        if group.scenario_sha256.nunique() != 1 or group.traffic_sha256.nunique() != 1:
            raise ValueError(f"Shared experimental inputs diverged in {case}")
        usable = group[group.dispatched]
        if usable.decision_clock_sha256.nunique() > 1:
            raise ValueError(f"Decision opportunities diverged in {case}")
        for _, rows in group.groupby("planner"):
            if rows.plan_sha256.nunique() != 1:
                raise ValueError(f"Identical planner should be reused in {case}")


def run(profile: str, output: Path, replay_plans: Path | None = None) -> None:
    started = perf_counter()
    config = yaml.safe_load(CONFIG.read_text())
    validate_config(config)
    scope = config["profiles"][profile]
    root = ROOT.resolve()
    target = output.resolve()
    if target in (root, *root.parents) or (root in target.parents and target.parent == root
                                          and target.name not in ALLOWED_OUTPUTS):
        raise ValueError("Choose a dedicated new output directory")
    if target.is_relative_to(root) and target.relative_to(root).parts[0] not in ALLOWED_OUTPUTS:
        raise ValueError("Benchmark output overlaps previous phases")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    reference, matrix, base, model, history, _, products, priority = load_reference()
    checkpoint_info, frozen = load_dispatch()
    frozen_config = dispatch_config()
    settings, rates = Settings(**frozen_config["dispatch"]), RewardRates(**frozen_config["reward"])
    hour, margin = reference["reference_hour"], reference["buffer_ratio"]
    save(output / "frozen_protocol.json", {
        "phase": 10, "checkpoint": "9a214e3", "profile": profile, "config": config,
        "config_sha256": digest(CONFIG), "phase9b_dispatch_settings": asdict(settings),
        "rates": asdict(rates), "traffic_model_sha256": digest(Path("models/traffic_model.joblib")),
        "traffic_train_table_sha256": reference["traffic_sha256"], "osrm_sha256": reference["matrix_sha256"],
        "reference_hour": hour, "buffer_ratio": margin, "checkpoint_selection": checkpoint_info,
        "history_ratios": [[*k, v] for k, v in sorted(history.ratios.items())], "history_global": history.fallback,
        "plan_source_sha256": digest(replay_plans) if replay_plans else None,
    })
    if profile != "development":
        freeze_approximation(output)
    draws = build_draws(config, profile, settings.latency_s)
    for name, file in (("q_learning", "scenario_manifest.json"), ("q_learning_refinement", "scenario_manifest.json"),
                       ("dqn_pilot", "scenario_manifest.json"), ("dqn_evaluation", "final_scenarios.json")):
        historical = json.loads((ROOT / name / file).read_text())
        if ({row.seed for row in draws} & {r["seed"] for r in historical}
                or {row.day.isoformat() for row in draws} & {r["day"] for r in historical}):
            raise ValueError("Held-out bank overlaps prior experimental dates or seeds")
    saved = ({(tuple(row["planning_key"]), row["planner"]): row
              for row in json.loads(replay_plans.read_text())} if replay_plans else {})
    records, traces, scenarios, predictions, urgent, plans = [], [], [], [], [], {}

    @lru_cache(maxsize=1)
    def forecast_for(day):
        begin = perf_counter()
        forecast = build_forecasts(matrix, day, model, history)
        return forecast, perf_counter() - begin

    @lru_cache(maxsize=1)
    def traffic_for(day, seed):
        draw = next(row for row in draws if row.day == day and row.seed == seed)
        forecast, _ = forecast_for(day)
        grid, errors = traffic_realization(base, forecast, draw, matrix.ids, model.zones, hour)
        return grid, fingerprint(grid), errors

    def flush():
        for name, rows in (("scenario_metrics", records), ("decisions", traces), ("prediction_errors", predictions),
                           ("urgent_diagnostics", urgent)):
            pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
        save(output / "scenarios.json", scenarios)
        save(output / "plans.json", [item.record(key, mode) for (key, mode), item in plans.items()])

    try:
        for draw in draws:
            if perf_counter() - started >= scope["wall_limit_s"]:
                raise TimeoutError("Benchmark reached its configured wall-time bound")
            forecast, forecast_s = forecast_for(draw.day)
            realized, traffic_hash, errors = traffic_for(draw.day, draw.seed)
            case = make_case(base, draw, config, matrix.ids, model.zones, realized, traffic_hash, errors)
            entry = case.metadata()
            scenarios.append({**entry, "design": asdict(draw.design), "vehicles": [v.id for v in case.problem.vehicles],
                              "starting_reserves": case.reserves, "stops": [asdict(s) for s in case.problem.stops],
                              "events": [{**asdict(e), "kind": type(e).__name__} for e in case.scenario.events]})
            if draw.variant == "reference":
                predictions.extend({**entry, **error} for error in case.traffic_error)
            selection = config["policies"] if draw.block == "benchmark" else config["sensitivity"]["policies"]
            served_by = defaultdict(set)
            key = draw.planning_key()
            key = (key[0].isoformat(), *key[1:])
            for label in selection:
                pair = config["policies"][label]
                mode = pair["planner"]
                plan_key = (key, mode)
                if plan_key not in plans:
                    plans[plan_key] = make_plan(case.problem, case.reserves, forecast, mode, hour, margin,
                                                scope["solver_budget_s"], saved[plan_key] if replay_plans else None)
                plan = plans[plan_key]
                plan_entry = {"plan_sha256": fingerprint((plan.fingerprint, [asdict(t) for t in plan.tours])),
                              "planning_status": plan.status, "planning_detail": plan.detail,
                              "planned_cost_mad": plan.cost_mad, "initial_solver_calls": plan.solver_calls,
                              "initial_planning_wall_s": plan.planning_wall_s,
                              "forecast_wall_s": forecast_s if mode in {"historical", "ml", "buffered"} else 0.,
                              "planner": mode, "dispatch": pair["dispatch"], "dispatched": plan.usable}
                random_seed = int(fingerprint((config["random_dispatch_seed"], draw.seed))[:8], 16)
                for training_seed, policy in policy_runs(pair["dispatch"], frozen, random_seed):
                    row = {**entry, **plan_entry, "policy": label, "training_seed": training_seed}
                    if plan.usable:
                        metrics, decisions, urgent_served = run_day(case, plan, policy, products, priority,
                                                                    settings, rates, frozen_config)
                        replay_s = metrics["replay_wall_s"]
                        row.update(metrics, end_to_end_wall_s=replay_s + row["forecast_wall_s"] + plan.planning_wall_s,
                                   total_solver_calls=plan.solver_calls + metrics["solver_calls"],
                                   inference_ms_per_decision=1000 * metrics["inference_wall_s"] / metrics["decisions"],
                                   invalid_action_frequency=metrics["invalid_actions"] / metrics["decisions"],
                                   safety_fallback_frequency=metrics["safety_fallbacks"] / metrics["decisions"])
                        traces.extend({"case": draw.key, "policy": label, "training_seed": training_seed,
                                       **decision} for decision in decisions)
                        for order in urgent_served:
                            served_by[order].add(label)
                    records.append(row)
            urgent.extend(urgent_witnesses(case, served_by))
            print(f"{draw.key} rows={len(records)} elapsed_s={perf_counter() - started:.1f}", flush=True)
    finally:
        flush()
    frame = pd.DataFrame(records)
    verify_fairness(frame, config)
    save(output / "manifest.json", {
        "profile": profile, "status": "complete", "scenarios": len(draws), "rows": len(frame),
        "planning_cases": len(plans), "dispatched_rows": int(frame.dispatched.sum()),
        "wall_s": perf_counter() - started, "peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
        "torch": torch.__version__, "threads": config["threads"], "historical_estimation": "frozen train-period grouping",
    })


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("development", "standard", "research"), default="development")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-plans", type=Path)
    args = parser.parse_args()
    suffix = "" if args.profile == "standard" else f"_{args.profile}"
    run(args.profile, args.output or ROOT / f"final_benchmark{suffix}", args.replay_plans)


if __name__ == "__main__":
    main()
