"""Train sparse Q-tables and compare dispatch policies on held-out paired cases."""

from argparse import ArgumentParser
from dataclasses import asdict, replace
from hashlib import sha256
from math import prod, isclose
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
import json

import numpy as np
import pandas as pd
import yaml

from src.rl.actions import DispatchAction
from src.rl.environment import DistributionEnv
from src.rl.observations import CARDINALITIES
from src.rl.q_learning import Parameters, QTable
from src.rl.rewards import RewardRates
from src.rl.scenarios import Case, build_banks
from src.simulation.policies import POLICIES
from src.simulation.state import Settings

CONFIG = Path("configs/rl_training.yaml")
MARK = {0: DispatchAction.KEEP_PLAN, 1: DispatchAction.LOCAL_HEURISTIC, 2: DispatchAction.GLOBAL_REOPTIMIZATION}
METRICS = ("otif", "regular_otif", "urgent_served", "operating_cost_mad", "realized_cost_mad", "delay_min",
           "route_changes", "replan_attempts", "replans_applied", "safety_fallbacks", "solver_calls", "objective_mad")


def rollout(env: DistributionEnv, policy, *, seed: int, scenario_index: int | None = None,
            learning: QTable | None = None) -> dict:
    started = perf_counter()
    state, info = env.reset(seed=seed, options={"scenario_index": scenario_index})
    total = inference_s = 0.
    seen = unseen = decisions = 0
    while True:
        key = tuple(map(int, state))
        before = perf_counter()
        action = policy(key, info, env.public_observation)
        inference_s += perf_counter() - before
        if info["action_mask"][1]:
            decisions += 1
            if isinstance(policy, QTablePolicy):
                seen += key in policy.table.values
                unseen += key not in policy.table.values
        following, reward, terminal, truncated, step_info = env.step(action)
        total += reward
        if learning is not None:
            learning.update(key, int(action), reward, tuple(map(int, following)), step_info["action_mask"],
                            terminated=terminal, elapsed_s=step_info["elapsed_s"])
        if terminal or truncated:
            if truncated:
                raise RuntimeError("Scientific experiments require complete distribution days")
            break
        state, info = following, step_info
    metrics = env.episode_metrics()
    rates = env.rates
    expected = (metrics["realized_cost_mad"] + metrics["delay_min"] * rates.delay_cents_per_min / 100
                + metrics["overtime_min"] * rates.overtime_cents_per_min / 100
                + metrics["route_changes"] * rates.change_cents / 100
                + metrics["replan_attempts"] * rates.attempt_cents / 100
                + metrics["unserved_mandatory"] * rates.mandatory_omission_cents / 100)
    if not isclose(-total, expected, rel_tol=0., abs_tol=1e-7):
        raise RuntimeError("Reward and independent end-of-day accounting did not reconcile")
    return {**metrics, "episode_wall_s": perf_counter() - started, "inference_s": inference_s,
            "episode_return": total, "actionable_decisions": decisions, "seen_decisions": seen,
            "unseen_decisions": unseen, "scenario_key": env.episode_key}


class QTablePolicy:
    def __init__(self, table: QTable, rng=None, epsilon: float = 0.):
        self.table = table
        self.rng = rng
        self.epsilon = epsilon

    def __call__(self, state, info, observation):
        return self.table.choose(state, info["action_mask"], self.rng, self.epsilon)


def random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def choose(state, info, observation):
        return int(rng.choice(np.flatnonzero(info["action_mask"])))

    return choose


def reference_policy(name: str):
    return lambda state, info, current: MARK[POLICIES[name](current)] if info["action_mask"][1] else DispatchAction.KEEP_PLAN


def make_env(cases: tuple[Case, ...], dispatch: Settings, config: dict, penalty: int | None = None) -> DistributionEnv:
    return DistributionEnv([case.episode(dispatch, penalty) for case in cases],
                           rates=RewardRates(**config["reward"]), solver_budget=config["solver_budget"])


def train(env: DistributionEnv, parameters: Parameters, seed: int, episodes: int, config: dict,
          output: Path | None = None, callback=None) -> tuple[QTable, list[dict]]:
    table = QTable(parameters)
    history = []
    for number in range(1, episodes + 1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, number]))
        epsilon = config["epsilon_start"] + (config["epsilon_end"] - config["epsilon_start"]) * (
            (number - 1) / max(1, episodes - 1)
        )
        policy = QTablePolicy(table, rng, epsilon)
        metrics = rollout(env, policy, seed=int(rng.integers(2**31)), learning=table)
        history.append({"training_seed": seed, "episode": number, "epsilon": epsilon,
                        "visited_states": len(table.values), "state_action_pairs": sum(
                            np.count_nonzero(count) for count in table.counts.values()),
                        **{k: metrics[k] for k in ("scenario_key", "episode_return", "episode_wall_s",
                                                   "replan_attempts", "replans_applied", "solver_calls",
                                                   "planning_wall_s", "otif", "urgent_served", "safety_fallbacks")}})
        checkpoint = number == episodes or number % config["checkpoint_every"] == 0
        if output is not None and checkpoint:
            path = output / f"seed_{seed}_episode_{number:04}.json"
            table.save(path, training_seed=seed, episode=number)
            if callback is not None:
                callback(number, table)
    return table, history


def evaluate_cases(env: DistributionEnv, cases: tuple[Case, ...], policy, name: str,
                   training_seed: int | None = None, trace: list | None = None):
    for index, case in enumerate(cases):
        metrics = rollout(env, policy, seed=case.seed, scenario_index=index)
        case_key = {"policy": name, "training_seed": training_seed, "day": str(case.day),
                    "case": case.key, "regime": case.regime}
        if trace is not None:
            trace.extend({**case_key, **decision} for decision in env.decision_trace())
        yield {**case_key, **metrics}


def paired_intervals(frame: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregated = frame.groupby(["policy", "day", "case", "regime"], as_index=False)[list(METRICS)].mean()
    learned = aggregated.loc[aggregated.policy.eq("q_learning")].set_index(["day", "case", "regime"])
    rows, intervals = [], []
    rng = np.random.default_rng(seed)
    days = sorted(learned.index.get_level_values("day").unique())
    resamples = rng.integers(len(days), size=(draws, len(days)))
    for name in sorted(set(aggregated.policy) - {"q_learning"}):
        baseline = aggregated.loc[aggregated.policy.eq(name)].set_index(["day", "case", "regime"])
        if not learned.index.equals(baseline.index):
            raise RuntimeError("Unpaired operational comparison")
        delta = learned[list(METRICS)] - baseline[list(METRICS)]
        rows.append(delta.reset_index().assign(baseline=name))
        samples = delta.groupby(level="day").mean().loc[days].to_numpy()[resamples].mean(axis=1)
        bounds = np.quantile(samples, [.025, .975], axis=0)
        for i, m in enumerate(METRICS):
            intervals.append({"baseline": name, "metric": m, "difference": delta[m].mean(),
                              "ci_low": bounds[0, i], "ci_high": bounds[1, i], "pairs": len(delta),
                              "day_blocks": len(days)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(intervals)


def compare(cases, dispatch, config, models):
    test = make_env(cases, dispatch, config)
    rows, trace = [], []
    for name in POLICIES:
        rows.extend(evaluate_cases(test, cases, reference_policy(name), name, trace=trace))
    rows.extend(evaluate_cases(test, cases, random_policy(config["bootstrap_seed"]), "random_feasible", trace=trace))
    for seed, table in models.items():
        rows.extend(evaluate_cases(test, cases, QTablePolicy(table), "q_learning", seed, trace=trace))
    frame = pd.DataFrame(rows)
    for _, group in frame.groupby("case"):
        if group.opportunities.nunique() != 1 or group.unserved_mandatory.any():
            raise RuntimeError("Paired decision opportunities or mandatory service failed")
    differences, intervals = paired_intervals(frame, config["bootstrap_draws"], config["bootstrap_seed"])
    return frame, trace, differences, intervals


def scenario_rows(banks):
    for split, cases in banks.items():
        for case in cases:
            yield {"split": split, "case": case.key, "day": str(case.day), "seed": case.seed,
                   "regime": case.regime, "customers": len(case.problem.stops),
                   "delivery_crates": sum(stop.delivery[2] for stop in case.problem.stops),
                   "vehicles": [vehicle.id for vehicle in case.problem.vehicles],
                   "reserve": case.reserves,
                   "event_times_s": [event.at_s for event in case.scenario.events],
                   "urgent": [asdict(event.order) for event in case.scenario.events if hasattr(event, "order")]}


def write_table(output: Path, name: str, rows):
    pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)


def run(profile: str, output: Path) -> None:
    started = perf_counter()
    config = yaml.safe_load(CONFIG.read_text())
    budgets = config["profiles"][profile]
    dispatch = Settings(**config["dispatch"])
    banks = build_banks(config, profile)
    training = make_env(banks["train"], dispatch, config)
    validation = make_env(banks["validation"], dispatch, config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(CONFIG.read_text())
    (output / "scenario_manifest.json").write_text(json.dumps(list(scenario_rows(banks)), indent=2) + "\n")

    tuning, tuning_log = [], []
    candidates = [Parameters(**row) for row in config["candidates"]]
    if budgets["tuning_episodes"]:
        for index, candidate in enumerate(candidates):
            table, log = train(training, candidate, budgets["seeds"][0], budgets["tuning_episodes"], config)
            tuning_log.extend({"candidate": index, **item} for item in log)
            runs = list(evaluate_cases(validation, banks["validation"], QTablePolicy(table), "q_learning"))
            tuning.append({"candidate": index, **asdict(candidate),
                           "validation_objective_mad": np.mean([r["objective_mad"] for r in runs]),
                           "validation_otif": np.mean([r["otif"] for r in runs])})
    selected = min(tuning, key=lambda row: row["validation_objective_mad"])["candidate"] if tuning else 0
    parameters = candidates[selected]
    if tuning:
        write_table(output, "tuning", tuning)
        write_table(output, "tuning_training_log", tuning_log)

    models, training_log, curves = {}, [], []
    model_path = output / "checkpoints"
    for seed in budgets["seeds"]:
        def validation_checkpoint(number, table):
            rows = list(evaluate_cases(validation, banks["validation"], QTablePolicy(table), "q_learning", seed))
            curves.append({"training_seed": seed, "episode": number, "states": len(table.values),
                           "mean_objective_mad": np.mean([r["objective_mad"] for r in rows]),
                           "mean_otif": np.mean([r["otif"] for r in rows]),
                           "unseen_actionable_pct": 100 * sum(r["unseen_decisions"] for r in rows) /
                           max(1, sum(r["actionable_decisions"] for r in rows))})

        table, log = train(training, parameters, seed, budgets["episodes"], config, model_path, validation_checkpoint)
        path = model_path / f"seed_{seed}_episode_{budgets['episodes']:04}.json"
        saved = QTable.load(path)
        for key in table.values:
            if not np.array_equal(table.values[key], saved.values[key]) or not np.array_equal(table.counts[key],
                                                                                              saved.counts[key]):
                raise RuntimeError("Saved Q-table failed reload verification")
        models[seed] = saved
        training_log.extend(log)
        print(f"seed={seed} episodes={len(log)} states={len(table.values)} "
              f"mean_episode_s={np.mean([row['episode_wall_s'] for row in log]):.4f}", flush=True)

    frame, trace, differences, intervals = compare(banks["test"], dispatch, config, models)

    sensitivity = []
    variants = [(4000, 20), (8000, 5), (8000, 20), (8000, 60), (12000, 20)]
    for penalty, latency in variants:
        env = make_env(banks["validation"], replace(dispatch, latency_s=latency), config, penalty)
        for name, policy in (("never", reference_policy("never")), ("multi_feature", reference_policy("multi_feature")),
                             ("q_learning", QTablePolicy(models[budgets["seeds"][0]]))):
            for row in evaluate_cases(env, banks["validation"], policy, name):
                sensitivity.append({"penalty_mad": penalty / 100, "latency_s": latency, **row})
    coverage = [{"training_seed": seed, "state": key, "action": DispatchAction(action).name,
                 "visits": int(n), "q_mad": float(table.values[key][action])}
                for seed, table in models.items() for key, counts in sorted(table.counts.items())
                for action, n in enumerate(counts)]
    for name, data in (("training_log", training_log), ("validation_curves", curves), ("state_coverage", coverage),
                       ("test_cases", frame), ("test_decisions", trace),
                       ("paired_differences", differences), ("paired_intervals", intervals),
                       ("validation_sensitivity", sensitivity)):
        write_table(output, name, data)
    (output / "manifest.json").write_text(json.dumps({
        "checkpoint": "f183ede", "profile": profile, "config_sha256": sha256(CONFIG.read_bytes()).hexdigest(),
        "settings": budgets, "selected_candidate": selected, "selected_parameters": asdict(parameters),
        "partitions": {key: len(value) for key, value in banks.items()},
        "theoretical_states": prod(CARDINALITIES), "canonical_states_upper_bound": 49,
        "wall_s": perf_counter() - started, "peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
    }, indent=2) + "\n")
    print(frame.groupby("policy")[list(METRICS)].mean().round(4).to_string())


def stress(source: Path, output: Path) -> None:
    """Evaluate standard checkpoints on a larger fleet without further training."""
    started = perf_counter()
    config = yaml.safe_load((source / "config.yaml").read_text())
    source_manifest = json.loads((source / "manifest.json").read_text())
    if source_manifest["profile"] != "standard":
        raise ValueError("Research stress requires the standard trained checkpoints")
    settings = source_manifest["settings"]
    paths = {seed: source / "checkpoints" / f"seed_{seed}_episode_{settings['episodes']:04}.json"
             for seed in settings["seeds"]}
    models = {seed: QTable.load(path) for seed, path in paths.items()}
    banks = build_banks(config, "research", only=("test",))
    original_cases = json.loads((source / "scenario_manifest.json").read_text())
    if {row["seed"] for row in original_cases} & {case.seed for case in banks["test"]}:
        raise ValueError("Research stress must use fresh scenario draws")
    frame, trace, differences, intervals = compare(banks["test"], Settings(**config["dispatch"]), config, models)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scenario_manifest.json").write_text(json.dumps(list(scenario_rows(banks)), indent=2) + "\n")
    for name, rows in (("test_cases", frame), ("test_decisions", trace), ("paired_differences", differences),
                       ("paired_intervals", intervals)):
        write_table(output, name, rows)
    (output / "manifest.json").write_text(json.dumps({
        "profile": "research_stress", "config_sha256": sha256((source / "config.yaml").read_bytes()).hexdigest(),
        "source_models": {seed: sha256(path.read_bytes()).hexdigest() for seed, path in paths.items()},
        "cases": len(banks["test"]), "wall_s": perf_counter() - started,
        "peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
    }, indent=2) + "\n")
    print(frame.groupby("policy")[list(METRICS)].mean().round(4).to_string())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("development", "standard", "research"), default="development")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate-checkpoints", type=Path, help="Standard results for research stress evaluation")
    args = parser.parse_args()
    if args.evaluate_checkpoints:
        if args.profile != "research":
            parser.error("--evaluate-checkpoints requires --profile research")
        stress(args.evaluate_checkpoints, args.output or Path("results/experiments/q_learning_research_stress"))
    else:
        suffix = "" if args.profile == "standard" else f"_{args.profile}"
        run(args.profile, args.output or Path(f"results/experiments/q_learning{suffix}"))


if __name__ == "__main__":
    main()
