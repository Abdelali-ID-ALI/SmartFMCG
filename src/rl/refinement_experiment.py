"""Phase 8.1 factorial comparison with a sealed, fresh paired evaluation bank."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from math import isclose, prod
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
import json

import numpy as np
import pandas as pd
import yaml

from src.rl.actions import DispatchAction, optimizer
from src.rl.experiment import QTablePolicy, random_policy, reference_policy, scenario_rows, write_table
from src.rl.observations import CARDINALITIES as ORIGINAL_CARDS
from src.rl.q_learning import Parameters, QTable
from src.rl.refined_observations import CARDINALITIES, CONTINUOUS_NAMES
from src.rl.refined_rewards import Feedback
from src.rl.refinement_env import RefinementEnv
from src.rl.refinement_scenarios import ARCHIVES, fresh_banks
from src.rl.rewards import RewardRates
from src.simulation.metrics import evaluate
from src.simulation.policies import Action, POLICIES
from src.simulation.state import Settings

CONFIG = Path("configs/rl_refinement.yaml")
LEGACY = Path("configs/rl_training.yaml")
ROOT = Path("results/experiments")
VARIANTS = {"A": ("original", "original"), "B": ("tabular", "original"),
            "C": ("original", "early"), "D": ("tabular", "early")}
METRICS = ("objective_mad", "otif", "regular_otif", "urgent_service_rate", "operating_cost_mad",
           "realized_cost_mad", "delay_min", "replan_attempts", "replans_applied", "route_changes",
           "solver_calls", "safety_fallbacks")


def objective(metrics, rates):
    return (metrics["realized_cost_mad"] + metrics["delay_min"] * rates.delay_cents_per_min / 100
            + metrics["overtime_min"] * rates.overtime_cents_per_min / 100
            + metrics["route_changes"] * rates.change_cents / 100
            + metrics["replan_attempts"] * rates.attempt_cents / 100
            + metrics["unserved_mandatory"] * rates.mandatory_omission_cents / 100)


def make_env(cases, config, encoding):
    dispatch = Settings(**config["dispatch"])
    return RefinementEnv([case.episode(dispatch) for case in cases], encoding=encoding,
                         tight_s=config["tight_slack_s"], probe_budget=config["probe_budget"],
                         rates=RewardRates(**config["reward"]), solver_budget=config["solver_budget"])


def rollout(env, policy, *, seed, case_index=None, table=None, mode="original", weight=.25, scale=1.,
            traces=None):
    started = perf_counter()
    observation, info = env.reset(seed=seed, options={"scenario_index": case_index})
    sim = env._sim
    feedback = Feedback(mode, weight, scale)
    parameters = table.parameters if table is not None else Parameters(.12, 1.)
    initial = feedback.initial(env.public_observation, sim.dispatch_forecast(), set(sim.products))
    total = signal = discounted = discounted_signal = inference_time = 0.
    factor = 1.
    actionable = unseen = 0
    while True:
        state = tuple(map(int, observation))
        if info["action_mask"][1]:
            actionable += 1
            if isinstance(policy, QTablePolicy):
                unseen += state not in policy.table.values
        before = perf_counter()
        action = policy(state, info, env.public_observation)
        inference_time += perf_counter() - before
        next_obs, base, ended, truncated, step = env.step(action)
        discount = parameters.discount(step["elapsed_s"])
        measured = feedback.transform(base, sim, env.public_observation,
                                      sim.dispatch_forecast() if not ended and mode in {"early", "potential"} else None,
                                      discount, ended)
        total += base
        signal += measured
        discounted += factor * base
        discounted_signal += factor * measured
        factor *= discount
        if table is not None:
            table.update(state, int(action), measured, tuple(map(int, next_obs)), step["action_mask"],
                         terminated=ended, elapsed_s=step["elapsed_s"])
        if ended or truncated:
            if truncated:
                raise RuntimeError("Refinement experiments require complete days")
            break
        observation, info = next_obs, step
    metrics = env.episode_metrics()
    if not isclose(-total, objective(metrics, env.rates), rel_tol=0., abs_tol=1e-8):
        raise RuntimeError("Common operational loss no longer reconciles")
    if mode == "potential":
        valid = isclose(discounted_signal / scale, discounted - initial, rel_tol=0., abs_tol=1e-8)
    else:
        valid = isclose(signal / scale, total, rel_tol=0., abs_tol=1e-8)
    if not valid:
        raise RuntimeError("Training reward failed to reconcile")
    if traces is not None:
        traces.extend(env.decision_trace())
    return {**metrics, "urgent_service_rate": metrics["urgent_served"] / max(1, metrics["urgent_total"]),
            "common_return": total, "training_return": signal, "episode_wall_s": perf_counter() - started,
            "probe_wall_s": env.probe_wall_s, "probe_candidates": env.probe_candidates,
            "probe_decisions": env.probe_decisions, "probe_urgent_wall_s": env.probe_urgent_wall_s,
            "probe_urgent_decisions": env.probe_urgent_decisions, "probe_exhaustions": env.probe_exhaustions,
            "inference_s": inference_time, "actionable": actionable, "unseen": unseen,
            "advanced_omissions": len(feedback.first_debit_s),
            "total_omission_advance_s": sum(sim.now_s - when for when in feedback.first_debit_s.values())}


def score_cases(cases, env, policy, identity, training_seed=None, traces=None):
    for index, case in enumerate(cases):
        key = {"policy": identity, "training_seed": training_seed, "case": case.key,
               "day": str(case.day), "regime": case.regime}
        decisions = [] if traces is not None else None
        row = rollout(env, policy, seed=case.seed, case_index=index, traces=decisions)
        if traces is not None:
            traces.extend({**key, **trace} for trace in decisions)
        yield {**key, **row}


def train(cases, config, encoding, mode, parameters, seed, episodes, *, scale=1., milestones=()):
    environment = make_env(cases, config, encoding)
    cards = ORIGINAL_CARDS if encoding == "original" else CARDINALITIES
    table = QTable(parameters, cardinalities=cards)
    for number in range(1, episodes + 1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, number]))
        case_index = int(rng.integers(len(cases)))
        epsilon = config["epsilon_start"] + (config["epsilon_end"] - config["epsilon_start"]) * (
            (number - 1) / max(1, episodes - 1)
        )
        policy = QTablePolicy(table, rng, epsilon)
        metrics = rollout(environment, policy, seed=seed, case_index=case_index, table=table,
                          mode=mode, weight=config["potential_weight"], scale=scale)
        yield number, table if number in milestones else None, {
            "episode": number, "epsilon": epsilon, "case": cases[case_index].key, "states": len(table.values),
            "visited_pairs": sum(np.count_nonzero(n) for n in table.counts.values()), **metrics,
        }


def diagnostic_feedback(train_cases, validation_cases, config, settings):
    env = make_env(validation_cases, config, "original")
    episodes = settings["diagnostic_episodes"]
    seed = settings["seeds"][0]
    variants = (("original", "original", 1., 1.), ("early", "early", 1., 1.),
                ("potential", "potential", 1., 1.), ("distance", "distance", 1., 1.),
                ("time_discount", "original", .995, 1.), ("reward_scaled", "original", 1., .01))
    inference = {}
    for label, mode, gamma, scale in variants:
        journal = []
        for _, snapshot, row in train(train_cases, config, "original", mode, Parameters(config["alpha"], gamma),
                                      seed, episodes, scale=scale, milestones=(episodes,)):
            journal.append(row)
        results = list(score_cases(validation_cases, env, QTablePolicy(snapshot), label, seed))
        inference[label] = (snapshot, [dict(policy=label, method=mode, gamma=gamma, scale=scale, **r)
                                       for r in journal], results)
    original = inference["original"][0]
    scaled = inference["reward_scaled"][0]
    same_support = original.counts.keys() == scaled.counts.keys() and all(
        np.array_equal(original.counts[key], scaled.counts[key]) for key in original.counts
    )
    agreement = sum(original.choose(key, (1, 1, 1, 1)) == scaled.choose(key, (1, 1, 1, 1))
                    for key in original.values) / len(original.values)
    return [row for _, train_rows, _ in inference.values() for row in train_rows], [
        row for _, _, val_rows in inference.values() for row in val_rows
    ], {"same_visited_counts": same_support, "greedy_agreement_on_original_states": agreement}


def causal_aliasing(cases, config):
    """Retrospective one-action forks used only for validation diagnostics."""
    env = make_env(cases, config, "original")
    rows, support = [], []
    for index, case in enumerate(cases):
        _, info = env.reset(seed=case.seed, options={"scenario_index": index})
        while True:
            obs = env.public_observation
            states = {"case": case.key, "decision_s": obs.now_s, "original": info["original_state"],
                      "refined": info["refined_state"]}
            support.append({**states, "eligible": obs.replan_available, "waiting": len(obs.waiting_urgent)})
            if obs.replan_available and obs.waiting_urgent:
                outcomes = {}
                for action in Action:
                    fork = deepcopy(env._sim)
                    planner = optimizer(config["solver_budget"]) if action == Action.LOCAL_REOPTIMIZATION else None
                    fork.act(action, planner=planner)
                    while fork.next_observation() is not None:
                        fork.act(Action.KEEP)
                    metrics = evaluate(fork)
                    outcomes[action] = (objective(metrics, env.rates), metrics["urgent_served"])
                keep = outcomes.pop(Action.KEEP)
                best_action = min(outcomes, key=lambda key: outcomes[key][0])
                best = outcomes[best_action]
                rows.append({**states, "gain_mad": keep[0] - best[0],
                             "global_gain_mad": keep[0] - outcomes[Action.GLOBAL][0],
                             "best_action": best_action.name, "urgent_gain": best[1] - keep[1],
                             **info["insertion_probe"]})
            action = reference_policy("multi_feature")(None, info, obs)
            _, _, terminated, _, info = env.step(action)
            if terminated:
                break
    result = pd.DataFrame(rows)
    statistics = []
    for encoding in ("original", "refined"):
        if result.empty:
            continue
        labels = result.gain_mad.gt(1e-6)
        group = result.assign(better=labels).groupby(encoding).better.agg(["size", "sum"])
        mixed = (group["sum"] > 0) & (group["sum"] < group["size"])
        errors = 0
        for key, part in result.groupby("case"):
            other = result[result.case != key]
            majorities = other.assign(better=other.gain_mad.gt(1e-6)).groupby(encoding).better.mean()
            predicted = part[encoding].map(majorities).fillna(0).gt(.5)
            errors += (predicted != part.gain_mad.gt(1e-6)).sum()
        statistics.append({"encoding": encoding, "decisions": len(result), "positive": labels.sum(),
                           "mixed_states": mixed.sum(), "mixed_decisions": group.loc[mixed, "size"].sum(),
                           "majority_error": np.minimum(group["sum"], group["size"] - group["sum"]).sum() / len(result),
                           "leave_case_out_error": errors / len(result)})
    return result, statistics, support


def paired(frame, draws, seed):
    grouped = frame.groupby(["policy", "day", "case"], as_index=False)[list(METRICS)].mean()
    tables = {name: sub.set_index(["day", "case"])[list(METRICS)] for name, sub in grouped.groupby("policy")}
    days = sorted(grouped.day.unique())
    rng = np.random.default_rng(seed)
    resamples = rng.integers(len(days), size=(draws, len(days)))
    spec = {f"{name}-{other}": tables[name] - tables[other] for name in VARIANTS
            for other in ("A", "multi_feature") if name != other}
    spec["observation_main"] = (tables["B"] - tables["A"] + tables["D"] - tables["C"]) / 2
    spec["reward_main"] = (tables["C"] - tables["A"] + tables["D"] - tables["B"]) / 2
    spec["interaction"] = tables["D"] - tables["B"] - tables["C"] + tables["A"]
    raw, rows = [], []
    for name, delta in spec.items():
        if delta.isna().any().any():
            raise RuntimeError("Missing comparison case in paired evaluation")
        raw.append(delta.reset_index().assign(contrast=name))
        matrix = delta.groupby(level="day").mean().loc[days].to_numpy()
        interval = np.quantile(matrix[resamples].mean(axis=1), [.025, .975], axis=0)
        for i, metric in enumerate(METRICS):
            rows.append({"contrast": name, "metric": metric, "difference": delta[metric].mean(),
                         "ci_low": interval[0, i], "ci_high": interval[1, i],
                         "pairs": len(delta), "day_blocks": len(days)})
    return pd.concat(raw, ignore_index=True), rows


def load_checkpoint(path, original):
    return QTable.load(path, cardinalities=ORIGINAL_CARDS if original else CARDINALITIES)


def run(profile, output):
    started = perf_counter()
    user_config = yaml.safe_load(CONFIG.read_text())
    config = {**yaml.safe_load(LEGACY.read_text()), **user_config}
    settings = config["profiles"][profile]
    for archive in ARCHIVES:
        if (ROOT / archive).resolve() in (output.resolve(), *output.resolve().parents):
            raise ValueError("A refinement experiment cannot overwrite phase-eight artifacts")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    development = fresh_banks(config, profile, ("train", "validation"))
    aliases, alias_stats, support = causal_aliasing(development["validation"], config)
    diag_train, diag_val, scale_stats = diagnostic_feedback(development["train"], development["validation"], config,
                                                           settings)
    write_table(output, "aliasing_decisions", aliases)
    write_table(output, "aliasing_summary", alias_stats)
    write_table(output, "validation_state_support", support)
    write_table(output, "feedback_validation", diag_val)
    write_table(output, "feedback_training", diag_train)

    models, history, validation, coverage = {}, [], [], []
    milestones = (settings["midpoint"], settings["episodes"])
    for variant, (encoding, mode) in VARIANTS.items():
        valid_env = make_env(development["validation"], config, encoding)
        for seed in settings["seeds"]:
            for number, snapshot, row in train(development["train"], config, encoding, mode,
                                                Parameters(config["alpha"], config["gamma"]), seed,
                                                settings["episodes"], milestones=milestones):
                history.append({"policy": variant, "training_seed": seed, **row})
                if snapshot is not None:
                    path = output / "checkpoints" / f"{variant}_{seed}_{number:04}.json"
                    snapshot.save(path, variant=variant, training_seed=seed, episodes=number,
                                  encoding=encoding, training_reward=mode)
                    restored = load_checkpoint(path, encoding == "original")
                    if any(not np.array_equal(v, restored.values[key]) for key, v in snapshot.values.items()):
                        raise RuntimeError("Checkpoint values did not reload exactly")
                    validation.extend(dict(episode=number, **r) for r in score_cases(
                        development["validation"], valid_env, QTablePolicy(restored), variant, seed
                    ))
                    if number == settings["episodes"]:
                        models[variant, seed] = restored
                        for key, counts in sorted(restored.counts.items()):
                            coverage.extend({"policy": variant, "training_seed": seed, "state": key,
                                             "action": DispatchAction(action).name, "visits": int(n),
                                             "q_mad": restored.values[key][action]}
                                            for action, n in enumerate(counts))
            print(f"{variant} seed={seed} episodes={number} states={len(restored.values)}", flush=True)

    # Generate the new held-out scenes only after the fixed training and diagnostics.
    final = fresh_banks(config, profile, ("test",))["test"]
    if {case.seed for part in development.values() for case in part} & {case.seed for case in final}:
        raise RuntimeError("Held-out random seed appeared in development")
    rows, traces = [], []
    baseline_env = make_env(final, config, "original")
    for name in POLICIES:
        rows.extend(score_cases(final, baseline_env, reference_policy(name), name, traces=traces))
    rows.extend(score_cases(final, baseline_env, random_policy(config["bootstrap_seed"]), "random_feasible", traces=traces))
    for (variant, seed), model in models.items():
        env = make_env(final, config, VARIANTS[variant][0])
        rows.extend(score_cases(final, env, QTablePolicy(model), variant, seed, traces=traces))
    frame = pd.DataFrame(rows)
    if frame.groupby("case").opportunities.nunique().ne(1).any() or frame.unserved_mandatory.any():
        raise RuntimeError("Paired decision clock or mandatory service changed")
    deltas, intervals = paired(frame, config["bootstrap_draws"], config["bootstrap_seed"])
    banks = {**development, "test": final}
    scenarios = []
    for metadata, case in zip(scenario_rows(banks), (case for part in banks.values() for case in part), strict=True):
        extra = [{**asdict(event), "kind": type(event).__name__} for event in case.scenario.events]
        scenarios.append({**metadata, "events": extra})
    (output / "scenario_manifest.json").write_text(json.dumps(scenarios, default=sorted, indent=2) + "\n")
    for name, table in (("train_log", history), ("validation_curves", validation), ("state_coverage", coverage),
                        ("test_cases", frame), ("test_decisions", traces), ("paired_differences", deltas),
                        ("paired_intervals", intervals)):
        write_table(output, name, table)
    (output / "manifest.json").write_text(json.dumps({
        "reference": "0346c7a", "profile": profile, "config_sha256": sha256(CONFIG.read_bytes()).hexdigest(),
        "legacy_config_sha256": sha256(LEGACY.read_bytes()).hexdigest(), "settings": settings,
        "partitions": {key: len(part) for key, part in banks.items()}, "scale_control": scale_stats,
        "cardinalities": {"original": ORIGINAL_CARDS, "refined": CARDINALITIES},
        "declared_states": {"original": prod(ORIGINAL_CARDS), "refined": prod(CARDINALITIES)},
        "canonical_states_upper_bound": {"original": 49, "refined": 73},
        "continuous_names": CONTINUOUS_NAMES, "wall_s": perf_counter() - started,
        "peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
    }, indent=2) + "\n")
    print(frame.groupby("policy")[list(METRICS)].mean().round(4).to_string())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("development", "standard", "research"), default="development")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = "" if args.profile == "standard" else f"_{args.profile}"
    run(args.profile, args.output or ROOT / f"q_learning_refinement{suffix}")


if __name__ == "__main__":
    main()
