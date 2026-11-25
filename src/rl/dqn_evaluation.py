"""Bounded Phase 9B experiments with an explicit validation review before final fitting."""

from argparse import ArgumentParser
from collections.abc import Iterator
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
import json
import random

PROCESS_STARTED = perf_counter()

from gymnasium.utils.env_checker import check_env as check_gym
import numpy as np
import pandas as pd
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env as check_sb3
import torch
import yaml

from src.rl.actions import DispatchAction
from src.rl.dqn import DQNEnv, DispatchDQN, Timing, create_model
from src.rl.dqn_diagnostics import controlled_forks, pilot_replay, replay_analysis
from src.rl.dqn_pilot import ARCHIVES, BASELINES, MODES, ROOT, TABULAR, load_config as pilot_config, make_env, replay_checks
from src.rl.experiment import random_policy, reference_policy, scenario_rows, write_table
from src.rl.q_learning import Parameters, QTable
from src.rl.refined_observations import CARDINALITIES as REFINED_CARDS
from src.rl.refined_rewards import Feedback
from src.rl.refinement_experiment import METRICS, objective
from src.rl.refinement_scenarios import fresh_banks
from src.rl.rewards import RewardRates
from src.rl.scenarios import Case
from src.simulation.state import Settings

CONFIG = Path("configs/dqn_evaluation.yaml")
OUTPUT = ROOT / "dqn_evaluation"
CASE_ARCHIVES = (*ARCHIVES, "dqn_pilot")
OUTCOME_METRICS = (*METRICS, "invalid_requests")


def save_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=sorted) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_config() -> dict:
    return {**pilot_config(), **yaml.safe_load(CONFIG.read_text())}


class TimeLimit(RuntimeError):
    pass


class Budget:
    """Persist charged interactions and cumulative experiment time between stages."""

    def __init__(self, output: Path, limits: dict, stage: str, *, started_at: float | None = None):
        self.path = output / "budget.json"
        self.limits = limits
        self.log = (json.loads(self.path.read_text()) if self.path.exists()
                    else {"wall_s": 0., "dqn": 0, "tabular": 0, "stages": {}})
        if stage in self.log["stages"]:
            raise ValueError("The experiment stage already used its budget in this output directory")
        self.stage = stage
        self.before = self.log["wall_s"]
        self.begin = perf_counter() if started_at is None else started_at
        self.sync("running")

    @property
    def elapsed(self) -> float:
        return self.before + perf_counter() - self.begin

    def check(self) -> None:
        if self.elapsed >= self.limits["wall_s"] - self.limits["checkpoint_reserve_s"]:
            raise TimeLimit("Cumulative experiment deadline reached; recoverable artifacts saved")

    def charge(self, kind: str, count: int = 1) -> None:
        if self.log[kind] + count > self.limits[kind]:
            raise ValueError(f"{kind} interaction budget exceeded")
        self.log[kind] += count

    def require(self, kind: str, count: int) -> None:
        if self.log[kind] + count > self.limits[kind]:
            raise ValueError(f"{kind} run would exceed authorized interactions")
        self.check()

    def sync(self, status: str = "running") -> None:
        self.log["wall_s"] = self.elapsed
        self.log["stages"][self.stage] = {"status": status, "wall_s": self.elapsed - self.before,
                                          "peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss}
        save_json(self.path, self.log)


def banks(config: dict, roles: tuple[str, ...], *, fleet: int = 3) -> dict[str, tuple[Case, ...]]:
    chosen = "standard" if fleet == 3 else "research"
    if fleet != 3 and tuple(roles) != ("test",):
        raise ValueError("Four-vehicle cases are reserved for final transfer")
    spec = config["partitions"]
    indices = [i for part in spec.values() for i in part["day_indices"]]
    if len(indices) != len(set(indices)) or len({part["namespace"] for part in spec.values()}) != len(spec):
        raise ValueError("Phase 9B date or seed roles overlap")
    phase_partitions = {name: spec[name] for name in ("train", "validation", "test")}
    output = {}
    ordinary = tuple(role for role in roles if role != "diagnostic")
    if ordinary:
        output.update(fresh_banks({**config, "partitions": phase_partitions}, chosen, ordinary))
    if "diagnostic" in roles:
        diagnostic_config = {**config, "partitions": {**phase_partitions, "test": spec["diagnostic"]}}
        diagnostic = fresh_banks(diagnostic_config, chosen, ("test",))["test"]
        output["diagnostic"] = tuple(replace(case, key=case.key.replace("test-", "diagnostic-", 1),
                                               split="diagnostic") for case in diagnostic)
    if fleet == 4:
        output["test"] = tuple(replace(case, key=f"{case.key}-fleet4") for case in output["test"])
    archived = {row["seed"] for name in CASE_ARCHIVES
                for row in json.loads((ROOT / name / "scenario_manifest.json").read_text())}
    current = [case.seed for cases in output.values() for case in cases]
    if archived.intersection(current) or len(current) != len(set(current)):
        raise ValueError("Scenario seed overlaps another bank")
    return output


def write_scenarios(path: Path, data: dict[str, tuple[Case, ...]]) -> None:
    cases = (case for part in data.values() for case in part)
    rows = [{**row, "events": [{**asdict(event), "kind": type(event).__name__} for event in case.scenario.events]}
            for row, case in zip(scenario_rows(data), cases, strict=True)]
    save_json(path, rows)


class LoggedEnv(DQNEnv):
    """Annotate training traces with public features independently of the learner."""

    def step(self, action):
        public = self._public_info
        result = super().step(action)
        self.timing.steps[-1].update(eligible=public["action_mask"][1],
                                     waiting=public["waiting_urgent"],
                                     original_state=str(public["original_state"]),
                                     refined_state=str(public["refined_state"]),
                                     costs_mad=result[-1]["costs_mad"])
        return result

    def _public(self):
        obs, info = super()._public()
        self._public_info = info
        return obs, info


def training_env(cases, config, mode, timing):
    settings = Settings(**config["dispatch"])
    return LoggedEnv([case.episode(settings) for case in cases], input_mode=mode, timing=timing,
                     probe_budget=config["probe_budget"], tight_s=config["tight_slack_s"],
                     rates=RewardRates(**config["reward"]), solver_budget=config["solver_budget"])


def score(cases, env, policy, budget: Budget, *, identity: dict, model=None, traces=None) -> Iterator[dict]:
    """Evaluate full days and expose Q distributions only for admissible actions."""
    for index, case in enumerate(cases):
        budget.check()
        obs, info = env.reset(seed=case.seed, options={"scenario_index": index})
        start = perf_counter()
        total = inference = 0.
        invalid = eligible = decisions = 0
        values = []
        key = {**identity, "case": case.key, "day": str(case.day), "regime": case.regime,
               "fleet": len(case.problem.vehicles)}
        while True:
            start_action = perf_counter()
            action = int(model.predict(obs, deterministic=True)[0] if model is not None
                         else policy(obs, info, env.public_observation))
            inference += perf_counter() - start_action
            if model is not None and info["action_mask"][1]:
                values.append(model.online_values(obs))
            invalid += not info["action_mask"][action]
            eligible += info["action_mask"][1]
            obs, reward, ended, truncated, info = env.step(action)
            total += reward
            decisions += 1
            if truncated:
                raise RuntimeError("Evaluation must end naturally")
            if ended:
                break
        metrics = env.episode_metrics()
        if (not np.isclose(-total, objective(metrics, env.rates), rtol=0, atol=1e-8)
                or invalid or metrics["unserved_mandatory"]):
            raise RuntimeError("Evaluation failed accounting, mask, or independent feasibility")
        if traces is not None:
            traces.extend({**key, **record} for record in env.decision_trace())
        q = {}
        if values:
            q_values = np.asarray(values)
            if not np.isfinite(q_values).all():
                raise RuntimeError("Non-finite admissible Q-values")
            for action_id in range(q_values.shape[1]):
                for label, val in zip((5, 50, 95), np.percentile(q_values[:, action_id], [5, 50, 95]), strict=True):
                    q[f"q_{action_id}_p{label}"] = float(val)
            q["q_intervene_minus_keep_median"] = float(np.median(q_values[:, 1:].max(axis=1) - q_values[:, 0]))
        yield {**key, **metrics, **q, "urgent_service_rate": metrics["urgent_served"] / metrics["urgent_total"],
               "invalid_requests": invalid, "eligible_decisions": eligible, "evaluation_decisions": decisions,
               "inference_s": inference, "episode_wall_s": perf_counter() - start,
               "probe_wall_s": env.probe_wall_s, "common_return": total}


def model_settings(config: dict, rate: float, steps: int) -> dict:
    return {**config["model"], "learning_rate": rate,
            "exploration_fraction": config["exploration_decay_steps"] / steps}


class TrainingCallback(BaseCallback):
    def __init__(self, budget, checkpoints, timing, milestone):
        super().__init__()
        self.budget, self.checkpoints, self.timing, self.milestone = budget, checkpoints, timing, milestone

    def _on_rollout_start(self):
        step = self.num_timesteps
        if step in self.checkpoints:
            self.milestone(step)
        self.budget.check()

    def _on_step(self):
        self.budget.charge("dqn")
        self.timing.steps[-1].update(epsilon=(1. if self.num_timesteps <= self.model.learning_starts
                                               else self.model.exploration_rate))
        if self.num_timesteps % 128 == 0:
            self.budget.sync()
        return True


def train_dqn(train, validation, config, budget, directory, *, mode, seed, rate, steps, name):
    budget.require("dqn", steps)
    directory.mkdir()
    timing = Timing()
    env = training_env(train, config, mode, timing)
    evaluator = make_env(validation, config, mode=mode)
    model = create_model(env, model_settings(config, rate, steps), seed)
    model.update_log = []
    identity = {"policy": name, "mode": mode, "training_seed": seed, "learning_rate": rate}
    rows, replay, checkpoint_profiles = [], [], []
    eval_s = 0.
    begin = perf_counter()

    def milestone(step, *, validate=True):
        nonlocal eval_s
        point_start = perf_counter()
        if step:
            path = directory / f"step_{step:04}.zip"
            buffer_path = directory / f"replay_{step:04}.pkl"
            model.save(path)
            model.save_replay_buffer(buffer_path)
            # SB3 load seeds global generators during setup. Preserve the collector's stream.
            numpy_state, python_state = np.random.get_state(), random.getstate()
            try:
                with torch.random.fork_rng(devices=[]):
                    restored = DispatchDQN.load(path, device="cpu")
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
            restored.load_replay_buffer(buffer_path)
            if replay_checks(restored) != replay_checks(model):
                raise RuntimeError("CPU replay reload changed composition")
            obs, _ = evaluator.reset(seed=validation[0].seed, options={"scenario_index": 0})
            if (any(not torch.equal(a, b) for a, b in
                    zip(restored.policy.parameters(), model.policy.parameters(), strict=True))
                    or not np.array_equal(restored.online_values(obs), model.online_values(obs))
                    or int(restored.predict(obs, deterministic=True)[0]) != int(model.predict(obs, deterministic=True)[0])):
                raise RuntimeError("CPU checkpoint reload changed network weights or predictions")
            replay.extend({**identity, "step": step, **row} for row in replay_analysis(restored))
            model_for_eval = restored
        else:
            model_for_eval = model
        if validate:
            rows.extend(score(validation, evaluator, None, budget, model=model_for_eval,
                              identity={**identity, "step": step}))
        gap = torch.cat([(online.detach() - target.detach()).flatten() for online, target in
                         zip(model.q_net.parameters(), model.q_net_target.parameters(), strict=True)])
        checkpoint_profiles.append({**identity, "step": step, "gradient_updates": model._n_updates,
                                    "target_syncs": step // model.target_update_interval,
                                    "online_target_parameter_l2": float(torch.linalg.vector_norm(gap)),
                                    "epsilon_for_next_step": model.exploration_rate})
        eval_s += perf_counter() - point_start
        budget.sync()
        if rows and validate:
            current = pd.DataFrame(rows)
            part = current[current.step.eq(step)]
            print(f"{name} seed={seed} step={step} loss={part.objective_mad.mean():.2f} "
                  f"attempts={part.replan_attempts.mean():.2f} cumulative={budget.elapsed:.1f}s", flush=True)

    callback = TrainingCallback(budget, range(config["checkpoint_interval"], steps,
                                              config["checkpoint_interval"]), timing, milestone)
    try:
        milestone(0)
        model.learn(steps, callback=callback)
        milestone(model.num_timesteps)
    except TimeLimit:
        if model.num_timesteps and not (directory / f"step_{model.num_timesteps:04}.zip").exists():
            milestone(model.num_timesteps, validate=False)
        raise
    finally:
        train_wall = max(0., perf_counter() - begin - eval_s)
        write_run(directory, rows, replay, checkpoint_profiles, timing, model.update_log,
                  {**identity, "steps": len(timing.steps), "gradient_updates": model._n_updates,
                   "train_wall_s": train_wall, "validation_and_checkpoint_wall_s": eval_s})


def write_run(directory, validation, replay, checkpoints, timing, updates, profile):
    write_table(directory, "validation", validation)
    if replay:
        write_table(directory, "replay_analysis", replay)
    write_table(directory, "checkpoint_profile", checkpoints)
    write_table(directory, "episodes", timing.episodes)
    if updates:
        write_table(directory, "gradient_updates", updates)
    pd.DataFrame(timing.steps).to_csv(directory / "training_steps.csv.gz", index=False,
                                     compression={"method": "gzip", "mtime": 0})
    save_json(directory / "profile.json", {**profile, "episodes": len(timing.episodes),
              "solver_calls": sum(row["solver_calls"] for row in timing.steps),
              "environment_wall_s": sum(row["step_wall_s"] for row in timing.steps),
              "planning_wall_s": sum(row["planning_wall_s"] for row in timing.steps),
              "reset_wall_s": sum(row["reset_wall_s"] for row in timing.resets),
              "update_wall_s": sum(row["update_wall_s"] for row in updates),
              "invalid_requests": sum(row["invalid_request"] for row in timing.steps)})


def screen(config, output, budget):
    (output / "screen").mkdir()
    data = banks(config, ("train", "validation", "diagnostic"))
    write_scenarios(output / "development_scenarios.json", data)
    for mode in MODES:
        check_gym(make_env(data["train"], config, mode=mode), skip_render_check=True)
        check_sb3(make_env(data["train"], config, mode=mode), warn=True)
    write_table(output, "pilot_replay_analysis", pilot_replay(ROOT))
    for rate in config["learning_rates"]:
        name = f"screen_lr_{rate:g}"
        train_dqn(data["train"], data["validation"], config, budget, output / "screen" / name,
                  mode="continuous", seed=config["screen_seed"], rate=rate, steps=config["screen_steps"], name=name)
    models = [(f"pilot_continuous_{seed}", "continuous", DispatchDQN.load(
        ROOT / f"dqn_pilot/checkpoints/dqn_continuous_{seed}_0768.zip", device="cpu")) for seed in (1901, 1907)]
    models.extend((run.name, "continuous", DispatchDQN.load(
        run / f"step_{config['screen_steps']:04}.zip", device="cpu")) for run in sorted((output / "screen").iterdir()))
    diagnostic, support, predictions = controlled_forks(data["diagnostic"], config, models, budget.check)
    write_table(output, "diagnostic_forks", diagnostic)
    write_table(output, "diagnostic_support", support)
    write_table(output, "diagnostic_q", predictions)
    rows = pd.concat([pd.read_csv(run / "validation.csv") for run in sorted((output / "screen").iterdir())])
    summary = rows.groupby(["learning_rate", "step"], as_index=False)[
        ["objective_mad", "otif", "urgent_service_rate", "replan_attempts", "solver_calls",
         "safety_fallbacks", "q_intervene_minus_keep_median"]].mean()
    write_table(output, "screen_summary", summary)
    print(summary.to_string(index=False), flush=True)


def train_tabular(train, validation, config, budget, directory, *, name, seed):
    steps = config["final_steps"]
    budget.require("tabular", steps)
    directory.mkdir()
    field, mode = TABULAR[name]
    cards = REFINED_CARDS if field == "refined_state" else None
    parameters = Parameters(config["alpha"], config["model"]["gamma"], time_aware=False)
    table = QTable(parameters, **({"cardinalities": cards} if cards else {}))
    timing = Timing()
    env = training_env(train, config, "continuous", timing)
    evaluator = make_env(validation, config)
    identity = {"policy": name, "training_seed": seed}
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    feedback = Feedback(mode, config["potential_weight"])
    raw_return = learned_return = 0.
    validation_rows, checkpoints = [], []
    eval_s = 0.
    begin = perf_counter()

    def milestone(step, *, validate=True):
        nonlocal eval_s
        start = perf_counter()
        restored = table
        if step:
            path = directory / f"step_{step:04}.json"
            table.save(path, training_seed=seed, interactions=step, reward_mode=mode, observation=field)
            restored = load_table(path, field)
            if any(not np.array_equal(restored.values[state], values) or
                   not np.array_equal(restored.counts[state], table.counts[state])
                   for state, values in table.values.items()):
                raise RuntimeError("Reloaded table changed values or visits")
        if validate:
            policy = lambda obs, info, public: restored.choose(info[field], info["action_mask"])
            validation_rows.extend(score(validation, evaluator, policy, budget, identity={**identity, "step": step}))
        checkpoints.append({**identity, "step": step, "states": len(table.values),
                            "visited_pairs": sum(np.count_nonzero(counts) for counts in table.counts.values())})
        eval_s += perf_counter() - start
        budget.sync()
        if validate and step:
            loss = pd.DataFrame(validation_rows).query("step == @step").objective_mad.mean()
            print(f"{name} seed={seed} step={step} loss={loss:.2f} cumulative={budget.elapsed:.1f}s", flush=True)

    try:
        milestone(0)
        for i in range(steps):
            budget.check()
            fraction = min(1., i / config["exploration_decay_steps"])
            model = config["model"]
            epsilon = model["exploration_initial_eps"] + fraction * (
                model["exploration_final_eps"] - model["exploration_initial_eps"])
            state = info[field]
            action = table.choose(state, info["action_mask"], rng, epsilon)
            _, reward, ended, truncated, info = env.step(action)
            budget.charge("tabular")
            if truncated:
                raise RuntimeError("Tabular training unexpectedly truncated")
            sim = env._sim
            signal = feedback.transform(reward, sim, env.public_observation,
                                        sim.dispatch_forecast() if mode == "early" and not ended else None,
                                        parameters.gamma, ended)
            timing.steps[-1].update(epsilon=epsilon, learning_reward_mad=signal)
            raw_return += reward
            learned_return += signal
            table.update(state, action, signal, info.get(field, (0, 0, 0, 0)), info["action_mask"],
                         terminated=ended, elapsed_s=info["elapsed_s"])
            if ended:
                if not np.isclose(raw_return, learned_return, rtol=0., atol=1e-8):
                    raise RuntimeError("Training transformed the common episode objective")
                if i + 1 < steps:
                    _, info = env.reset()
                    raw_return = learned_return = 0.
                    feedback = Feedback(mode, config["potential_weight"])
            if (i + 1) % config["checkpoint_interval"] == 0:
                milestone(i + 1)
            elif (i + 1) % 128 == 0:
                budget.sync()
    except TimeLimit:
        step = len(timing.steps)
        if step and not (directory / f"step_{step:04}.json").exists():
            milestone(step, validate=False)
        raise
    finally:
        write_run(directory, validation_rows, [], checkpoints, timing, [],
                  {**identity, "steps": len(timing.steps), "gradient_updates": 0,
                   "table_updates": len(timing.steps), "train_wall_s": max(0., perf_counter() - begin - eval_s),
                   "validation_and_checkpoint_wall_s": eval_s})


def load_table(path, field):
    return QTable.load(path, **({"cardinalities": REFINED_CARDS} if field == "refined_state" else {}))


def select_checkpoint(path: Path) -> tuple[int, float]:
    frame = pd.read_csv(path / "validation.csv")
    trained = frame[frame.step.gt(0)]
    means = trained.groupby("step").objective_mad.mean().sort_index()
    return int(means.idxmin()), float(means.min())


def intermediate_review(output, config):
    summary_path = output / "screen_summary.csv"
    screened = pd.read_csv(summary_path)
    endpoint = screened[screened.step.eq(config["screen_steps"])].sort_values(
        ["objective_mad", "learning_rate"])
    if len(endpoint) != len(config["learning_rates"]):
        raise RuntimeError("The screen requires a finished validation endpoint for each rate")
    selected = endpoint.iloc[0]
    forks = pd.read_csv(output / "diagnostic_forks.csv")
    best = forks.groupby(["case", "decision_s"]).gain_mad.max()
    diagnostic_gains = int(best.gt(1e-6).sum())
    both_keep = bool(endpoint.replan_attempts.eq(0).all())
    evidence = {
        "validation_endpoints": endpoint.to_dict("records"),
        "sampled_diagnostic_decisions": len(best),
        "positive_diagnostic_decisions": diagnostic_gains,
        "both_rates_systematic_keep_on_validation": both_keep,
        "changes_after_screen": [],
    }
    review = {
        "decision": "investigate" if both_keep else "extend",
        "learning_rate": float(selected.learning_rate),
        "screen_summary_sha256": digest(summary_path),
        "evidence": evidence,
    }
    if both_keep:
        review["reason"] = "Both endpoint policies kept every plan; inspect replay and public-state forks before extending."
    else:
        review["reason"] = ("Select the endpoint with the lowest validation operational loss. "
                            "At least one policy uses interventions with real operational consequences. "
                            "Compare both fixed representations with three new training seeds.")
    save_json(output / "review.json", review)
    print(json.dumps(review, indent=2), flush=True)


def verify_review(output, config):
    review_path = output / "review.json"
    review = json.loads(review_path.read_text())
    if review["screen_summary_sha256"] != digest(output / "screen_summary.csv"):
        raise ValueError("The screen metrics differ from the documented review")
    if review["learning_rate"] not in config["learning_rates"] or review["decision"] != "extend":
        raise ValueError("The reviewed learning rate and decision must authorize the next 9B experiments")
    if not review["evidence"]:
        raise ValueError("The intermediate review must record its validation evidence")
    return review


def fit(config, output, budget, review):
    data = banks(config, ("train", "validation"))
    directory = output / "fit"
    directory.mkdir()
    for mode in MODES:
        for seed in config["seeds"]:
            name = f"dqn_{mode}"
            train_dqn(data["train"], data["validation"], config, budget, directory / f"{name}_{seed}",
                      mode=mode, seed=seed, rate=review["learning_rate"], steps=config["final_steps"], name=name)
    for name in TABULAR:
        for seed in config["seeds"]:
            train_tabular(data["train"], data["validation"], config, budget, directory / f"{name}_{seed}",
                          name=name, seed=seed)


def historical_models():
    rows = []
    for seed in (5107, 7613, 9817):
        rows.append({"policy": "historical_phase8", "training_seed": seed, "field": "original_state",
                     "path": ROOT / f"q_learning/checkpoints/seed_{seed}_episode_0180.json"})
    for label, field in (("A", "original_state"), ("B", "refined_state"),
                         ("C", "original_state"), ("D", "refined_state")):
        for seed in (8817, 9137, 9473):
            rows.append({"policy": f"historical_81_{label}", "training_seed": seed, "field": field,
                         "path": ROOT / f"q_learning_refinement/checkpoints/{label}_{seed}_0072.json"})
    for mode in MODES:
        for seed in (1901, 1907):
            rows.append({"policy": f"historical_9A_{mode}", "training_seed": seed, "mode": mode,
                         "path": ROOT / f"dqn_pilot/checkpoints/dqn_{mode}_{seed}_0768.zip"})
    return rows


def paths_and_protocol(config, output, test_per_day):
    rows = []
    for policy in (*[f"dqn_{mode}" for mode in MODES], *TABULAR):
        for seed in config["seeds"]:
            directory = output / "fit" / f"{policy}_{seed}"
            step, validation_loss = select_checkpoint(directory)
            row = {"policy": policy, "training_seed": seed, "step": step,
                   "validation_loss_mad": validation_loss}
            if policy.startswith("dqn_"):
                row.update(mode=policy.removeprefix("dqn_"), path=directory / f"step_{step:04}.zip",
                           replay=directory / f"replay_{step:04}.pkl")
            else:
                row.update(field=TABULAR[policy][0], path=directory / f"step_{step:04}.json")
            rows.append(row)
    means = pd.DataFrame(row for row in rows if row["policy"].startswith("dqn_"))
    primary = means.groupby("policy").validation_loss_mad.mean().sort_index().idxmin()
    rows.extend(historical_models())
    for row in rows:
        for name in ("path", "replay"):
            if name in row:
                artifact = row[name]
                row[f"{name}_sha256"] = digest(artifact)
                row[name] = str(artifact.relative_to(output) if artifact.is_relative_to(output) else artifact)
    return {"stage": "9B", "reference": "a6968e5", "device": "cpu", "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__, "primary_dqn": primary,
            "config_sha256": digest(output / "config.yaml"), "review_sha256": digest(output / "review.json"),
            "bootstrap_seed": config["bootstrap_seed"], "bootstrap_draws": config["bootstrap_draws"],
            "bootstrap": "Paired date blocks, cases within date and fleet, and training seeds",
            "test_partition": config["partitions"]["test"], "test_fleets": [3, 4],
            "test_per_day_per_fleet": test_per_day,
            "selection": "Mean validation operational loss; ties use earlier trained step or alphabetic mode",
            "baseline_names": list(BASELINES), "models": rows}


def comparison(frame: pd.DataFrame, config: dict, primary: str) -> tuple[list[dict], list[dict]]:
    """Resample shared dates and paired cases, then seed IDs for each policy group."""
    metrics = list(OUTCOME_METRICS)
    deltas, intervals = [], []
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = config["bootstrap_draws"]
    for fleet in (0, *sorted(frame.fleet.unique())):
        data = frame if fleet == 0 else frame[frame.fleet.eq(fleet)]
        days = sorted(data.day.unique())
        date_draw = rng.integers(len(days), size=(draws, len(days)))
        new_seeds = sorted(config["seeds"])
        shared_draw = rng.integers(len(new_seeds), size=(draws, len(new_seeds)))
        policies = {name: part.copy() for name, part in data.groupby("policy")}
        raw = {name: part.groupby(["day", "case"])[metrics].mean() for name, part in policies.items()}

        # Conditional case draws preserve each date's fleet composition and pair all policies.
        case_draws = []
        source = data[["day", "fleet", "case"]].drop_duplicates()
        for i, day in enumerate(days):
            locations = date_draw == i
            daily = source[source.day.eq(day)]
            for _, group in daily.groupby("fleet"):
                cases = sorted(group.case)
                draws_in_fleet = rng.integers(len(cases), size=(int(locations.sum()), len(cases)))
                case_draws.append((locations, cases, draws_in_fleet, len(cases) / len(daily)))

        def boot(part):
            keyed = part.assign(training_seed=part.training_seed.fillna(-1))
            samples = [rows.set_index("case")[metrics] for _, rows in keyed.groupby("training_seed")]
            per_block = np.zeros((*date_draw.shape, len(samples), len(metrics)))
            for locations, cases, sampled_cases, weight in case_draws:
                values = np.stack([rows.loc[cases].to_numpy() for rows in samples], axis=1)
                per_block[locations] += weight * values[sampled_cases].mean(axis=1)
            resampled = per_block.mean(axis=1)
            if len(samples) == 1:
                return resampled[:, 0, :]
            matched = sorted(keyed.training_seed.unique()) == new_seeds
            indices = shared_draw if matched else rng.integers(len(samples), size=(draws, len(samples)))
            return resampled[np.arange(draws)[:, None], indices].mean(axis=1)

        main_boot = boot(policies[primary])
        for name, other in raw.items():
            if name == primary:
                continue
            if not raw[primary].index.equals(other.index):
                raise RuntimeError("Final policy cases are not paired")
            gap = raw[primary] - other
            deltas.extend(gap.assign(fleet=fleet, reference=primary, comparison=name).reset_index().to_dict("records"))
            daily_gap = gap.groupby(level="day").mean().loc[days].to_numpy()
            date_only = np.quantile(daily_gap[date_draw].mean(axis=1), [.025, .975], axis=0)
            both = np.quantile(main_boot - boot(policies[name]), [.025, .975], axis=0)
            for i, metric in enumerate(metrics):
                intervals.append({"fleet": fleet, "reference": primary, "comparison": name, "metric": metric,
                                  "difference": gap[metric].mean(), "ci_low": both[0, i], "ci_high": both[1, i],
                                  "ci_date_only_low": date_only[0, i], "ci_date_only_high": date_only[1, i],
                                  "cases": len(gap), "date_blocks": len(days)})
    return deltas, intervals


def final_evaluation(config, output, budget, test_per_day):
    profiles = config["profiles"]
    if not 1 <= test_per_day <= min(profile["per_day"][2] for profile in profiles.values()):
        raise ValueError("Final test draw count exceeds its configured bounds")
    protocol_path = output / "frozen_protocol.json"
    protocol = paths_and_protocol(config, output, test_per_day)
    save_json(protocol_path, protocol)
    # Neither final draws nor outcomes exist before checkpoint and protocol freeze.
    test_profiles = {name: {**profile, "per_day": [*profile["per_day"][:2], test_per_day]}
                     for name, profile in profiles.items()}
    test_config = {**config, "profiles": test_profiles}
    data = banks(test_config, ("test",))["test"] + banks(test_config, ("test",), fleet=4)["test"]
    previous = json.loads((output / "development_scenarios.json").read_text())
    if len({case.seed for case in data}) != len(data) or {c.seed for c in data} & {r["seed"] for r in previous}:
        raise ValueError("A final scenario reuses a seed")
    write_scenarios(output / "final_scenarios.json", {"test": data})
    env = make_env(data, config)
    blocks, traces, reload_checks = [], [], []
    progress = output / "final_progress"
    progress.mkdir()

    def record(identity, *, policy=None, model=None, mode="continuous"):
        candidates = env if mode == "continuous" else make_env(data, config, mode=mode)
        offset = len(traces)
        rows = list(score(data, candidates, policy, budget, identity=identity, model=model, traces=traces))
        blocks.extend(rows)
        stem = f"{identity['policy']}_{identity.get('training_seed', 'rule')}"
        write_table(progress, stem, rows)
        write_table(progress, f"{stem}_decisions", traces[offset:])
        budget.sync()
        print(f"final {stem} cases={len(rows)} cumulative={budget.elapsed:.1f}s", flush=True)
        return rows

    def verify_reload(identity, observed, *, policy=None, model=None, mode="continuous"):
        witnesses = (data[0], next(case for case in data if len(case.problem.vehicles) == 4))
        repeated = score(witnesses, make_env(witnesses, config, mode=mode), policy, budget,
                         identity=identity, model=model)
        first = {row["case"]: row for row in observed}
        for row in repeated:
            fields = (*OUTCOME_METRICS, "eligible_decisions", "evaluation_decisions", "unserved_mandatory",
                      *(key for key in row if key.startswith(("requested_", "executed_"))))
            if any(row[key] != first[row["case"]][key] for key in fields):
                raise RuntimeError("Repeated CPU evaluation changed operational results")
            reload_checks.append({**identity, "case": row["case"], "fleet": row["fleet"],
                                  "operational_metrics_equal": True})

    for name in protocol["baseline_names"]:
        policy = random_policy(config["bootstrap_seed"]) if name == "random_feasible" else reference_policy(name)
        record({"policy": name}, policy=policy)
    for item in protocol["models"]:
        path = Path(item["path"])
        if path.parts[0] == "fit":
            path = output / path
        if digest(path) != item["path_sha256"]:
            raise RuntimeError("Frozen checkpoint changed")
        identity = {field: item[field] for field in ("policy", "training_seed", "step") if field in item}
        if "mode" in item:
            model = DispatchDQN.load(path, device="cpu")
            rows = record(identity, model=model, mode=item["mode"])
            if "replay" in item:
                replay_path = output / item["replay"]
                if digest(replay_path) != item["replay_sha256"]:
                    raise RuntimeError("Frozen replay changed")
                duplicate = DispatchDQN.load(path, device="cpu")
                duplicate.load_replay_buffer(replay_path)
                replay_checks(duplicate)
                verify_reload(identity, rows, model=duplicate, mode=item["mode"])
        else:
            table = load_table(path, item["field"])
            rows = record(identity, policy=lambda obs, info, public: table.choose(info[item["field"]], info["action_mask"]))
            if "step" in item:
                duplicate = load_table(path, item["field"])
                verify_reload(identity, rows,
                              policy=lambda obs, info, public: duplicate.choose(info[item["field"]], info["action_mask"]))
    frame = pd.DataFrame(blocks)
    if (frame.groupby("case").opportunities.nunique().ne(1).any()
            or frame.groupby("case").size().ne(len(BASELINES) + len(protocol["models"])).any()):
        raise RuntimeError("Paired evaluation opportunities or policy counts differ")
    deltas, intervals = comparison(frame, config, protocol["primary_dqn"])
    write_table(output, "final_cases", frame)
    write_table(output, "cpu_reloads", reload_checks)
    pd.DataFrame(traces).to_csv(output / "final_decisions.csv.gz", index=False,
                                compression={"method": "gzip", "mtime": 0})
    write_table(output, "paired_differences", deltas)
    write_table(output, "paired_intervals", intervals)
    write_table(output, "final_summary", frame.groupby(["policy", "fleet"], as_index=False)[
        list(OUTCOME_METRICS)].mean())
    actions = []
    for (policy, fleet), group in frame.groupby(["policy", "fleet"]):
        decisions = int(group.evaluation_decisions.sum())
        eligible = int(group.eligible_decisions.sum())
        for action in DispatchAction:
            requested = int(group[f"requested_{action.name}"].sum())
            executed = int(group[f"executed_{action.name}"].sum())
            forced_keeps = decisions - eligible if action == DispatchAction.KEEP_PLAN else 0
            actions.append({"policy": policy, "fleet": fleet, "action": action.name,
                            "requested": requested, "executed": executed, "decisions": decisions,
                            "eligible_decisions": eligible, "requested_when_eligible": requested - forced_keeps,
                            "executed_when_eligible": executed - forced_keeps})
    write_table(output, "final_action_distribution", actions)
    # Consolidated outputs retain the same records; remove only this run's streaming copies.
    for path in progress.iterdir():
        path.unlink()
    progress.rmdir()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "review", "fit", "final"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--test-per-day", type=int, default=None,
                        help="Final-stage draws per date and fleet, set before the test is generated")
    args = parser.parse_args()
    if args.test_per_day is not None and args.stage != "final":
        parser.error("--test-per-day is only available for the final stage")
    output = args.output.resolve()
    allowed = OUTPUT.resolve()
    if ROOT.resolve() in output.parents and allowed != output:
        parser.error("Use the Phase 9B output directory or a directory outside historical results")
    if output == ROOT.resolve() or output in ROOT.resolve().parents:
        parser.error("Output overlaps project or historical results")
    config = load_config()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "config.yaml"
    resolved = yaml.safe_dump(config, sort_keys=False)
    if path.exists() and path.read_text() != resolved:
        parser.error("The resolved protocol changed since experiment initialization")
    path.write_text(resolved)
    torch.set_num_threads(config["threads"])
    if args.stage != "screen":
        previous = json.loads((output / "budget.json").read_text())["stages"]
        needed = {"review": "screen", "fit": "review", "final": "fit"}[args.stage]
        if previous.get(needed, {}).get("status") != "complete":
            parser.error(f"Complete the {needed} stage first")
        if args.stage != "review":
            review = verify_review(output, config)
    budget = Budget(output, config["limits"], args.stage, started_at=PROCESS_STARTED)
    try:
        if args.stage == "screen":
            screen(config, output, budget)
        elif args.stage == "review":
            intermediate_review(output, config)
        elif args.stage == "fit":
            fit(config, output, budget, review)
        else:
            test_per_day = args.test_per_day if args.test_per_day is not None else config["profiles"]["standard"]["per_day"][2]
            final_evaluation(config, output, budget, test_per_day)
    except TimeLimit as error:
        budget.sync("deadline")
        parser.exit(2, f"{error}\n")
    except Exception:
        budget.sync("failed")
        raise
    else:
        budget.sync("complete")


if __name__ == "__main__":
    main()
