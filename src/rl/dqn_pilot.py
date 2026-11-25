"""Run the fixed CPU pilot with independent development-scene partitions."""

from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
import json

import numpy as np
import pandas as pd
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env as check_sb3
from gymnasium.utils.env_checker import check_env as check_gym
import torch
import yaml

from src.rl.dqn import DispatchDQN, DQNEnv, Timing, create_model
from src.rl.experiment import random_policy, reference_policy, scenario_rows, write_table
from src.rl.q_learning import Parameters, QTable
from src.rl.refined_observations import CARDINALITIES as REFINED_CARDS, CONTINUOUS_NAMES
from src.rl.refined_rewards import Feedback
from src.rl.refinement_experiment import LEGACY, CONFIG as REFINEMENT, METRICS, objective
from src.rl.refinement_scenarios import fresh_banks
from src.rl.rewards import RewardRates
from src.simulation.policies import POLICIES
from src.simulation.state import Settings

CONFIG = Path("configs/dqn_pilot.yaml")
ROOT = Path("results/experiments")
ARCHIVES = ("q_learning", "q_learning_research_stress", "q_learning_refinement")
BASELINES = tuple(POLICIES) + ("random_feasible",)
TABULAR = {"tabular_A": ("original_state", "original"),
           "tabular_B": ("refined_state", "original"),
           "tabular_D": ("refined_state", "early")}
MODES = ("continuous", "compact")


def load_config():
    return {**yaml.safe_load(LEGACY.read_text()), **yaml.safe_load(REFINEMENT.read_text()),
            **yaml.safe_load(CONFIG.read_text())}


def pilot_banks(config, splits):
    banks = fresh_banks(config, "development", splits)
    archived = {row["seed"] for name in ARCHIVES
                for row in json.loads((ROOT / name / "scenario_manifest.json").read_text())}
    if archived & {case.seed for cases in banks.values() for case in cases}:
        raise RuntimeError("Pilot draw overlaps an earlier benchmark")
    return banks


def make_env(cases, config, *, mode="continuous", timing=None):
    dispatch = Settings(**config["dispatch"])
    return DQNEnv([case.episode(dispatch) for case in cases], input_mode=mode, timing=timing,
                  probe_budget=config["probe_budget"], tight_s=config["tight_slack_s"],
                  rates=RewardRates(**config["reward"]), solver_budget=config["solver_budget"])


def replay_checks(model):
    buffer = model.replay_buffer
    n = buffer.size()
    states = buffer.observations[:n, 0]
    following = buffer.next_observations[:n, 0]
    actions = buffer.actions[:n, 0, 0]
    if not n or not np.isfinite(states).all() or not np.isfinite(following).all() or not np.isfinite(
        buffer.rewards[:n]
    ).all() or np.any((states[:, 0] == 0) & (actions != 0)):
        raise RuntimeError("Replay contains an invalid transition")
    return {"buffer_size": n, "real_terminations": int(np.count_nonzero(
        buffer.dones[:n] * (1 - buffer.timeouts[:n]))),
        "timeouts": int(np.count_nonzero(buffer.timeouts[:n])),
        "action_counts": np.bincount(actions, minlength=4).tolist()}


def neural_row(model, observation, info, current):
    return int(model.predict(observation, deterministic=True)[0])


def evaluate_cases(cases, env, name, policy, *, seed=None, step=None, model=None, traces=None):
    for index, case in enumerate(cases):
        observation, info = env.reset(seed=case.seed, options={"scenario_index": index})
        start = perf_counter()
        reward_sum = inference_s = q_abs = 0.
        decisions = invalid = 0
        while True:
            before = perf_counter()
            action = int(policy(observation, info, env.public_observation))
            inference_s += perf_counter() - before
            if model is not None:
                values = model.online_values(observation)
                if not np.isfinite(values).all():
                    raise RuntimeError("Non-finite admissible Q-value")
                q_abs = max(q_abs, float(np.max(np.abs(values))))
            invalid += not info["action_mask"][action]
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            decisions += 1
            if terminated or truncated:
                if truncated:
                    raise RuntimeError("An evaluation day was truncated")
                break
        metrics = env.episode_metrics()
        if not np.isclose(-reward_sum, objective(metrics, env.rates), rtol=0, atol=1e-8):
            raise RuntimeError("Evaluation failed to reconcile with the common objective")
        identity = {"policy": name, "training_seed": seed, "step": step,
                    "case": case.key, "day": str(case.day), "regime": case.regime}
        if traces is not None:
            traces.extend({**identity, **row} for row in env.decision_trace())
        yield {**identity, **metrics, "urgent_service_rate": metrics["urgent_served"] / metrics["urgent_total"],
               "common_return": reward_sum, "invalid_requests": invalid, "evaluation_decisions": decisions,
               "inference_s": inference_s, "episode_wall_s": perf_counter() - start,
               "probe_wall_s": env.probe_wall_s, "max_absolute_q_mad": q_abs if model is not None else None}


class Checkpoint(BaseCallback):
    def __init__(self, directory, stem, limit, interval, evaluate):
        super().__init__()
        self.directory, self.stem = directory, stem
        self.limit, self.interval, self.evaluate = limit, interval, evaluate

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step < self.limit and step % self.interval == 0:
            self.model.save(self.directory / f"{self.stem}_{step:04}.zip")
            self.evaluate(self.model, step)
        return True


def train_dqn(cases, validation, config, mode, seed, destination):
    name = f"dqn_{mode}"
    timing = Timing()
    env = make_env(cases, config, mode=mode, timing=timing)
    evaluator = make_env(validation, config, mode=mode)
    model = create_model(env, config["model"], seed)
    model.update_log = []
    initial = torch.cat([p.detach().flatten() for p in model.q_net.parameters()]).clone()
    snapshots = []
    evaluation_wall_s = 0.

    def validate(current, step):
        nonlocal evaluation_wall_s
        started = perf_counter()
        snapshots.extend(evaluate_cases(validation, evaluator, name,
                                        lambda obs, info, public: neural_row(current, obs, info, public),
                                        seed=seed, step=step, model=current))
        evaluation_wall_s += perf_counter() - started

    stem = f"{name}_{seed}"
    validate(model, 0)
    prior_eval = evaluation_wall_s
    begin = perf_counter()
    callback = Checkpoint(destination, stem, config["steps"], config["validation_interval"], validate)
    model.learn(total_timesteps=config["steps"], callback=callback)
    training_s = perf_counter() - begin - (evaluation_wall_s - prior_eval)
    validate(model, model.num_timesteps)
    final = torch.cat([p.detach().flatten() for p in model.q_net.parameters()])
    parameter_change = float(torch.linalg.vector_norm(final - initial))
    if not np.isfinite(parameter_change) or not parameter_change or not model.update_log:
        raise RuntimeError("Neural training did not update the model")
    path = destination / f"{stem}_{config['steps']:04}.zip"
    replay_path = destination / f"{stem}_replay.pkl"
    model.save(path)
    model.save_replay_buffer(replay_path)
    loaded = DispatchDQN.load(path, device="cpu")
    obs, _ = evaluator.reset(seed=validation[0].seed, options={"scenario_index": 0})
    if (not np.array_equal(model.online_values(obs), loaded.online_values(obs))
            or any(not torch.equal(a, b) for a, b in zip(model.q_net.parameters(), loaded.q_net.parameters(), strict=True))
            or not np.array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])):
        raise RuntimeError("Checkpoint failed CPU replay verification")
    loaded.load_replay_buffer(replay_path)
    checks = replay_checks(loaded)
    assert checks == replay_checks(model)
    profile = {"policy": name, "training_seed": seed, "steps": model.num_timesteps,
               "train_wall_s": training_s, "validation_wall_s": evaluation_wall_s,
               "episodes": len(timing.episodes), "parameter_change_l2": parameter_change,
               "parameters": int(final.numel()), "gradient_updates": model._n_updates,
               "environment_wall_s": sum(row["step_wall_s"] for row in timing.steps),
               "reset_wall_s": sum(row["reset_wall_s"] for row in timing.resets),
               "update_wall_s": sum(row["update_wall_s"] for row in model.update_log),
               "solver_calls": sum(row["solver_calls"] for row in timing.steps),
               **checks}
    if any(row["invalid_request"] for row in timing.steps):
        raise RuntimeError("The DQN collector violated the public mask")
    return loaded, profile, snapshots, timing, model.update_log


def train_reference(cases, config, name, seed, destination):
    field, mode = TABULAR[name]
    cards = REFINED_CARDS if field == "refined_state" else None
    params = Parameters(config["alpha"], config["model"]["gamma"], time_aware=False)
    table = QTable(params, cardinalities=cards) if cards is not None else QTable(params)
    timing = Timing()
    env = make_env(cases, config, timing=timing)
    rng = np.random.default_rng(seed)
    start = perf_counter()
    _, info = env.reset(seed=seed)
    raw_return = learned_return = 0.
    feedback = Feedback(mode, config["potential_weight"])
    model = config["model"]
    decay_steps = max(1, round(model["exploration_fraction"] * config["steps"]))
    for i in range(config["steps"]):
        ratio = min(1., i / decay_steps)
        epsilon = model["exploration_initial_eps"] + ratio * (
            model["exploration_final_eps"] - model["exploration_initial_eps"]
        )
        state = info[field]
        action = table.choose(state, info["action_mask"], rng, epsilon)
        _, reward, ended, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("Tabular comparator requires the same full episode semantics")
        sim = env._sim
        signal = feedback.transform(reward, sim, env.public_observation,
                                    sim.dispatch_forecast() if mode == "early" and not ended else None,
                                    params.gamma, ended)
        raw_return += reward
        learned_return += signal
        table.update(state, action, signal, info.get(field, (0, 0, 0, 0)), info["action_mask"],
                     terminated=ended, elapsed_s=info["elapsed_s"])
        if ended:
            if not np.isclose(raw_return, learned_return, rtol=0., atol=1e-8):
                raise RuntimeError("Reference reward did not reconcile at the end of the day")
            if i + 1 < config["steps"]:
                _, info = env.reset()
                raw_return = learned_return = 0.
                feedback = Feedback(mode, config["potential_weight"])
    seconds = perf_counter() - start
    path = destination / f"{name}_{seed}_{config['steps']:04}.json"
    table.save(path, training_seed=seed, interactions=config["steps"], reward_mode=mode, observation=field)
    loaded = QTable.load(path, cardinalities=cards) if cards is not None else QTable.load(path)
    return loaded, {"policy": name, "training_seed": seed, "steps": config["steps"], "train_wall_s": seconds,
                    "episodes": len(timing.episodes), "solver_calls": sum(r["solver_calls"] for r in timing.steps)}, timing


def paired(frame, config):
    metrics = (*METRICS, "invalid_requests")
    means = frame.groupby(["policy", "day", "case"])[list(metrics)].mean()
    days = sorted(frame.day.unique())
    rng = np.random.default_rng(config["bootstrap_seed"])
    indices = rng.integers(len(days), size=(config["bootstrap_draws"], len(days)))
    rows = []
    neural = means.loc["dqn_continuous"]
    for name in sorted(set(frame.policy) - {"dqn_continuous"}):
        other = means.loc[name]
        if not neural.index.equals(other.index):
            raise RuntimeError("Pilot rows are not paired")
        gap = neural - other
        daily = gap.groupby(level="day").mean().loc[days].to_numpy()
        limits = np.quantile(daily[indices].mean(axis=1), [.025, .975], axis=0)
        for i, metric in enumerate(metrics):
            rows.append({"comparison": name, "metric": metric, "difference": gap[metric].mean(),
                         "ci_low": limits[0, i], "ci_high": limits[1, i], "cases": len(gap), "date_blocks": len(days)})
    return rows


def run(output: Path):
    start = perf_counter()
    config = load_config()
    if any((ROOT / name).resolve() in (output.resolve(), *output.resolve().parents) for name in ARCHIVES):
        raise ValueError("Pilot output would overwrite a validated experiment")
    torch.set_num_threads(config["threads"])
    development = pilot_banks(config, ("train", "validation"))
    for mode in MODES:
        check_gym(make_env(development["train"], config, mode=mode), skip_render_check=True)
        check_sb3(make_env(development["train"], config, mode=mode), warn=True)
    destination = output / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    models, tables = {}, {}
    profiles, train_steps, train_days, validation, updates = [], [], [], [], []

    def collect(profile, timing):
        profiles.append(profile)
        identity = {key: profile[key] for key in ("policy", "training_seed")}
        train_steps.extend({**identity, **row} for row in timing.steps)
        train_days.extend({**identity, **row} for row in timing.episodes)

    for mode in MODES:
        for seed in config["seeds"]:
            model, profile, points, timing, losses = train_dqn(
                development["train"], development["validation"], config, mode, seed, destination
            )
            models[mode, seed] = model
            collect(profile, timing)
            validation.extend(points)
            updates.extend({"policy": profile["policy"], "training_seed": seed, **row} for row in losses)
            print(f"{profile['policy']} seed={seed} steps={profile['steps']} updates={profile['gradient_updates']}",
                  flush=True)
    for name in TABULAR:
        for seed in config["seeds"]:
            table, profile, timing = train_reference(development["train"], config, name, seed, destination)
            tables[name, seed] = table
            collect(profile, timing)

    # The independent pilot check is generated only after fixing all pilot models.
    held = pilot_banks(config, ("test",))["test"]
    if {case.seed for case in held} & {case.seed for bank in development.values() for case in bank}:
        raise RuntimeError("Pilot hold-out overlaps development")
    env = make_env(held, config)
    records, traces = [], []
    for name in BASELINES:
        policy = random_policy(config["bootstrap_seed"]) if name == "random_feasible" else reference_policy(name)
        records.extend(evaluate_cases(held, env, name, policy, traces=traces))
    for (mode, seed), model in models.items():
        records.extend(evaluate_cases(held, make_env(held, config, mode=mode), f"dqn_{mode}",
                                      lambda obs, info, public: neural_row(model, obs, info, public),
                                      seed=seed, model=model, traces=traces))
    for (name, seed), table in tables.items():
        field = TABULAR[name][0]
        records.extend(evaluate_cases(held, env, name,
                                      lambda obs, info, public: table.choose(info[field], info["action_mask"]),
                                      seed=seed, traces=traces))
    for seed in (5107, 7613, 9817):
        table = QTable.load(ROOT / f"q_learning/checkpoints/seed_{seed}_episode_0180.json")
        records.extend(evaluate_cases(held, env, "historical_phase8",
                                      lambda obs, info, public: table.choose(info["original_state"], info["action_mask"]),
                                      seed=seed, traces=traces))
    frame = pd.DataFrame(records)
    if frame.groupby("case").opportunities.nunique().ne(1).any() or frame.unserved_mandatory.any():
        raise RuntimeError("Paired decision opportunities or mandatory service failed")
    data = {**development, "test": held}
    cases = []
    for row, case in zip(scenario_rows(data), (c for cases in data.values() for c in cases), strict=True):
        cases.append({**row, "events": [{**asdict(e), "kind": type(e).__name__} for e in case.scenario.events]})
    (output / "scenario_manifest.json").write_text(json.dumps(cases, default=sorted, indent=2) + "\n")
    for name, artifact in (("training_profile", profiles), ("train_steps", train_steps), ("train_episodes", train_days),
                           ("gradient_updates", updates), ("validation_curves", validation), ("pilot_cases", frame),
                           ("pilot_decisions", traces), ("paired_intervals", paired(frame, config))):
        write_table(output, name, artifact)
    (output / "manifest.json").write_text(json.dumps({
        "reference": "7ef38e4", "stage": "9A", "settings_sha256": sha256(CONFIG.read_bytes()).hexdigest(),
        "resolved_config_sha256": sha256((output / "config.yaml").read_bytes()).hexdigest(),
        "partitions": {key: len(cases) for key, cases in data.items()},
        "steps_per_run": config["steps"], "seeds": config["seeds"], "continuous_names": CONTINUOUS_NAMES,
        "torch": torch.__version__, "sb3": stable_baselines3.__version__, "device": "cpu",
        "wall_s": perf_counter() - start, "peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
    }, indent=2) + "\n")
    print(frame.groupby("policy")[["objective_mad", "otif", "regular_otif", "urgent_service_rate",
                                   "replan_attempts", "solver_calls", "invalid_requests"]].mean().round(4).to_string())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dqn_pilot")
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
