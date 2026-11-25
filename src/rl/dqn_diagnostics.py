"""Retrospective diagnostics. Neither fork outcomes nor replay labels reach a policy."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.rl.actions import ENGINE_ACTIONS, DispatchAction, optimizer
from src.rl.dqn import DispatchDQN, compact_control
from src.rl.dqn_pilot import make_env, reference_policy, replay_checks
from src.rl.refinement_experiment import objective
from src.simulation.metrics import evaluate


def replay_analysis(model: DispatchDQN) -> list[dict]:
    """Summarize chronological transitions, honoring ring order and real terminals."""
    replay_checks(model)
    buffer = model.replay_buffer
    order = (np.arange(buffer.size()) + (buffer.pos if buffer.full else 0)) % buffer.buffer_size
    obs = buffer.observations[order, 0]
    nxt = buffer.next_observations[order, 0]
    actions = buffer.actions[order, 0, 0]
    rewards = buffer.rewards[order, 0]
    terminal = (buffer.dones[order, 0] * (1 - buffer.timeouts[order, 0])).astype(bool)
    boundary = buffer.dones[order, 0].astype(bool)
    with torch.inference_mode():
        current = model.q_net(torch.as_tensor(obs, device=model.device)).cpu().numpy()
        next_input = torch.as_tensor(nxt, device=model.device)
        target = model.q_net_target(next_input).cpu().numpy()
        online_next = model.q_net(next_input).cpu().numpy()
    best_target = target.max(axis=1)
    best_online = online_next.argmax(axis=1)
    selected = current[np.arange(len(actions)), actions]
    bellman = rewards + model.gamma * (~terminal) * best_target
    target_max_gap = best_target - target[np.arange(len(actions)), best_online]
    continuation_error = np.full(len(actions), np.nan)
    result = []
    future = 0.
    complete = False
    horizon = 0
    horizons = np.zeros(len(actions), dtype=int)
    for i in reversed(range(len(actions))):
        if boundary[i]:
            future, horizon, complete = 0., 0, bool(terminal[i])
        future = rewards[i] + model.gamma * future
        if complete:
            continuation_error[i] = selected[i] - future
            horizons[i] = horizon
        horizon += 1
    eligible = obs[:, 0] > .5
    for name, group in (("all", np.ones(len(actions), dtype=bool)), ("eligible", eligible),
                        ("ineligible", ~eligible)):
        for action in DispatchAction:
            subset = group & (actions == action)
            if not subset.any():
                continue
            valid_mc = subset & np.isfinite(continuation_error)
            next_eligible = subset & ~terminal & (nxt[:, 0] > .5)
            limited = subset & ~terminal & (nxt[:, 0] <= .5)
            if not np.array_equal(best_target[limited], target[limited, 0]):
                raise RuntimeError("A masked action entered a Bellman target")
            row = {"eligibility": name, "action": action.name, "count": int(subset.sum()),
                   "share_in_group": float(subset.sum() / group.sum()),
                   "terminal_count": int(terminal[subset].sum()),
                   "reward_mean": float(rewards[subset].mean()),
                   "reward_zero_share": float((rewards[subset] == 0).mean()),
                   "td_abs_mean": float(np.abs(bellman[subset] - selected[subset]).mean()),
                   "terminal_abs_reward_share": float(np.abs(rewards[subset & terminal]).sum() /
                                                       max(1e-12, np.abs(rewards[subset]).sum())),
                   "masked_next_count": int(limited.sum()),
                   "target_max_minus_online_choice": (float(target_max_gap[next_eligible].mean())
                                                      if next_eligible.any() else None),
                   "target_online_agreement": (float((target.argmax(axis=1)[next_eligible] ==
                                                      best_online[next_eligible]).mean())
                                               if next_eligible.any() else None),
                   "q_minus_behavior_return": (float(continuation_error[valid_mc].mean())
                                               if valid_mc.any() else None),
                   "behavior_return_count": int(valid_mc.sum()),
                   "steps_to_terminal_mean": float(horizons[valid_mc].mean()) if valid_mc.any() else None}
            for label, values in (("reward", rewards), ("q_played", selected), ("bellman", bellman)):
                for percentile, value in zip((5, 50, 95), np.percentile(values[subset], [5, 50, 95]), strict=True):
                    row[f"{label}_p{percentile}"] = float(value)
            result.append(row)
    return result


def pilot_replay(root: Path) -> list[dict]:
    checkpoints = root / "dqn_pilot/checkpoints"
    rows = []
    for mode in ("continuous", "compact"):
        for seed in (1901, 1907):
            stem = f"dqn_{mode}_{seed}"
            model = DispatchDQN.load(checkpoints / f"{stem}_0768.zip", device="cpu")
            model.load_replay_buffer(checkpoints / f"{stem}_replay.pkl")
            rows.extend({"mode": mode, "seed": seed, **row} for row in replay_analysis(model))
    return rows


def controlled_forks(cases, config, models, check) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute one distinct action per clone, then keep to the natural day end."""
    env = make_env(cases, config)
    rows, support, scores = [], [], []
    baseline = reference_policy("multi_feature")
    for i, case in enumerate(cases):
        obs, info = env.reset(seed=case.seed, options={"scenario_index": i})
        ordinal = 0
        while True:
            check()
            current = env.public_observation
            eligible = bool(info["action_mask"][1])
            waiting = len(current.waiting_urgent)
            identity = {"split": case.split, "case": case.key, "day": str(case.day), "regime": case.regime,
                        "decision_s": current.now_s, "eligible": eligible, "waiting": waiting,
                        "event_types": "|".join(current.event_types),
                        "original_state": str(info["original_state"]), "refined_state": str(info["refined_state"])}
            support.append(identity)
            if eligible and (waiting or ordinal % config["diagnostic_stride"] == 0):
                outcomes = {}
                for action in DispatchAction:
                    fork = deepcopy(env._sim)
                    planner = optimizer(config["solver_budget"]) if action == DispatchAction.LOCAL_REOPTIMIZATION else None
                    fork.act(ENGINE_ACTIONS[action], planner=planner)
                    first_time = fork.now_s
                    for _ in iter(fork.next_observation, None):
                        fork.act(ENGINE_ACTIONS[DispatchAction.KEEP_PLAN])
                    metrics = evaluate(fork)
                    if metrics["unserved_mandatory"]:
                        raise RuntimeError("Diagnostic fork failed independent validation")
                    outcomes[action] = (objective(metrics, env.rates), metrics["urgent_served"], fork.now_s - first_time)
                keep = outcomes[DispatchAction.KEEP_PLAN][0]
                for action, (loss, served, horizon) in outcomes.items():
                    rows.append({**identity, "action": action.name, "gain_mad": keep - loss,
                                 "urgent_served": served, "remaining_time_s": horizon,
                                 **info["insertion_probe"]})
                best_gain = keep - min(loss for loss, _, _ in outcomes.values())
                for name, mode, model in models:
                    inputs = obs if mode == "continuous" else compact_control(info["original_state"], eligible)
                    values = model.online_values(inputs)
                    chosen = DispatchAction(int(model.predict(inputs, deterministic=True)[0]))
                    for action, q in zip(DispatchAction, values, strict=True):
                        scores.append({**identity, "model": name, "action": action.name,
                                       "chosen": chosen.name, "q_mad": float(q),
                                       "q_minus_keep_mad": float(q - values[0]), "gain_mad": keep - outcomes[action][0],
                                       "best_gain_mad": best_gain,
                                       "chosen_gain_mad": keep - outcomes[chosen][0]})
            ordinal += 1
            obs, _, ended, _, info = env.step(baseline(obs, info, current))
            if ended:
                break
    return pd.DataFrame(rows), pd.DataFrame(support), pd.DataFrame(scores)
