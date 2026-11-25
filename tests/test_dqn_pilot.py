"""Independent DQN environment, replay, masking and CPU persistence checks."""

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
import torch
from gymnasium.utils.env_checker import check_env as check_gym
from stable_baselines3.common.env_checker import check_env as check_sb3

from src.rl.actions import DispatchAction
from src.rl.dqn import DQNEnv, DispatchDQN, WIDTH, create_model
from src.rl.dqn_pilot import load_config, pilot_banks, replay_checks
from test_q_learning import miniature


def adapted(mode="continuous", **kwargs):
    core = miniature(**kwargs)
    return DQNEnv(core.episodes, input_mode=mode, tight_s=30, probe_budget=2, max_steps=core.max_steps)


def tiny_config():
    config = load_config()["model"]
    return {**config, "buffer_size": 64, "learning_starts": 4, "batch_size": 4,
            "train_freq": 2, "target_update_interval": 8}


@pytest.mark.parametrize("mode", ["continuous", "compact"])
def test_gymnasium_and_sb3_inputs_preserve_information_boundary(mode):
    ordinary = adapted(mode)
    check_gym(ordinary, skip_render_check=True)
    check_sb3(ordinary)
    alternative = adapted(mode, delayed=True, hidden=((0, 10, 10), (10, 0, 10), (25, 10, 0)))
    a, x = ordinary.reset(seed=40)
    b, y = alternative.reset(seed=40)
    while not x.get("waiting_urgent"):
        assert x == y and np.array_equal(a, b)
        assert a.shape == (22,) and a.dtype == np.float32 and ordinary.observation_space.contains(a)
        a, _, _, _, x = ordinary.step(0)
        b, _, _, _, y = alternative.step(0)
    assert x == y and np.array_equal(a, b)
    if mode == "compact":
        assert np.all(a[11:] == 0) and a[1:6].sum() == 1 and a[6:9].sum() == 1


def test_actual_action_execution_and_common_reward():
    env = adapted()
    observation, info = env.reset(seed=1)
    total = 0.
    while True:
        action = DispatchAction.GLOBAL_REOPTIMIZATION if info.get("waiting_urgent") else DispatchAction.KEEP_PLAN
        observation, reward, terminated, _, info = env.step(action)
        total += reward
        if terminated:
            break
    assert env.episode_metrics()["urgent_served"] == 1
    assert env.episode_metrics()["objective_mad"] == pytest.approx(-total)
    applied = [d for d in env.decision_trace() if d["status"] == "applied"]
    assert applied[0]["requested"] == applied[0]["executed"] == "GLOBAL_REOPTIMIZATION"


def test_unavailable_request_uses_validated_environment_fallback():
    env = adapted()
    _, info = env.reset(seed=1)
    while info["action_mask"][1]:
        _, _, _, _, info = env.step(0)
    _, _, _, _, info = env.step(2)
    assert info["decision"]["requested"] == "LOCAL_REOPTIMIZATION"
    assert info["decision"]["executed"] == "KEEP_PLAN"
    assert info["decision"]["fallback_reason"] == "unavailable"
    assert info["decision"]["solver_calls"] == 0


def test_mask_controls_greedy_random_and_target_network():
    torch.set_num_threads(2)
    model = create_model(adapted(), tiny_config(), seed=40)
    eligible = np.zeros(WIDTH, dtype=np.float32)
    blocked = eligible.copy()
    eligible[0] = 1
    with torch.no_grad():
        for network in (model.q_net, model.q_net_target):
            network.q_net[-1].weight.zero_()
            network.q_net[-1].bias.copy_(torch.tensor([1., 2., 3., 4.]))
    assert int(model.predict(eligible, deterministic=True)[0]) == 3
    assert int(model.predict(blocked, deterministic=True)[0]) == 0
    model.exploration_rate = 1.
    assert {int(model.predict(eligible)[0]) for _ in range(64)} == {0, 1, 2, 3}
    assert all(int(model.predict(blocked)[0]) == 0 for _ in range(32))
    batch = torch.tensor(np.stack((blocked, eligible)))
    assert model.q_net_target(batch).max(dim=1).values.tolist() == [1., 4.]


def test_replay_bootstraps_truncation_and_stops_on_real_termination():
    torch.set_num_threads(2)
    truncated = create_model(adapted(max_steps=2), tiny_config(), seed=42)
    truncated.learn(24)
    assert replay_checks(truncated)["timeouts"] == 12
    assert replay_checks(truncated)["real_terminations"] == 0
    assert not truncated.replay_buffer.sample(50).dones.any()
    normal = create_model(adapted(), tiny_config(), seed=42)
    normal.learn(24)
    checks = replay_checks(normal)
    assert checks["buffer_size"] == 24 and checks["real_terminations"] >= 1 and checks["timeouts"] == 0


def test_training_updates_weights_replays_and_round_trips_on_cpu(tmp_path):
    torch.set_num_threads(2)
    first = create_model(adapted(), tiny_config(), seed=13)
    initial = [p.detach().clone() for p in first.q_net.parameters()]
    first.update_log = []
    first.learn(40)
    second = create_model(adapted(), tiny_config(), seed=13)
    second.update_log = []
    second.learn(40)
    for model in (first, second):
        assert model._n_updates > 0 and replay_checks(model)["buffer_size"] == 40
        assert all(np.isfinite(row["loss"]) for row in model.update_log)
    assert any(not torch.equal(a, b) for a, b in zip(initial, first.q_net.parameters(), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(first.q_net.parameters(), second.q_net.parameters(), strict=True))
    target = tmp_path / "model.zip"
    replay = tmp_path / "replay.pkl"
    first.save(target)
    first.save_replay_buffer(replay)
    restored = DispatchDQN.load(target, device="cpu")
    restored.load_replay_buffer(replay)
    assert replay_checks(restored) == replay_checks(first)
    obs, _ = adapted().reset(seed=35)
    assert np.array_equal(restored.online_values(obs), first.online_values(obs))
    assert int(restored.predict(obs, deterministic=True)[0]) == int(first.predict(obs, deterministic=True)[0])


def test_pilot_partitions_are_new_and_role_disjoint():
    banks = pilot_banks(load_config(), ("train", "validation", "test"))
    assert {name: len(rows) for name, rows in banks.items()} == {"train": 12, "validation": 6, "test": 10}
    assert len({case.seed for part in banks.values() for case in part}) == 28
    dates = [{case.day for case in part} for part in banks.values()]
    assert not any(a & b for i, a in enumerate(dates) for b in dates[i + 1:])
