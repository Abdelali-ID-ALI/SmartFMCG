"""Regression checks for the bounded Phase 9B protocol and actual SB3 loss."""

import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import ReplayBufferSamples
import torch

from src.rl.dqn import WIDTH, create_model
from src.rl.dqn_evaluation import Budget, TimeLimit, comparison, select_checkpoint
from test_dqn_pilot import adapted, tiny_config


def test_sb3_update_bootstraps_only_admissible_actions_and_seals_terminal():
    torch.set_num_threads(2)
    model = create_model(adapted(), tiny_config(), seed=41)
    model.set_logger(configure(format_strings=[]))
    with torch.no_grad():
        for network in (model.q_net, model.q_net_target):
            network.q_net[-1].weight.zero_()
        model.q_net.q_net[-1].bias.copy_(torch.tensor([1., 0., 0., 0.]))
        model.q_net_target.q_net[-1].bias.copy_(torch.tensor([2., 10., 30., 40.]))
    states = torch.zeros((3, WIDTH))
    next_states = states.clone()
    next_states[1:, 0] = 1.
    samples = ReplayBufferSamples(states, torch.zeros(3, 1, dtype=torch.long), next_states,
                                  torch.tensor([[0.], [0.], [1.]]), torch.tensor([[-2.], [-2.], [-2.]]))
    model.replay_buffer.sample = lambda *args, **kwargs: samples
    model.train(gradient_steps=1, batch_size=3)
    # Targets 0, 38, and -2, with current values of 1 in all three rows.
    assert model.logger.name_to_value["train/loss"] == pytest.approx((.5 + 36.5 + 2.5) / 3)


def test_interaction_and_time_budget_are_cumulative(tmp_path):
    limits = {"dqn": 10, "tabular": 20, "wall_s": 1200, "checkpoint_reserve_s": 25}
    first = Budget(tmp_path, limits, "screen")
    first.charge("dqn", 4)
    first.sync("complete")
    next_stage = Budget(tmp_path, limits, "fit")
    next_stage.require("dqn", 6)
    with pytest.raises(ValueError, match="authorized interactions"):
        next_stage.require("dqn", 7)
    next_stage.charge("tabular", 20)
    with pytest.raises(ValueError, match="budget"):
        next_stage.charge("tabular")
    next_stage.sync("complete")
    assert json.loads((tmp_path / "budget.json").read_text())["dqn"] == 4
    next_stage.before = 1190
    with pytest.raises(TimeLimit):
        next_stage.check()


def test_checkpoint_selection_uses_validation_only_and_keeps_earlier_tie(tmp_path):
    (tmp_path / "validation.csv").write_text("step,objective_mad\n0,1\n1000,11\n1000,9\n2000,10\n")
    (tmp_path / "final_cases.csv").write_text("step,objective_mad\n2000,-100000\n")
    assert select_checkpoint(tmp_path) == (1000, 10.)


def test_paired_bootstrap_keeps_matched_seed_and_date_effects():
    import pandas as pd
    from src.rl.dqn_evaluation import OUTCOME_METRICS

    rows = []
    for name, effect in (("dqn", 3), ("table", 1)):
        for day in (1, 2, 3):
            for seed in (7, 8, 9):
                rows.append({"policy": name, "fleet": 3, "day": day, "case": day, "training_seed": seed,
                             **{metric: effect + day * seed for metric in OUTCOME_METRICS}})
    _, intervals = comparison(pd.DataFrame(rows), {"bootstrap_seed": 3, "bootstrap_draws": 100,
                                                    "seeds": [7, 8, 9]}, "dqn")
    assert all(np.allclose([row["difference"], row["ci_low"], row["ci_high"]], 2.) for row in intervals)
