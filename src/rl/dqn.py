"""CPU SB3 DQN with the public dispatch eligibility mask."""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from src.rl.actions import DispatchAction
from src.rl.refined_observations import CONTINUOUS_NAMES
from src.rl.refinement_env import RefinementEnv

WIDTH = len(CONTINUOUS_NAMES)


@dataclass
class Timing:
    steps: list[dict] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)
    resets: list[dict] = field(default_factory=list)


def compact_control(original: tuple[int, ...], eligible: bool) -> np.ndarray:
    """Give the control network the same input width and only phase-eight bins."""
    context, risk, remaining, recent = original
    result = np.zeros(WIDTH, dtype=np.float32)
    result[0] = eligible
    result[1 + context] = 1.
    result[6 + risk] = 1.
    result[9:11] = (remaining, recent)
    return result


class DQNEnv(RefinementEnv):
    """Use phase-eight economics and the validated continuous Gymnasium space."""

    def __init__(self, episodes, *, input_mode: str = "continuous", timing: Timing | None = None, **kwargs):
        if input_mode not in {"continuous", "compact"}:
            raise ValueError("Unknown DQN input")
        self.input_mode, self.timing = input_mode, timing
        super().__init__(episodes, encoding="continuous", **kwargs)

    def reset(self, *, seed=None, options=None):
        started = perf_counter()
        observation, info = super().reset(seed=seed, options=options)
        self._episode_started = started
        self._episode_return = 0.
        self._episode_steps = 0
        self._episode_environment_s = perf_counter() - started
        if self.timing is not None:
            self.timing.resets.append({"case": self.episode_key, "reset_wall_s": self._episode_environment_s})
        return observation, info

    def _public(self):
        observation, info = super()._public()
        eligible = bool(info["action_mask"][1])
        if tuple(info["action_mask"]) != (1, int(eligible), int(eligible), int(eligible)):
            raise RuntimeError("DQN eligibility feature requires the phase-eight action mask")
        if self._current is not None and self.input_mode == "compact":
            observation = compact_control(info["original_state"], eligible)
        if bool(observation[0]) != eligible:
            raise RuntimeError("Observation eligibility and dispatch mask disagree")
        return observation, info

    def step(self, action):
        started = perf_counter()
        eligible = self.public_observation is not None and self.public_observation.replan_available
        observation, reward, terminated, truncated, info = super().step(action)
        wall = perf_counter() - started
        self._episode_environment_s += wall
        self._episode_return += reward
        self._episode_steps += 1
        decision = info["decision"]
        if self.timing is not None:
            self.timing.steps.append({
                "step": len(self.timing.steps) + 1, "case": self.episode_key,
                "step_wall_s": wall, "planning_wall_s": self.last_planning_wall_s,
                "requested": decision["requested"], "executed": decision["executed"],
                "status": decision["status"], "fallback": decision["fallback_reason"],
                "solver_calls": decision["solver_calls"], "invalid_request": bool(action and not eligible),
                "retained_reward_mad": reward, "terminated": terminated, "truncated": truncated,
            })
        if terminated:
            metrics = self.episode_metrics()
            if not np.isclose(-self._episode_return, metrics["objective_mad"], rtol=0., atol=1e-8):
                raise RuntimeError("DQN day did not reconcile with phase-eight costs")
            if self.timing is not None:
                self.timing.episodes.append({"global_step": len(self.timing.steps), "case": self.episode_key,
                                             "steps": self._episode_steps, "episode_return": self._episode_return,
                                             "episode_environment_wall_s": self._episode_environment_s,
                                             "episode_wall_s": perf_counter() - self._episode_started, **metrics})
        return observation, reward, terminated, truncated, info


class DispatchQNetwork(QNetwork):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        values = super().forward(obs)
        keep = torch.arange(values.shape[-1], device=values.device) == DispatchAction.KEEP_PLAN
        allowed = (obs[..., :1] > .5) | keep
        return values.masked_fill(~allowed, torch.finfo(values.dtype).min)


class DispatchPolicy(DQNPolicy):
    def make_q_net(self) -> DispatchQNetwork:
        kwargs = self._update_features_extractor(self.net_args, features_extractor=None)
        return DispatchQNetwork(**kwargs).to(self.device)


class DispatchDQN(DQN):
    """Keep SB3's replay, optimizer, DQN target and target-network schedule."""

    def _random_valid(self, observation: np.ndarray) -> np.ndarray:
        eligible = np.asarray(observation)[..., 0] > .5
        choices = np.random.randint(self.action_space.n, size=eligible.shape)
        return np.where(eligible, choices, 0).astype(np.int64)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            action = self._random_valid(self._last_obs)
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.random() < self.exploration_rate:
            return self._random_valid(observation), state
        return self.policy.predict(observation, state, episode_start, deterministic)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        started = perf_counter()
        super().train(gradient_steps, batch_size)
        elapsed = perf_counter() - started
        if hasattr(self, "update_log"):
            self.update_log.append({"step": self.num_timesteps, "gradient_steps": gradient_steps,
                                    "update_wall_s": elapsed, "epsilon": self.exploration_rate,
                                    "loss": self.logger.name_to_value["train/loss"]})

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "update_log"]

    def online_values(self, observation):
        values, _ = self.policy.obs_to_tensor(observation)
        with torch.inference_mode():
            result = self.q_net(values).cpu().numpy()[0]
        return result if observation[0] > .5 else result[:1]


def create_model(env: DQNEnv, config: dict, seed: int) -> DispatchDQN:
    settings = dict(config)
    architecture = settings.pop("net_arch")
    return DispatchDQN(DispatchPolicy, env, **settings, seed=seed, device="cpu", verbose=0,
                       policy_kwargs={"net_arch": architecture, "activation_fn": torch.nn.ReLU})
