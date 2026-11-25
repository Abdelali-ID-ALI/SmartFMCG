"""Opt-in observations for the phase-eight distribution environment."""

from dataclasses import asdict
from time import perf_counter

import gymnasium as gym
import numpy as np

from src.rl.environment import DistributionEnv
from src.rl.refined_observations import CARDINALITIES, CONTINUOUS_NAMES, continuous, probe, tabular


class RefinementEnv(DistributionEnv):
    def __init__(self, episodes, *, encoding="original", probe_budget=32, tight_s=300, **kwargs):
        super().__init__(episodes, **kwargs)
        if encoding not in {"original", "tabular", "continuous"} or probe_budget < 1 or tight_s < 0:
            raise ValueError("Invalid observation configuration")
        self.encoding, self.probe_budget, self.tight_s = encoding, probe_budget, tight_s
        if encoding == "tabular":
            self.observation_space = gym.spaces.MultiDiscrete(CARDINALITIES)
        elif encoding == "continuous":
            self.observation_space = gym.spaces.Box(-1., 1., (len(CONTINUOUS_NAMES),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.probe_wall_s = 0.
        self.probe_candidates = 0
        self.probe_decisions = 0
        self.probe_urgent_wall_s = 0.
        self.probe_urgent_decisions = 0
        self.probe_exhaustions = 0
        return super().reset(seed=seed, options=options)

    def _public(self):
        original, info = super()._public()
        if self._current is None:
            return np.zeros(len(CONTINUOUS_NAMES), dtype=np.float32) if self.encoding == "continuous" else original, info
        started = perf_counter()
        measured = probe(self._current, self._sim.dispatch_forecast(), self._sim.settings, self.rates,
                         self.probe_budget)
        raw = continuous(self._current, measured, info["since_attempt_s"])
        refined = tabular(original, measured, info["waiting_urgent"], self.tight_s)
        took = perf_counter() - started
        self.probe_wall_s += took
        self.probe_candidates += measured.inspected
        self.probe_decisions += 1
        if info["waiting_urgent"]:
            self.probe_urgent_wall_s += took
            self.probe_urgent_decisions += 1
        self.probe_exhaustions += measured.exhausted
        info.update(original_state=tuple(map(int, original)), refined_state=tuple(map(int, refined)),
                    insertion_probe=asdict(measured), continuous_features=tuple(map(float, raw)))
        return {"original": original, "tabular": refined, "continuous": raw}[self.encoding], info
