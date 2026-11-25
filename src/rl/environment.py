"""Gymnasium adapter over the validated distribution-day simulator."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import gymnasium as gym
import numpy as np

from src.optimization.fmcg_validator import UNITS_PER_MAD
from src.rl.actions import ACTION_NAMES, ENGINE_ACTIONS, DispatchAction, optimizer
from src.rl.observations import CARDINALITIES, encode
from src.rl.rewards import Ledger, RewardRates
from src.simulation.metrics import evaluate
from src.simulation.simulator import Simulator


@dataclass(frozen=True)
class Episode:
    key: str
    make_simulator: Callable[[], Simulator]


class DistributionEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, episodes: Sequence[Episode], *, rates: RewardRates = RewardRates(),
                 solver_budget: float = .004, max_steps: int | None = None):
        if not episodes or not np.isfinite(solver_budget) or solver_budget <= 0 or (
            max_steps is not None and (type(max_steps) is not int or max_steps < 1)
        ):
            raise ValueError("Expected scenarios, a positive solver budget and a valid step limit")
        self.episodes = tuple(episodes)
        self.action_space = gym.spaces.Discrete(len(DispatchAction))
        self.observation_space = gym.spaces.MultiDiscrete(CARDINALITIES)
        self.rates = rates
        self.solver_budget = solver_budget
        self.max_steps = max_steps
        self._done = True

    @property
    def public_observation(self):
        return self._current

    @property
    def episode_key(self) -> str:
        """Scenario provenance for evaluation logging outside the policy interface."""
        return self._key

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options
        index = options.get("scenario_index")
        if index is None:
            index = int(self.np_random.integers(len(self.episodes)))
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.episodes):
            raise ValueError("Unknown scenario index")
        episode = self.episodes[index]
        self._sim = episode.make_simulator()
        self._key = episode.key
        self._ledger = Ledger(self.rates)
        self._last_attempt = None
        self._steps = 0
        self._done = False
        self.last_wall_s = 0.
        self.last_planning_wall_s = 0.
        self._current = self._sim.next_observation()
        if self._current is None:
            raise ValueError("Distribution day must expose at least one decision")
        return self._public()

    def _public(self):
        if self._current is None:
            return np.zeros(len(CARDINALITIES), dtype=np.int64), {
                "action_mask": (1, 0, 0, 0), "now_s": self._sim.now_s,
            }
        observation = self._current
        state, info = encode(observation, self._sim.dispatch_forecast(), self._last_attempt,
                             sum(d.changes for d in self._sim.decisions), self._sim.settings.latency_s)
        mask = (1, 1, 1, 1) if observation.replan_available else (1, 0, 0, 0)
        return state, {"action_mask": mask, **info.public_dict()}

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset the environment before stepping")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError("Action outside the Gymnasium space")
        begin = perf_counter()
        simulation = self._sim
        start_s = self._current.now_s
        previous_wall = sum(d.planning_wall_s for d in simulation.decisions)
        pending_indices = [i for i, d in enumerate(simulation.decisions) if d.status == "pending"]
        index = len(simulation.decisions)
        requested = DispatchAction(int(action))
        planner = optimizer(self.solver_budget) if requested == DispatchAction.LOCAL_REOPTIMIZATION else None
        simulation.act(ENGINE_ACTIONS[requested], planner=planner)
        decision = simulation.decisions[-1]
        if decision.modeled_latency_s:
            self._last_attempt = start_s
        self._current = simulation.next_observation()
        terminated = self._current is None
        self._steps += 1
        truncated = bool(not terminated and self.max_steps is not None and self._steps >= self.max_steps)
        self._done = terminated or truncated
        reward, costs = self._ledger.advance(simulation, terminated)
        state, info = self._public()

        info.update(elapsed_s=simulation.now_s - start_s, decision=self._decision_record(index), costs_mad=costs,
                    settled_decisions=tuple(self._decision_record(i) for i in pending_indices
                                                                   if simulation.decisions[i].status != "pending"))
        self.last_planning_wall_s = sum(d.planning_wall_s for d in simulation.decisions) - previous_wall
        self.last_wall_s = perf_counter() - begin
        return state, reward, terminated, truncated, info

    def _decision_record(self, index: int) -> dict:
        item = self._sim.decisions[index]
        calls, status = self._sim.planning_details.get(index, (0, ""))
        fallback = item.status if item.status in {
            "busy", "unavailable", "invalid_candidate", "invalid_at_application"
        } else ""
        if not fallback and status in {"INFEASIBLE", "UNKNOWN", "MODEL_INVALID", "infeasible_window"}:
            fallback = f"solver_{status.lower()}"
        return {"decision_index": index, "time_s": item.time_s, "requested": ACTION_NAMES[item.requested],
                "executed": ACTION_NAMES[item.executed], "status": item.status, "fallback_reason": fallback,
                "attempted": bool(item.modeled_latency_s), "applied_s": item.applied_s,
                "modeled_latency_s": item.modeled_latency_s, "solver_calls": calls, "solver_status": status,
                "search_items": item.candidates, "changes": item.changes}

    def decision_trace(self) -> tuple[dict, ...]:
        """Finalized diagnostics with nondeterministic timings, separate from Gym observations."""
        if not self._ledger.closed:
            raise RuntimeError("Decision trace requires a complete day")
        return tuple({**self._decision_record(i), "planning_wall_s": item.planning_wall_s}
                     for i, item in enumerate(self._sim.decisions) if item.requested != "KEEP")

    def episode_metrics(self) -> dict[str, float]:
        if not self._ledger.closed:
            raise RuntimeError("Operational metrics require a complete day")
        metrics = evaluate(self._sim)
        counts = {name: 0 for name in ACTION_NAMES.values()}
        applied = dict(counts)
        for decision in self._sim.decisions:
            counts[ACTION_NAMES[decision.requested]] += 1
            applied[ACTION_NAMES[decision.executed]] += 1
        return {**metrics, "solver_calls": sum(calls for calls, _ in self._sim.planning_details.values()),
                "safety_fallbacks": sum(d.status in {"busy", "unavailable", "invalid_candidate", "invalid_at_application"}
                                        for d in self._sim.decisions),
                "solver_no_incumbent": sum(status in {"INFEASIBLE", "UNKNOWN", "MODEL_INVALID"}
                                            for _, status in self._sim.planning_details.values()),
                "objective_mad": self._ledger.charged_units / UNITS_PER_MAD,
                **{f"requested_{name}": count for name, count in counts.items()},
                **{f"executed_{name}": count for name, count in applied.items()}}
