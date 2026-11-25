"""Sparse tabular Q-learning with masked epsilon-greedy behavior."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import json

import numpy as np

from src.rl.actions import DispatchAction
from src.rl.observations import CARDINALITIES

State = tuple[int, ...]
ACTIONS = len(DispatchAction)


@dataclass(frozen=True)
class Parameters:
    alpha: float
    gamma: float
    time_aware: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1 or not 0 < self.gamma <= 1:
            raise ValueError("Invalid Q-learning hyperparameters")

    def discount(self, elapsed_s: int) -> float:
        return self.gamma ** (elapsed_s / 900 if self.time_aware else 1.)


class QTable:
    def __init__(self, parameters: Parameters, *, cardinalities: tuple[int, ...] = CARDINALITIES):
        self.parameters = parameters
        self.cardinalities = cardinalities
        self.values: dict[State, np.ndarray] = {}
        self.counts: dict[State, np.ndarray] = {}

    def choose(self, state: State, mask: tuple[int, ...], rng: np.random.Generator | None = None,
               epsilon: float = 0.) -> int:
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError("At least one action must be available")
        if rng is not None and rng.random() < epsilon:
            return int(rng.choice(valid))
        if state not in self.values:
            return int(valid[0])
        if rng is None:
            trained = valid[self.counts[state][valid] > 0]
            if len(trained):
                valid = trained
            else:
                return int(valid[0])
        return int(valid[np.argmax(self.values[state][valid])])

    def update(self, state: State, action: int, reward: float, following: State,
               following_mask: tuple[int, ...], *, terminated: bool, elapsed_s: int) -> float:
        values = self.values.setdefault(state, np.zeros(ACTIONS))
        counts = self.counts.setdefault(state, np.zeros(ACTIONS, dtype=np.int64))
        future = 0.
        if not terminated and following in self.values:
            future = np.max(self.values[following][np.asarray(following_mask, dtype=bool)])
        target = reward + self.parameters.discount(elapsed_s) * future
        values[action] += self.parameters.alpha * (target - values[action])
        counts[action] += 1
        return float(values[action])

    def save(self, path: Path, **metadata) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump({"schema": 1, "cardinalities": self.cardinalities, "actions": [a.name for a in DispatchAction],
                       "parameters": vars(self.parameters), "metadata": metadata,
                       "entries": [{"state": key, "values": self.values[key].tolist(), "visits": self.counts[key].tolist()}
                                   for key in sorted(self.values)]}, stream, indent=2)
            stream.write("\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, *, cardinalities: tuple[int, ...] = CARDINALITIES) -> "QTable":
        return cls.from_json(path.read_bytes(), cardinalities=cardinalities)

    @classmethod
    def from_json(cls, contents: bytes, *, cardinalities: tuple[int, ...] = CARDINALITIES) -> "QTable":
        raw = json.loads(contents)
        if (raw["schema"] != 1 or tuple(raw["cardinalities"]) != cardinalities
                or raw["actions"] != [a.name for a in DispatchAction]):
            raise ValueError("Q-table feature or action schema mismatch")
        result = cls(Parameters(**raw["parameters"]), cardinalities=cardinalities)
        for row in raw["entries"]:
            key = tuple(row["state"])
            if (len(key) != len(cardinalities) or key in result.values
                    or any(type(k) is not int or not 0 <= k < size for k, size in zip(key, cardinalities, strict=True))
                    or len(row["values"]) != ACTIONS or any(not isfinite(q) for q in row["values"])
                    or len(row["visits"]) != ACTIONS or any(type(n) is not int or n < 0 for n in row["visits"])):
                raise ValueError("Invalid Q-table entry")
            result.values[key] = np.array(row["values"], dtype=float)
            result.counts[key] = np.array(row["visits"], dtype=np.int64)
        return result
