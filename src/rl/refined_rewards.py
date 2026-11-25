"""Training-only retiming and discount-consistent potential shaping."""

from dataclasses import dataclass, field
from math import isfinite

from src.optimization.fmcg_validator import UNITS_PER_MAD, operating_cost
from src.simulation.replanning import Forecast
from src.simulation.simulator import Simulator
from src.simulation.state import Observation

MODES = ("original", "early", "potential", "distance")


def allocated_orders(obs: Observation) -> set[str]:
    return {key for view in obs.vehicles for key in (*view.tail, view.committed) if key is not None}


def irreversible(obs: Observation, forecast: Forecast, regular_ids: set[str]) -> dict[str, int]:
    """Public proof of omission for a released, unallocated urgent order."""
    allocated = allocated_orders(obs)
    return {key: order.stop.penalty_cents for key, order in forecast.orders.items()
            if key not in regular_ids and key not in allocated and key not in obs.completed
            and obs.now_s + order.stop.service_s > order.stop.window[1]}


def potential(obs: Observation, forecast: Forecast, regular_ids: set[str], weight: float) -> float:
    allocated = allocated_orders(obs)
    return -weight * sum(order.stop.penalty_cents * (.5 if key in allocated else 1.) / 100
                         for key, order in forecast.orders.items()
                         if key not in regular_ids and key not in obs.completed)


@dataclass
class Feedback:
    mode: str
    weight: float
    scale: float = 1.
    advance_cents: dict[str, int] = field(default_factory=dict)
    first_debit_s: dict[str, int] = field(default_factory=dict)
    movement_index: int = 0
    outstanding_distance: dict[int, int] = field(default_factory=dict)
    correction: float = 0.

    def __post_init__(self):
        if self.mode not in MODES or not isfinite(self.scale) or self.scale <= 0 or not 0 <= self.weight <= 1:
            raise ValueError("Invalid training reward")

    def initial(self, obs, forecast, regular_ids):
        if self.mode == "potential":
            self.correction = potential(obs, forecast, regular_ids, self.weight)
        return self.correction

    def transform(self, reward: float, sim: Simulator, obs: Observation | None,
                  forecast: Forecast | None, discount: float, terminated: bool) -> float:
        previous = self.correction
        if self.mode == "early":
            if obs is not None:
                for key, cents in irreversible(obs, forecast, set(sim.products)).items():
                    if key not in self.advance_cents:
                        self.advance_cents[key] = cents
                        self.first_debit_s[key] = obs.now_s
            if terminated and self.advance_cents.keys() & sim.completed.keys():
                raise RuntimeError("A supposedly irreversible omission was later served")
            self.correction = 0. if terminated else -sum(self.advance_cents.values()) / 100
        elif self.mode == "distance":
            for event in sim.movements[self.movement_index:]:
                if event.kind == "leg":
                    distance = sim.problem.distance_m[event.node][event.destination]
                    truck = sim.problem.vehicles[event.vehicle]
                    self.outstanding_distance[event.vehicle] = operating_cost(truck, distance, 0)[1]
                elif event.kind == "arrive":
                    self.outstanding_distance.pop(event.vehicle)
            self.movement_index = len(sim.movements)
            if terminated and self.outstanding_distance:
                raise RuntimeError("Committed distances failed to settle")
            self.correction = -sum(self.outstanding_distance.values()) / UNITS_PER_MAD
        elif self.mode == "potential":
            self.correction = 0. if terminated else potential(obs, forecast, set(sim.products), self.weight)
            return (reward + discount * self.correction - previous) * self.scale
        return (reward + self.correction - previous) * self.scale
