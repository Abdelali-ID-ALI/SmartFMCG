"""Accrual of incremental logistics costs between irregular decision times."""

from dataclasses import dataclass, field

from src.optimization.fmcg_validator import UNITS_PER_CENT, UNITS_PER_MAD, operating_cost
from src.simulation.simulator import Simulator


@dataclass(frozen=True)
class RewardRates:
    delay_cents_per_min: int = 180
    overtime_cents_per_min: int = 300
    change_cents: int = 200
    attempt_cents: int = 50
    mandatory_omission_cents: int = 25000

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in (
            self.delay_cents_per_min, self.overtime_cents_per_min, self.change_cents,
            self.attempt_cents, self.mandatory_omission_cents
        )):
            raise ValueError("Reward rates require nonnegative integer centimes")


@dataclass
class Ledger:
    rates: RewardRates
    movement_index: int = 0
    departures: dict[int, int] = field(default_factory=dict)
    distances: dict[int, int] = field(default_factory=dict)
    returns: dict[int, int] = field(default_factory=dict)
    fixed: int = 0
    distance: int = 0
    delay_s: int = 0
    overtime_s: int = 0
    charged_units: int = 0
    closed: bool = False

    def advance(self, sim: Simulator, terminated: bool) -> tuple[float, dict[str, float]]:
        if self.closed:
            raise RuntimeError("Terminal omissions have already been settled")
        orders = sim.orders
        for entry in sim.movements[self.movement_index:]:
            truck = sim.problem.vehicles[entry.vehicle]
            if entry.kind == "depart":
                self.departures[entry.vehicle] = entry.time_s
                self.fixed += truck.fixed_cents * UNITS_PER_CENT
            elif entry.kind == "arrive":
                new = entry.distance_m - self.distances.get(entry.vehicle, 0)
                self.distance += operating_cost(truck, new, 0)[1]
                self.distances[entry.vehicle] = entry.distance_m
            elif entry.kind == "service_end":
                self.delay_s += max(0, entry.time_s - orders[entry.order_id].stop.window[1])
            elif entry.kind == "unload":
                self.returns[entry.vehicle] = entry.time_s
                self.overtime_s += max(0, entry.time_s - sim.problem.operating_window(truck)[1])
        self.movement_index = len(sim.movements)
        time_units = sum((self.returns.get(i, sim.now_s) - start) * sim.problem.vehicles[i].cents_per_hour * 5
                         for i, start in self.departures.items())
        cents = self.rates
        unmet = orders.keys() - sim.completed.keys() if terminated else ()
        omitted = sum(cents.mandatory_omission_cents if orders[key].stop.penalty_cents is None
                      else orders[key].stop.penalty_cents for key in unmet)
        totals = {"fixed_mad": self.fixed, "distance_mad": self.distance, "time_mad": time_units,
                  "lateness_mad": self.delay_s * cents.delay_cents_per_min * (UNITS_PER_CENT // 60),
                  "overtime_mad": self.overtime_s * cents.overtime_cents_per_min * (UNITS_PER_CENT // 60),
                  "instability_mad": sum(d.changes for d in sim.decisions) * cents.change_cents * UNITS_PER_CENT,
                  "attempt_mad": sum(d.modeled_latency_s > 0 for d in sim.decisions) * cents.attempt_cents * UNITS_PER_CENT,
                  "omission_mad": omitted * UNITS_PER_CENT}
        units = sum(totals.values())
        reward = (self.charged_units - units) / UNITS_PER_MAD
        self.charged_units = units
        self.closed = terminated
        return reward, {key: value / UNITS_PER_MAD for key, value in totals.items()}
