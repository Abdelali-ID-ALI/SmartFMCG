"""Public dispatch indicators and fixed, interpretable tabular bins."""

from dataclasses import dataclass, asdict

import numpy as np

from src.rl.actions import can_supply
from src.simulation.replanning import Forecast
from src.simulation.state import Observation


# Context, pressure, workload, recent computation.
CARDINALITIES = (5, 3, 2, 2)


@dataclass(frozen=True)
class Indicators:
    now_s: int
    event_types: tuple[str, ...]
    remaining_regular: int
    waiting_urgent: int
    active_vehicles: int
    regular_stock_crates: int
    spare_capacity: tuple[int, ...]
    congestion_factor: float
    predicted_delay_s: int
    at_risk: int
    priority_at_risk: int
    direct_urgent_slack_s: int | None
    matching_reserve: bool
    since_attempt_s: int | None
    route_changes: int

    def public_dict(self) -> dict:
        return asdict(self)


def encode(obs: Observation, forecast: Forecast, last_attempt_s: int | None, changes: int,
           latency_s: int) -> tuple[np.ndarray, Indicators]:
    fleet = [view for view in obs.vehicles if view.mutable]
    regular = [key for key, order in forecast.orders.items() if order.owner is not None]
    left = sum(key not in obs.completed for key in regular)
    spare = tuple(sum(forecast.problem.vehicles[view.index].capacity[d] - view.onboard[d] for view in fleet)
                  for d in range(4))
    slacks = []
    matching = False
    for key in obs.waiting_urgent:
        order = forecast.orders[key]
        for view in fleet:
            if can_supply(forecast, view, key):
                matching = True
                start = max(view.anchor_s, obs.now_s + latency_s)
                travel = forecast.travel(view.anchor_node, order.node, start)
                if travel is not None:
                    end = max(start + travel, order.stop.window[0]) + forecast.service(key)
                    slacks.append(order.stop.window[1] - end)
    since = None if last_attempt_s is None else obs.now_s - last_attempt_s
    info = Indicators(obs.now_s, obs.event_types, left, len(obs.waiting_urgent), len(fleet),
                      sum(view.regular[2] for view in fleet), spare, obs.congestion_factor,
                      obs.predicted_delay_s, obs.at_risk, obs.priority_at_risk, max(slacks, default=None),
                      matching, since, changes)
    if not obs.replan_available:
        state = (0, 0, 0, 0)
    else:
        if obs.waiting_urgent:
            context = 3 if matching else 4
        else:
            context = 2 if obs.event_types or obs.congestion_factor >= 1.3 else 1
        pressure = (2 if obs.predicted_delay_s >= 300 or obs.predicted_overtime_s >= 180
                    or obs.priority_at_risk or obs.at_risk >= 2 else int(bool(obs.at_risk)))
        state = (context, pressure, int(3 * left > len(regular)), int(since is not None and since < 600))
    return np.array(state, dtype=np.int64), info
