"""Independent day traffic draws and fixed-plan execution for phase six."""

from datetime import date
from math import ceil

import numpy as np

from src.optimization.fmcg_problem import FMCGProblem
from src.optimization.fmcg_validator import Tour, UNITS_PER_MAD, operating_cost


def actual_traffic(arcs: tuple[tuple[int, int], ...], static: np.ndarray, day: date, seed: int,
                   ids: tuple[str, ...], zones: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Draw one latent condition per arc-hour using an evaluation-only seed."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, day.toordinal()]))
    hours = np.arange(24)
    peak = np.isin(hours, [7, 8, 9, 16, 17, 18])
    geography = np.array([.12 * any(zones[ids[k]] in {"maarif", "bourgogne"} for k in (i, j))
                          + .08 * (zones[ids[i]] == zones[ids[j]]) for i, j in arcs])
    shape = (24, len(arcs))
    incidents = rng.random(shape) < .035
    ratio = 1.04 + peak[:, None] * (.46 if day.weekday() < 5 else .15) + geography
    ratio = ratio + .65 * incidents + rng.normal(0, .12, shape)
    return np.maximum(.8, ratio) * static, incidents


def replay(problem: FMCGProblem, tours: tuple[Tour, ...], arcs: tuple[tuple[int, int], ...],
           actual: np.ndarray, incidents: np.ndarray) -> dict[str, float]:
    """Replay independently from the planned service times, using only fixed departures/sequences."""
    lookup = {arc: k for k, arc in enumerate(arcs)}
    late = delay = distance = cost = overtime = affected = served = mandatory_on_time = 0
    planned_nodes = {node for tour in tours for node in tour.stops}
    for tour in tours:
        clock, previous, meters = tour.departure_s, 0, 0
        vehicle = problem.vehicles[tour.vehicle]
        for node in (*tour.stops, 0):
            arc = previous, node
            if arc in lookup:
                hour, position = min(23, clock // 3600), lookup[arc]
                clock += ceil(actual[hour, position] * 60)
                affected += int(incidents[hour, position])
            else:
                clock += problem.travel_s[previous][node]
            meters += problem.distance_m[previous][node]
            if node:
                stop = problem.stops[node - 1]
                clock = max(clock, stop.window[0]) + stop.service_s
                tardiness = max(0, clock - stop.window[1])
                delay += tardiness
                late += bool(tardiness)
                served += 1
                mandatory_on_time += int(not tardiness and stop.penalty_cents is None)
            previous = node
        overtime += max(0, clock - problem.operating_window(vehicle)[1])
        distance += meters
        cost += sum(operating_cost(vehicle, meters, clock - tour.departure_s))
    missed = [stop for i, stop in enumerate(problem.stops, 1) if i not in planned_nodes]
    cost += sum(stop.penalty_cents or 0 for stop in missed) * (UNITS_PER_MAD // 100)
    return {"realized_cost_mad": cost / UNITS_PER_MAD, "distance_km": distance / 1000,
            "late_stops": late, "delay_min": delay / 60, "overtime_min": overtime / 60,
            "otif": (served - late) / len(problem.stops), "served": served,
            "realized_feasible": int(not overtime and not late and all(s.penalty_cents is not None for s in missed)),
            "mandatory_otif": mandatory_on_time / sum(s.penalty_cents is None for s in problem.stops)
            if any(s.penalty_cents is None for s in problem.stops) else float("nan"),
            "incident_arcs": affected}
