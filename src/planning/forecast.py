"""Adapt departure-time forecasts to the validated integer routing model."""

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from src.ml.features import STATIC, make_features
from src.ml.model import HistoricalAverage, TrafficModel
from src.optimization.fmcg_problem import FMCGProblem
from src.optimization.fmcg_solver import solve_fmcg
from src.optimization.fmcg_validator import FMCGResult, PlanReport, validate_fmcg
from src.routing.base_provider import TravelMatrix


@dataclass(frozen=True)
class Forecasts:
    arcs: tuple[tuple[int, int], ...]
    static: np.ndarray
    historical: np.ndarray
    ml: np.ndarray

    def grid(self, base: FMCGProblem, strategy: str, hours: np.ndarray, buffer: float = 0) -> FMCGProblem:
        """Only replace durations on routable positive-distance arcs."""
        values = {"static": None, "historical": self.historical, "ml": self.ml}[strategy]
        if values is None:
            return base
        prediction = values[hours, np.arange(len(self.arcs))] + buffer * self.static
        if not (np.isfinite(prediction) & (prediction > 0)).all():
            raise ValueError("Expected finite positive predicted durations")
        grid = [list(row) for row in base.travel_s]
        for (i, j), duration in zip(self.arcs, prediction, strict=True):
            grid[i][j] = max(1, int(np.ceil(np.nextafter(duration * 60, 0))))
        return replace(base, travel_s=tuple(map(tuple, grid)))


def build_forecasts(matrix: TravelMatrix, day: date, model: TrafficModel,
                    history: HistoricalAverage) -> Forecasts:
    """Batch all hour bins from clock, calendar and qualified static geography only."""
    matrix.validate()
    if not np.array_equal(matrix.distance_km == 0, matrix.duration_min == 0):
        raise ValueError("Static distances and durations disagree on zero arcs")
    arcs = tuple((int(i), int(j)) for i, j in zip(
        *np.where((matrix.distance_km > 0) & (matrix.duration_min > 0)), strict=True))
    origins, destinations = zip(*[(matrix.ids[i], matrix.ids[j]) for i, j in arcs], strict=True)
    road = ["local" if model.zones[i] == model.zones[j] else "arterial"
            for i, j in zip(origins, destinations, strict=True)]
    static = np.array([matrix.duration_min[i, j] for i, j in arcs])
    distance = [matrix.distance_km[i, j] for i, j in arcs]
    frame = pd.DataFrame({
        "timestamp": np.repeat([f"{day}T{hour:02}:30" for hour in range(24)], len(arcs)),
        "origin_id": origins * 24, "destination_id": destinations * 24,
        "road_distance": distance * 24, STATIC: np.tile(static, 24),
        "road_characteristics": road * 24,
    })
    features = make_features(frame, model.zones)
    return Forecasts(arcs, static, history.predict(features).reshape(24, -1),
                     model.predict_features(features).reshape(24, -1))


def plan(base: FMCGProblem, forecasts: Forecasts, strategy: str, reference_hour: int,
         budget_s: float, iterations: int, buffer: float) -> tuple[FMCGResult, PlanReport | None, int]:
    """Solve single-table policies or update visited-arc hours and re-solve."""
    if strategy not in {"static", "historical", "ml", "iterative", "buffered"}:
        raise ValueError("Unknown planning strategy")
    if not 0 <= reference_hour < 24 or iterations < 1 or not np.isfinite(budget_s) or budget_s <= 0:
        raise ValueError("Invalid planning settings")
    rounds = iterations if strategy == "iterative" else 1
    source = "ml" if strategy in {"iterative", "buffered"} else strategy
    hours = np.full(len(forecasts.arcs), reference_hour, dtype=int)
    index = {arc: i for i, arc in enumerate(forecasts.arcs)}
    accepted = None
    for attempt in range(1, rounds + 1):
        problem = forecasts.grid(base, source, hours, buffer if strategy == "buffered" else 0)
        result = solve_fmcg(problem, time_limit_s=budget_s / rounds)
        if result.status not in {"feasible", "optimal"}:
            if accepted is not None:
                return (*accepted, attempt)
            continue
        report = validate_fmcg(problem, result.tours)
        accepted = result, report
        updated = np.full_like(hours, reference_hour)
        for tour in result.tours:
            previous, departure = 0, tour.departure_s
            for node, start in zip(tour.stops, tour.service_starts_s, strict=True):
                if (previous, node) in index:
                    updated[index[previous, node]] = min(23, departure // 3600)
                previous, departure = node, start + base.stops[node - 1].service_s
            if (previous, 0) in index:
                updated[index[previous, 0]] = min(23, departure // 3600)
        if np.array_equal(hours, updated):
            break
        hours = updated
    return (*accepted, attempt) if accepted is not None else (result, None, attempt)
