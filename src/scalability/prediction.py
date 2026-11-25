"""Bound temporary inference arrays without changing the frozen feature contract."""

from datetime import date
from time import perf_counter

import numpy as np
import pandas as pd

from src.ml.features import STATIC, make_features
from src.ml.model import HistoricalAverage, TrafficModel
from src.planning.forecast import Forecasts
from src.routing.base_provider import TravelMatrix


def batched_forecasts(matrix: TravelMatrix, day: date, model: TrafficModel, history: HistoricalAverage,
                      batch_arcs: int) -> tuple[Forecasts, dict]:
    matrix.validate()
    if batch_arcs < 1 or not np.array_equal(matrix.distance_km == 0, matrix.duration_min == 0):
        raise ValueError("Invalid prediction batch size or inconsistent zero arcs")
    started = perf_counter()
    origins, destinations = np.where((matrix.distance_km > 0) & (matrix.duration_min > 0))
    indices = tuple(zip(origins.tolist(), destinations.tolist(), strict=True))
    if not indices:
        raise ValueError("No positive routable arcs")
    ids = np.array(matrix.ids)
    zone = np.array([model.zones[key] for key in matrix.ids])
    static = matrix.duration_min[origins, destinations]
    distance = matrix.distance_km[origins, destinations]
    roads = np.where(zone[origins] == zone[destinations], "local", "arterial")
    ml = np.empty((24, len(indices)))
    historical = np.empty_like(ml)
    setup_s = perf_counter() - started
    feature_s = inference_s = history_s = 0.
    for hour in range(24):
        for first in range(0, len(indices), batch_arcs):
            span = slice(first, first + batch_arcs)
            began = perf_counter()
            frame = pd.DataFrame({
                "timestamp": f"{day}T{hour:02}:30", "origin_id": ids[origins[span]],
                "destination_id": ids[destinations[span]], "road_distance": distance[span],
                STATIC: static[span], "road_characteristics": roads[span],
            })
            features = make_features(frame, model.zones)
            feature_s += perf_counter() - began
            began = perf_counter()
            ml[hour, span] = model.predict_features(features)
            inference_s += perf_counter() - began
            began = perf_counter()
            historical[hour, span] = history.predict(features)
            history_s += perf_counter() - began
    return Forecasts(indices, static, historical, ml), {
        "prediction_rows": int(ml.size), "feature_wall_s": feature_s, "prediction_wall_s": inference_s,
        "historical_wall_s": history_s, "prediction_setup_wall_s": setup_s,
        "forecast_wall_s": perf_counter() - started, "forecast_array_bytes": ml.nbytes + historical.nbytes + static.nbytes,
    }
