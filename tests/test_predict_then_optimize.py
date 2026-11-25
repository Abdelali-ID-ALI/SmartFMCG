"""Check forecast input isolation, preserved constraints and execution accounting."""

from datetime import date

import numpy as np
import pytest

from src.optimization.fmcg_problem import FMCGProblem, Stop, Vehicle, ZERO
from src.optimization.fmcg_validator import Tour
from src.planning.forecast import Forecasts, build_forecasts, plan
from src.planning.replay import actual_traffic, replay
from src.routing.base_provider import TravelMatrix

ARCS = ((0, 1), (1, 0))
DAY = date(2026, 5, 21)


def instance():
    stop = Stop("A", (9, 0, 0, 0), ZERO, (100, 200), 20, refrigerated=True)
    vehicle = Vehicle("V", (10, 1, 1, 1), (0, 400), 1000, 200, 3600, refrigerated=True)
    return FMCGProblem((stop,), (vehicle,), (0, 400), ((0, 1000), (1000, 0)), ((0, 60), (60, 0)))


def test_forecast_inputs_have_no_actual_information():
    class Predictor:
        zones = {"D": "depot", "A": "maarif"}

        def predict_features(self, frame):
            assert len(frame) == 48
            assert not {"observed_travel_time", "congestion_indicators"} & set(frame.columns)
            return frame.static_travel_time.to_numpy() * 1.1

        predict = predict_features

    directed = np.array([[0., 1.], [2., 0.]])
    matrix = TravelMatrix(("D", "A"), directed, directed, {})
    forecasts = build_forecasts(matrix, DAY, Predictor(), Predictor())
    np.testing.assert_allclose(forecasts.ml[11], [1.1, 2.2])


def test_replacement_preserves_problem_and_validates_incumbent():
    base = instance()
    forecasts = Forecasts(ARCS, np.ones(2), np.ones((24, 2)), np.tile([1.01, 1.02], (24, 1)))
    changed = forecasts.grid(base, "ml", np.array([9, 9]))
    assert changed.travel_s == ((0, 61), (62, 0))
    assert changed.stops is base.stops and changed.vehicles is base.vehicles
    assert changed.distance_m == base.distance_m
    forecasts.ml[11] = [1.2, 1.3]
    result, report, calls = plan(base, forecasts, "iterative", 11, .45, 3, .2)
    assert result.status in {"optimal", "feasible"} and calls == 2
    assert report.tours[0].duration_s == 61 + 20 + 62
    assert report.tours[0].loads[0].onboard == (9, 0, 0, 0)
    with pytest.raises(ValueError, match="finite positive"):
        forecasts.grid(base, "ml", np.array([9, 9]), buffer=-2)


def test_actual_time_propagates_late_service_and_cost():
    problem = instance()
    tour = Tour(0, (1,), 100, (160,), 240)
    actual = np.tile([2., 1.], (24, 1))
    incidents = np.tile([True, False], (24, 1))
    result = replay(problem, (tour,), ARCS, actual, incidents)
    assert result["delay_min"] == pytest.approx(40/60)
    assert result["realized_cost_mad"] == 16
    assert result["otif"] == 0
    assert result["incident_arcs"] == 1
    assert result["realized_feasible"] == 0


def test_day_conditions_are_strategy_independent_and_reproducible():
    args = (ARCS, np.ones(2), DAY, 681, ("D", "A"), {"D": "depot", "A": "maarif"})
    a, b = actual_traffic(*args)
    c, d = actual_traffic(*args)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert (a >= .8).all()
