"""Regression checks for Phase 12.1 input boundaries and inference integrity."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import engine, science
from src import model_artifacts
from src.benchmark import evaluation
from src.optimization.fmcg_problem import FMCGProblem, Stop, Vehicle, ZERO, integer_road_grids
from src.optimization.problem import CVRP, load_cvrp
from src.routing.base_provider import Location, TravelMatrix
from src.routing.haversine import HaversineProvider
from src.routing.matrices import cached_matrix


@pytest.mark.parametrize(("field", "index", "value"), [
    ("distance", (0, 1), -1.0), ("duration", (0, 1), -1.0),
    ("distance", (0, 1), np.nan), ("duration", (0, 1), np.inf),
    ("distance", (1, 1), np.nan), ("duration", (1, 1), 1.0),
])
def test_matrix_rejects_invalid_cells(field, index, value):
    distance = np.array([[0., 1.], [2., 0.]])
    duration = np.array([[0., 2.], [3., 0.]])
    {"distance": distance, "duration": duration}[field][index] = value
    with pytest.raises(ValueError):
        TravelMatrix(("D", "A"), distance, duration, {})


def test_matrix_contract_at_consumer_boundaries(tmp_path):
    a = np.array([[0., 1.], [2., 0.]])
    with pytest.raises(ValueError):
        TravelMatrix(("D", "D"), a, a, {})
    with pytest.raises(ValueError):
        TravelMatrix(("D", "A"), a[:, :1], a, {})
    matrix = TravelMatrix(("D", "A"), a.copy(), a.copy(), {})
    matrix.duration_min[0, 1] = 0
    for consume in (integer_road_grids, lambda m: load_cvrp(m, tmp_path / "orders.csv", 1, 10)):
        with pytest.raises(ValueError, match="Positive distance"):
            consume(matrix)
    d = ((0, 1000), (2000, 0))
    with pytest.raises(ValueError, match="nonnegative"):
        CVRP(("D", "A"), (0, 1), ((0, float("nan")), d[1]), 1, 10)
    stop = Stop("A", (1, 1, 1, 0), ZERO, (0, 60), 0)
    fleet = (Vehicle("V", (1, 1, 1, 1), (0, 60), 0, 0, 0),)
    with pytest.raises(ValueError, match="Positive distance"):
        FMCGProblem((stop,), fleet, (0, 60), d, ((0, 0), (2, 0)))


def test_colocated_and_unreachable_locations_keep_direction():
    matrix = HaversineProvider().matrix((Location("D", 33., -7.), Location("A", 33., -7.)))
    assert integer_road_grids(matrix) == (((0, 0), (0, 0)), ((0, 0), (0, 0)))
    asymmetric = TravelMatrix(("D", "A"), np.array([[0., np.nan], [2., 0.]]),
                             np.array([[0., np.nan], [3., 0.]]), {})
    assert integer_road_grids(asymmetric) == (((0, None), (2000, 0)), ((0, None), (180, 0)))


def test_invalid_cached_arc_is_not_used_offline(tmp_path):
    points = (Location("D", 33., -7.), Location("A", 33.1, -7.))
    provider = HaversineProvider()
    cached_matrix(provider, points, tmp_path)
    path, = tmp_path.glob("*.npz")
    with np.load(path, allow_pickle=False) as stored:
        contents = dict(stored)
    contents["duration_min"][0, 1] = 0
    np.savez_compressed(path, **contents)
    with pytest.raises(FileNotFoundError, match="No valid cached matrix"):
        cached_matrix(provider, points, tmp_path, cache_only=True)


@pytest.mark.parametrize("path", sorted(Path("data").glob("*/*.npz")))
def test_archived_matrix_values(path):
    with np.load(path, allow_pickle=False) as archive:
        distance, duration = archive["distance_km"], archive["duration_min"]
        ids = tuple(archive["ids"]) if "ids" in archive else tuple(map(str, range(len(distance))))
        TravelMatrix(ids, distance, duration, {})
        assert not np.any((distance > 0) & (duration <= 0))


def test_verified_traffic_load_rejects_tampering_before_deserialization(monkeypatch, tmp_path):
    model = tmp_path / model_artifacts.TRAFFIC_MODEL
    model.parent.mkdir()
    model.write_bytes(b"benign changed fixture")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_artifacts.joblib, "load", lambda _: pytest.fail("Unexpected deserialization"))
    with pytest.raises(ValueError, match="integrity"):
        model_artifacts.load_traffic_model()
    model.unlink()
    with pytest.raises(FileNotFoundError):
        model_artifacts.load_traffic_model()


def test_frozen_protocol_is_verified_and_checkpoint_cannot_escape(monkeypatch, tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "frozen_protocol.json").write_text('{"models": []}')
    monkeypatch.setattr(evaluation, "ARCHIVE", archive)
    with pytest.raises(ValueError, match="integrity"):
        evaluation.load_dispatch()
    outside = tmp_path / "elsewhere.zip"
    outside.write_bytes(b"benign test fixture")
    (archive / "linked.zip").symlink_to(outside)
    for path in (Path("linked.zip"), Path("../elsewhere.zip"), outside):
        with pytest.raises(ValueError, match="escapes"):
            model_artifacts.verified_bytes(archive, path, sha256(outside.read_bytes()).hexdigest())


def test_residual_view_handles_a_genuinely_empty_filter(monkeypatch):
    table = science.table

    def ordinary_only(experiment, name):
        data = table(experiment, name)
        return data[data.incident.eq("normal")] if name == "test_residuals" else data

    monkeypatch.setattr(science, "table", ordinary_only)
    app = AppTest.from_file(Path("dashboard/app.py").resolve(), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Travel-time prediction").run()
    app.radio[0].set_value("Incidents").run()
    assert not app.exception
    assert any(item.value.startswith("0 rows") for item in app.caption)
    assert any(item.value == "No observations match this filter." for item in app.info)


def test_missing_forecast_model_is_reported_by_scenario_view(monkeypatch):
    monkeypatch.setattr(model_artifacts, "TRAFFIC_MODEL", Path("models/absent_release_model.joblib"))
    engine.reference.cache_clear()
    try:
        app = AppTest.from_file(Path("dashboard/app.py").resolve(), default_timeout=45).run()
        app.sidebar.radio[0].set_value("Scenario configuration").run()
        app.button[0].click().run()
        assert not app.exception
        assert any("absent_release_model.joblib" in item.value for item in app.error)
        assert "prepared" not in app.session_state
    finally:
        engine.reference.cache_clear()
