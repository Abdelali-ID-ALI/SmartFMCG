"""Independent scaling, frozen inference and physical-accounting checks."""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.benchmark.evaluation import load_dispatch
from src.benchmark.scenarios import Case, Draw, fingerprint
from src.optimization.fmcg_problem import ZERO, integer_road_grids
from src.optimization.fmcg_solver import solve_fmcg
from src.optimization.fmcg_validator import validate_fmcg
from src.planning.forecast import build_forecasts
from src.rl.dqn_pilot import load_config
from src.rl.rewards import RewardRates
from src.routing.base_provider import Location
from src.routing.matrices import cached_matrix
from src.routing.osrm_client import OSRMProvider
from src.scalability.evaluation import observe_dispatch, training_support
from src.scalability.experiment import CONFIG, geography
from src.scalability.geography import BlockTables, diagnostics, locations_for
from src.scalability.prediction import batched_forecasts
from src.scalability.scenarios import design_for, events_for, size_fleet, stops_for
from src.simulation.experiment import load_reference
from src.simulation.state import Scenario, Settings


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def prepare(reference, customers, seed=721, reserve=8):
    _, matrix, base, model, *_ = reference
    road = replace(matrix, ids=matrix.ids[:customers + 1],
                   distance_km=matrix.distance_km[:customers + 1, :customers + 1],
                   duration_min=matrix.duration_min[:customers + 1, :customers + 1])
    distance, travel = integer_road_grids(road)
    stops = stops_for(road.ids, base, seed, 19800)
    physical = replace(base, stops=stops, distance_m=distance, travel_s=travel)
    return size_fleet(physical, model.zones, reserve, .82, 8, .1)


def test_large_scenario_layout_preserves_distinct_ids_and_prefix():
    original = locations_for(30, geographic=True)
    road = locations_for(100, geographic=True)
    proxy = locations_for(500, geographic=False)
    assert road[:31] == proxy[:31] == original
    assert proxy == locations_for(500, geographic=False)
    assert len({(x.latitude, x.longitude) for x in road}) == 101
    assert len({x.id for x in proxy}) == 501


@pytest.mark.parametrize("customers", [7, 19, 30])
def test_constructed_fleet_feasibility_and_reproducibility(reference, customers):
    first = prepare(reference, customers)
    second = prepare(reference, customers)
    assert first == second
    assert first.problem.stops != prepare(reference, customers, seed=722).problem.stops
    assert len(first.problem.travel_s) == customers + 1
    assert len(first.problem.stops) == customers
    assert all(len(row) == customers + 1 for row in first.problem.distance_m)
    assert {node for t in first.witness for node in t.stops} == set(range(1, customers + 1))
    assert not validate_fmcg(first.effective, first.witness).unserved
    assert not validate_fmcg(first.problem, first.witness).unserved
    for stored, full, reduced in zip(first.reserves, first.problem.vehicles, first.effective.vehicles, strict=True):
        assert tuple(a + b + c for a, b, c in zip(*stored, reduced.capacity, strict=True)) == full.capacity


def test_fleet_and_first_solution_status(reference):
    small, larger = (prepare(reference, n) for n in (7, 30))
    assert len(small.problem.vehicles) < len(larger.problem.vehicles)
    first = []
    warm = solve_fmcg(small.effective, time_limit_s=.2, initial_tours=small.witness, on_incumbent=first.append)
    assert warm.status in ("feasible", "optimal")
    assert first and 0 < first[0] <= warm.elapsed_s
    assert warm.objective_units == validate_fmcg(small.effective, warm.tours).cost
    no_reefer = replace(larger.effective, vehicles=tuple(v for v in larger.effective.vehicles if not v.refrigerated))
    failure = solve_fmcg(no_reefer, time_limit_s=.05)
    assert failure.status == "infeasible" and not failure.tours


def test_batched_forecast_matches_unbatched_and_static_conversion(reference):
    config, matrix, base, model, history, *_ = reference
    day = date(2026, 8, 17)
    assert integer_road_grids(matrix) == (base.distance_m, base.travel_s)
    original = build_forecasts(matrix, day, model, history)
    planned, metrics = batched_forecasts(matrix, day, model, history, 300)
    assert planned.arcs == original.arcs
    assert metrics["prediction_rows"] == 24 * len(original.arcs)
    for name in ("static", "historical", "ml"):
        np.testing.assert_allclose(getattr(planned, name), getattr(original, name), rtol=0, atol=1e-12)
    hours = np.full(len(original.arcs), config["reference_hour"])
    assert planned.grid(base, "ml", hours, config["buffer_ratio"]) == original.grid(
        base, "ml", hours, config["buffer_ratio"])


def test_qualified_large_matrix_load_size_and_frozen_feature_consistency(reference):
    _, matrix, base, model, history, *_ = reference
    config = yaml.safe_load(Path(CONFIG).read_text())
    with pytest.raises(ValueError, match="Unknown geography"):
        geography(100, "unknown", config, matrix)
    locations, large, stats = geography(100, "osrm", config, matrix)
    assert stats["matrix_cells"] == 101**2
    assert not stats["duplicate_snaps"] and not stats["unreachable_arcs"]
    zones = {**model.zones, **{p.id: "maarif" for p in locations[31:]}}
    model = replace(model, zones=zones)
    day = date(2026, 9, 14)
    original = build_forecasts(large, day, model, history)
    batched, stats = batched_forecasts(large, day, model, history, 4096)
    assert stats["prediction_rows"] == 24 * 101 * 100
    np.testing.assert_allclose(batched.ml, original.ml, rtol=0, atol=1e-12)
    distance, travel = integer_road_grids(large)
    physical = replace(base, stops=stops_for(large.ids, base, 731, 19800), distance_m=distance, travel_s=travel)
    fleet = size_fleet(physical, zones, 8, .82, 8, .1)
    assert len(fleet.problem.stops) == 100 and len(fleet.problem.vehicles) > len(base.vehicles)
    assert not validate_fmcg(fleet.effective, fleet.witness).unserved
    assert all(len(row) == 101 for row in fleet.problem.travel_s)


def test_directed_osrm_block_cache_and_overlap_validation(tmp_path, monkeypatch):
    locations = tuple(Location(str(i), 33.6, -7.6 + i * .001) for i in range(7))
    calls = []

    def fake_request(self, service, batch, query):
        calls.append(len(batch))
        points = [{"location": [x.longitude, x.latitude], "distance": 0} for x in batch]
        ids = [int(x.id) for x in batch]
        meters = [[0 if a == b else a * 100 + b + 1 for b in ids] for a in ids]
        seconds = [[x / 5 for x in row] for row in meters]
        return {"distances": meters, "durations": seconds, "sources": points, "destinations": points,
                "data_version": "test-graph"}

    monkeypatch.setattr(OSRMProvider, "_request", fake_request)
    base = OSRMProvider("http://127.0.0.1:5300", max_locations=6)
    blocks = BlockTables(base, 3, "graph-a", tmp_path / "pieces")
    assemble = tmp_path / "merged"
    matrix = cached_matrix(blocks, locations, assemble)
    assert calls == [6, 4, 4]
    assert matrix.distance_km[5, 1] == .502
    assert matrix.distance_km[1, 5] == .106
    assert diagnostics(matrix)["duplicate_snaps"] == 0
    replay = cached_matrix(blocks, locations, assemble, cache_only=True)
    np.testing.assert_array_equal(matrix.distance_km, replay.distance_km)
    cached_matrix(blocks, locations, tmp_path / "new-merged")
    assert calls == [6, 4, 4]
    with pytest.raises(FileNotFoundError):
        cached_matrix(BlockTables(base, 3, "graph-b", tmp_path / "pieces"), locations, assemble, cache_only=True)

    def inconsistent(self, service, batch, query):
        table = fake_request(self, service, batch, query)
        if len(batch) == 4:
            table["distances"][0][1] += 1
        return table

    monkeypatch.setattr(OSRMProvider, "_request", inconsistent)
    with pytest.raises(ValueError, match="overlap disagrees"):
        BlockTables(base, 3, "graph-c", tmp_path / "pieces").matrix(locations)


def test_frozen_observation_and_metric_accounting(reference):
    instance = prepare(reference, 12)
    _, matrix, _, model, *_ = reference
    frozen_config = load_config()
    settings, rates = Settings(**frozen_config["dispatch"]), RewardRates(**frozen_config["reward"])
    checkpoints, frozen = load_dispatch()
    support = training_support(checkpoints)
    design, delays = design_for({"urgent_rate": .2, "service_event_rate": .2, "reserve_crates": 8,
                                  "traffic_factor": 1.35, "service_extra_s": 480,
                                  "urgent_window_s": 3300, "urgent_crates": [2, 6]}, 12)
    draw = Draw("unit", date(2026, 8, 1) + timedelta(days=1), 531, "test", "test", "reference", design, 8000,
                settings.latency_s)
    events = events_for(instance.problem, draw, matrix.ids[:13], model.zones, delays, 180)
    assert events == events_for(instance.problem, draw, matrix.ids[:13], model.zones, delays, 180)
    grid = (instance.problem.travel_s,) * 24
    case = Case(draw, instance.problem, instance.reserves, Scenario(events, grid), fingerprint(grid), (), "test")
    selected = {family: {2917: policies[2917]} for family, policies in frozen.items()}
    rows, trace, obs = observe_dispatch(case, instance, instance.witness, "witness", ("B4", "B5", "B9", "B10"),
                                       100, selected, support, settings, rates, frozen_config)
    for row in rows:
        assert not row["physical_violations"] and not row["unserved_mandatory"]
        assert row["regular_loss_per_customer_mad"] * 12 == pytest.approx(row["objective_mad"])
        assert 0 <= row["otif"] <= 1
        assert row["modeled_latency_s"] == row["replan_attempts"] * settings.latency_s
        subset = [r for r in trace if r["policy"] == row["policy"]]
        assert sum(r["search_items"] for r in subset) == row["candidates"]
    assert all(0 <= row["action"] <= 3 and (row["eligible"] or not row["action"]) for row in obs)
    assert any(row["policy"] == "B10" and row["ood_box"] is not None for row in obs)
    assert all(not any(np.isnan(v) for k, v in row.items() if k.startswith("feature_")) for row in obs)
    assert sum(a[2] + b[2] for a, b in instance.reserves if (a, b) != (ZERO, ZERO)) > 0
