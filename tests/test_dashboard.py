"""Phase 11 integration checks against the committed CPU artifacts."""

from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import engine, maps, operations, science
from dashboard.app import PAGES
from src.optimization.fmcg_validator import validate_fmcg

ENTRY = Path(__file__).resolve().parents[1] / "dashboard/app.py"


@pytest.fixture(scope="module")
def demo():
    return engine.prepare(engine.Inputs())


@pytest.fixture(scope="module")
def robust(demo):
    return engine.plan(demo, "Robust ML")


@pytest.mark.parametrize("changes", [
    {"customers": 250}, {"max_vehicles": 17}, {"seed": -1}, {"demand": float("nan")},
    {"traffic": float("inf")}, {"reserve": -1}, {"templates": ()}, {"templates": ("V99",)},
    {"urgent_crates": (9, 3)}, {"urgents": 11},
])
def test_scenario_input_validation(changes):
    with pytest.raises(ValueError):
        replace(engine.Inputs(), **changes)


def test_layout_cache_is_correct_and_read_only():
    points, small = engine.layout(12)
    _, large = engine.layout(30)
    assert engine.layout(12)[1] is small
    assert len(points) == 13
    assert small.ids == large.ids[:13]
    assert (small.distance_km == large.distance_km[:13, :13]).all()
    with pytest.raises(ValueError, match="read-only"):
        small.distance_km[0, 1] = 0
    leg = list(maps.pieces(engine.layout(30)[0], (0, 30, 0)))
    assert [road for _, road in leg] == [True, False]


def test_physically_inconsistent_fleet_rejected():
    with pytest.raises(ValueError, match="requires|require|needs|constructive"):
        engine.prepare(replace(engine.Inputs(), max_vehicles=1))
    with pytest.raises(ValueError, match="refrigerated"):
        engine.prepare(replace(engine.Inputs(), customers=30, templates=("V01", "V02")))


@pytest.mark.parametrize("strategy", engine.PLANNERS)
def test_planning_is_independently_validated(demo, strategy):
    result = engine.plan(demo, strategy)
    assert not result.report.unserved
    effective = engine.planning_capacity(result.problem, demo.reserves)
    assert validate_fmcg(effective, result.tours) == result.report
    assert validate_fmcg(result.problem, result.tours).cost == result.report.cost


@pytest.mark.parametrize("policy", engine.POLICIES)
def test_policy_inference_and_replay(demo, robust, policy):
    run = engine.replay(demo, robust, policy)
    assert run.metrics["unserved_mandatory"] == 0
    assert run.metrics["vehicles"] == len(robust.tours)
    assert run.metrics["urgent_total"] == demo.inputs.urgents
    assert run.metrics["replan_attempts"] >= run.metrics["replans_applied"]
    assert run.frames[-1].loss_mad == pytest.approx(run.metrics["objective_mad"])
    assert run.frames[-1].events == demo.scenario.events
    assert all(a.time_s <= b.time_s and set(a.completed) <= set(b.completed)
               for a, b in zip(run.frames, run.frames[1:]))
    assert all(event.at_s <= frame.time_s for frame in run.frames for event in frame.events)
    assert all(record["time_s"] <= frame.time_s for frame in run.frames for record in frame.decisions)
    if policy == "B4":
        assert run.metrics["replan_attempts"] == 0
    if policy in {"B9", "B10", "B12"}:
        assert run.training_seed == 2917


def open_page(page):
    app = AppTest.from_file(ENTRY, default_timeout=45).run()
    if page != "Overview":
        app.sidebar.radio[0].set_value(page).run()
    return app


@pytest.mark.parametrize("page", PAGES)
def test_app_starts_and_pages_render(page):
    app = open_page(page)
    assert not app.exception
    assert not app.error


def test_scenario_form_applies_preset_and_keeps_previous_valid_state():
    app = open_page("Scenario configuration")
    app.selectbox(key="preset").set_value("stress").run()
    assert app.slider(key="traffic_input").value == 1.85
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.session_state["scenario_inputs"].regime == "stress"
    app.number_input[1].set_value(1)
    app.button[0].click().run()
    assert not app.exception
    assert "needs" in app.error[0].value
    assert app.session_state["scenario_inputs"].max_vehicles == 16


def test_paired_ui_cache_and_isolated_sessions(monkeypatch, demo):
    monkeypatch.setattr(engine, "prepare", lambda _: demo)
    replay = engine.replay
    calls = []

    def counted(case, initial, policy, seed):
        calls.append((case, initial, policy))
        return replay(case, initial, policy, seed)

    monkeypatch.setattr(engine, "replay", counted)
    first = open_page("Dispatch replay")
    second = open_page("Dispatch replay")
    first.button[0].click().run()
    assert not first.exception and not first.error
    assert calls[0][:2] == calls[1][:2]
    assert [item[2] for item in calls] == ["B4", "B5"]
    assert "runs" not in second.session_state
    first.button[2].click().run()
    assert first.slider[0].value == 1
    first.slider[0].set_value(first.slider[0].max).run()
    assert not first.exception
    assert "Loss savings vs B4" in [m.label for m in first.metric]
    a = first.session_state["runs"]
    assert operations.loss_savings(a["B4", None], a["B5", None]) == pytest.approx(
        a["B4", None].metrics["objective_mad"] - a["B5", None].metrics["objective_mad"])
    first.button[0].click().run()
    assert len(calls) == 2
    first.button[4].click().run()
    assert "active_run" not in first.session_state


def test_route_ui_shows_plan_and_offline_map():
    app = open_page("Route planning")
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert "Robust ML" in app.session_state["plans"]
    app.toggle[0].set_value(True).run()
    assert not app.exception and app.get("plotly_chart")


def test_missing_archives_are_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(science, "ARCHIVE", tmp_path)
    archive = open_page("Travel-time prediction")
    assert not archive.exception
    assert "test_metrics.csv" in archive.error[0].value

    def missing(_):
        raise FileNotFoundError(2, "Missing archive", "/private/path/checkpoint.zip")

    monkeypatch.setattr(engine, "prepare", missing)
    replay = open_page("Dispatch replay")
    replay.button[0].click().run()
    assert not replay.exception and "checkpoint.zip" in replay.error[0].value
    assert "/private/path" not in replay.error[0].value
    assert "active_run" not in replay.session_state
