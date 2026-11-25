"""Run reproducible scalability profiles in separate processes for peak RSS."""

from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
import json

import numpy as np
import pandas as pd
import torch
import yaml

from src.benchmark.evaluation import load_dispatch
from src.benchmark.scenarios import Case, Draw, fingerprint, traffic_realization
from src.optimization.fmcg_problem import ZERO, integer_road_grids
from src.rl.dqn_pilot import load_config as dispatch_config
from src.rl.rewards import RewardRates
from src.routing.base_provider import TravelMatrix
from src.routing.haversine import HaversineProvider
from src.simulation.experiment import digest, load_reference
from src.simulation.state import Scenario, Settings

from src.scalability.evaluation import observe_dispatch, planning_row, training_support
from src.scalability.geography import diagnostics, locations_for
from src.scalability.prediction import batched_forecasts
from src.scalability.scenarios import design_for, events_for, size_fleet, stops_for

CONFIG = Path("configs/scalability.yaml")
CACHE = Path("data/scalability")
ROOT = Path("results/experiments")


def save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=sorted, allow_nan=False) + "\n")


def peak_mib() -> float:
    return getrusage(RUSAGE_SELF).ru_maxrss / 1024


def choose_output(output: Path) -> None:
    parent = ROOT.resolve()
    target = output.resolve()
    if target in (parent, *parent.parents) or (target.is_relative_to(parent)
            and target.relative_to(parent).parts[0] != "scalability"):
        raise ValueError("Use a new destination outside prior experimental artifacts")
    output.mkdir(parents=True, exist_ok=False)


def geography(customers, source, config, reference):
    if source not in {"osrm", "haversine"}:
        raise ValueError(f"Unknown geography: {source}")
    start = perf_counter()
    locations = locations_for(customers, geographic=source == "osrm")
    location_wall = perf_counter() - start
    ids = tuple(x.id for x in locations)
    begin = perf_counter()
    if source == "osrm":
        if customers == 30:
            matrix = reference
        else:
            entry = config["geographic_cache"][customers]
            path = CACHE / entry["file"]
            if digest(path) != entry["sha256"]:
                raise ValueError("The qualified OSRM table changed")
            with np.load(path, allow_pickle=False) as data:
                matrix = TravelMatrix(tuple(data["ids"].tolist()), data["distance_km"], data["duration_min"],
                                      json.loads(data["provenance"].item()))
    else:
        matrix = HaversineProvider().matrix(locations)
    matrix_wall = perf_counter() - begin
    if matrix.ids != ids or (source == "osrm" and diagnostics(matrix)["zero_off_diagonal"]):
        raise ValueError("Matrix must match the ordered geography and have distinct access points")
    mode = ("reference_in_memory" if customers == 30 else "cache_hit") if source == "osrm" else "local_generation"
    return locations, matrix, {"location_wall_s": location_wall, "matrix_wall_s": matrix_wall,
                               "matrix_mode": mode,
                               "matrix_peak_process_mib": peak_mib(), **diagnostics(matrix)}


def run(profile: str, customers: int, output: Path):
    began = perf_counter()
    config = yaml.safe_load(CONFIG.read_text())
    scope = config["profiles"][profile]
    if customers not in scope["customers"]:
        raise ValueError("Customer count outside the selected profile")
    choose_output(output)
    torch.set_num_threads(config["threads"])
    ref, reference, base, model, history, *_ = load_reference()
    dispatch = dispatch_config()
    settings, rates = Settings(**dispatch["dispatch"]), RewardRates(**dispatch["reward"])
    checkpoints, frozen = load_dispatch() if scope["policies"] else ({}, {})
    support = training_support(checkpoints) if frozen else {}
    save(output / "protocol.json", {"checkpoint": "940b60e", "profile": profile, "customers": customers,
         "config_sha256": digest(CONFIG), "config": config, "ml_sha256": digest(Path("models/traffic_model.joblib")),
         "reference_osrm_sha256": ref["matrix_sha256"], "frozen_checkpoints": checkpoints,
         "benchmark_settings": asdict(settings), "rates": asdict(rates),
         "reference_hour": ref["reference_hour"], "buffer_ratio": ref["buffer_ratio"]})
    locations, matrix, geo = geography(customers, scope["geography"], config, reference)
    zone_begin = perf_counter()
    anchors = locations[:len(reference.ids)]
    zones = {p.id: model.zones[min(anchors[1:], key=lambda a:
              (a.latitude - p.latitude)**2 + ((a.longitude - p.longitude) * .83)**2).id]
             for p in locations[len(anchors):]}
    model = replace(model, zones={**model.zones, **zones})
    geo.update(zone_wall_s=perf_counter() - zone_begin)
    common = {"profile": profile, "customers": customers, "geography": scope["geography"]}
    save(output / "matrix.json", {**common, **geo, "provenance": matrix.provenance,
                                 "locations": [{**asdict(point), "zone": model.zones[point.id]}
                                               for point in locations]})
    plans, dispatch_rows, cases, forecasts_log, traces, observations = [], [], [], [], [], []
    tables = {"planning": plans, "dispatch": dispatch_rows, "prediction": forecasts_log,
              "decisions": traces, "observations": observations}

    def save_results():
        for name, rows in tables.items():
            if rows:
                pd.DataFrame(rows).to_csv(output / (f"{name}.csv.gz" if name == "observations" else f"{name}.csv"),
                                          index=False, compression={"method": "gzip", "mtime": 0}
                                          if name == "observations" else None)
        save(output / "cases.json", cases)

    status = "complete"
    try:
        for day_index, raw in enumerate(scope["dates"]):
            day = date.fromisoformat(raw)
            start = perf_counter()
            forecasts, prediction = batched_forecasts(matrix, day, model, history, config["prediction_batch_arcs"])
            prediction_record = {**common, "day": raw, **prediction, "prediction_process_peak_mib": peak_mib()}
            forecasts_log.append(prediction_record)
            if customers in scope.get("workload_only_sizes", []):
                prediction_record["scope"] = "matrix_and_prediction_only"
                continue
            prediction_record["scope"] = "planning_and_prediction"
            distance, travel = integer_road_grids(matrix)
            prediction_record["integer_matrix_wall_s"] = perf_counter() - start - prediction["forecast_wall_s"]
            for regime in scope["regimes"]:
                if perf_counter() - began >= scope["max_wall_s"]:
                    status = "incomplete_wall_limit"
                    return
                seed = int(np.random.SeedSequence([scope["namespace"], day.toordinal()]).generate_state(1)[0])
                case_id = f"{profile}-{customers}-{raw}-{regime}"
                meta = {**common, "case": case_id, "day": raw, "seed": seed, "regime": regime}
                design, delays = design_for(config["regimes"][regime], customers)
                start = perf_counter()
                stops = stops_for(matrix.ids, base, seed, config["fleet"]["window_s"])
                physical = replace(base, stops=stops, distance_m=distance, travel_s=travel)
                stops_s = perf_counter() - start
                start = perf_counter()
                buffered = forecasts.grid(physical, "ml", np.full(len(forecasts.arcs), ref["reference_hour"]),
                                          ref["buffer_ratio"])
                margin_s = perf_counter() - start
                start = perf_counter()
                fleet_config = {k: v for k, v in config["fleet"].items() if k != "window_s"}
                instance = size_fleet(buffered, model.zones, design.reserve, **fleet_config)
                sizing_s = perf_counter() - start
                meta.update(fleet=len(instance.problem.vehicles), refrigerated=sum(v.refrigerated
                            for v in instance.problem.vehicles), stops_wall_s=stops_s, safety_matrix_wall_s=margin_s,
                            fleet_sizing_wall_s=sizing_s)
                comparison = (("witness", None),)
                if customers not in scope.get("skip_greedy_sizes", []):
                    comparison += (("greedy", None),)
                comparison += tuple((mode, budget) for budget in scope["budgets_s"]
                                    for mode in ("ortools_cold", "ortools_warm"))
                selected, best = (), float("inf")
                source = ""
                saved_plans = []
                for mode, budget in comparison:
                    row, tours = planning_row(instance, mode, instance.effective, budget)
                    plans.append({**meta, **row, "planning_process_peak_mib": peak_mib()})
                    saved_plans.append({"planner": mode, "budget_s": budget, "status": row["status"],
                                        "tours": [asdict(t) for t in tours]})
                    if row["usable"] and row["cost_mad"] < best:
                        selected, best = tours, row["cost_mad"]
                        source = f"{mode}/{budget}" if budget is not None else mode
                if profile == "standard" and day_index == 0 and regime == "stress":
                    warm = tuple(v for v in instance.effective.vehicles if not v.refrigerated)
                    if warm:
                        incompatible = replace(instance.effective, vehicles=warm)
                        row, _ = planning_row(instance, "ortools_cold", incompatible, scope["budgets_s"][0])
                        if row["status"] != "infeasible" or not any(s.refrigerated for s in incompatible.stops):
                            raise ValueError("Cold-chain impossibility witness was not recognized")
                        plans.append({**meta, "case": case_id + "-no-reefer", "regime": "no_reefer",
                                      "fleet": len(warm), "refrigerated": 0, **row,
                                      "planning_process_peak_mib": peak_mib()})
                case_row = {**meta, "physical_stops": [asdict(s) for s in instance.problem.stops],
                            "physical_fleet": [asdict(v) for v in instance.problem.vehicles],
                            "depot_window": instance.problem.depot_window, "reserves": instance.reserves,
                            "selected_plan": source, "plans": saved_plans,
                            "delivery": [sum(s.delivery[k] for s in instance.problem.stops) for k in range(4)],
                            "pickup": [sum(s.pickup[k] for s in instance.problem.stops) for k in range(4)]}
                cases.append(case_row)
                if not scope["policies"]:
                    continue
                start = perf_counter()
                realized, errors = traffic_realization(physical, forecasts, Draw(
                    case_id, day, seed, regime, profile, "reference", design, config["urgent_penalty_cents"],
                    settings.latency_s), matrix.ids, model.zones, ref["reference_hour"])
                traffic_hash = fingerprint(realized)
                case_row.update(realization_wall_s=perf_counter() - start, traffic_sha256=traffic_hash,
                                synthetic_traffic_errors=errors)
                draw = Draw(case_id, day, seed, regime, profile, "reference", design,
                            config["urgent_penalty_cents"], settings.latency_s)
                events = events_for(instance.problem, draw, matrix.ids, model.zones, delays, config["urgent_service_s"])
                case_row.update(events=[asdict(event) for event in events], event_count=len(events))
                signature = fingerprint({"events": case_row["events"], "traffic": traffic_hash,
                                         "stops": case_row["physical_stops"], "fleet": case_row["physical_fleet"],
                                         "reserves": instance.reserves})
                case_row["scenario_sha256"] = signature
                case = Case(draw, instance.problem, instance.reserves, Scenario(events, realized),
                            traffic_hash, errors, signature)
                runs = [(config["reference_candidates"], scope["policies"])]
                if profile == "standard" and customers in config["sweep_sizes"] and regime == "stress":
                    runs.extend((budget, config["sweep_policies"]) for budget in config["candidate_budgets"]
                                if budget != config["reference_candidates"])
                for budget, labels in runs:
                    frame, trace, obs = observe_dispatch(case, instance, selected, source, labels, budget, frozen,
                                                         support, settings, rates, dispatch)
                    run_meta = {**meta, "traffic_sha256": traffic_hash, "scenario_sha256": signature,
                                "event_count": len(events), "realization_wall_s": case_row["realization_wall_s"]}
                    dispatch_rows.extend({**run_meta, **r} for r in frame)
                    traces.extend({**meta, **r} for r in trace)
                    observations.extend({**meta, **r} for r in obs)
                case_results = [row for row in dispatch_rows if row["case"] == case_id]
                for column in ("plan_sha256", "scenario_sha256", "traffic_sha256", "decision_clock_sha256"):
                    if len({row[column] for row in case_results}) != 1:
                        raise ValueError(f"Unpaired scenario column {column}: {case_id}")
                if any(r["physical_violations"] or r["unserved_mandatory"] for r in case_results):
                    raise ValueError(f"Physical validation or mandatory service failed: {case_id}")
                if profile == "standard" and customers in config["reserve_ablation_sizes"] and regime == "stress":
                    # Remove stocked reserves after planning to hold the tours, fleet and events fixed.
                    empty = replace(instance, reserves=((ZERO, ZERO),) * len(instance.reserves))
                    budget = config["reference_candidates"]
                    frame, trace, obs = observe_dispatch(case, empty, selected, source, ("B4", "B5", "B8"),
                                                         budget, frozen, support, settings, rates, dispatch)
                    if any(r["physical_violations"] or r["unserved_mandatory"] for r in frame):
                        raise ValueError(f"Physical validation failed in reserve ablation: {case_id}")
                    ablation = {**run_meta, "scenario_sha256": fingerprint([signature, "zero_reserve"]),
                                "ablation": "zero_reserve"}
                    dispatch_rows.extend({**ablation, **r} for r in frame)
                    traces.extend({**ablation, **r} for r in trace)
                    observations.extend({**ablation, **r} for r in obs)
                print(f"{case_id} | {len(instance.problem.vehicles)} vehicles | {source} | {len(case_results)} replays",
                      flush=True)
                save_results()
    except Exception:
        status = "failed"
        raise
    finally:
        save_results()
        save(output / "resources.json", {**common, "status": status, "process_peak_mib": peak_mib(),
                                         "wall_s": perf_counter() - began, "cases": len(cases),
                                         "replays": len(dispatch_rows)})


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("development", "standard", "proxy"), required=True)
    parser.add_argument("--customers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.profile, args.customers, args.output)


if __name__ == "__main__":
    main()
