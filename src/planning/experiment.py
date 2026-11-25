"""Run paired routing experiments with a frozen phase-five predictor."""

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from src.ml.features import INPUTS, STATIC, TARGET, chronological_split, make_features
from src.ml.model import HistoricalAverage
from src.model_artifacts import load_traffic_model
from src.optimization.fmcg_problem import load_fmcg
from src.optimization.fmcg_validator import UNITS_PER_MAD
from src.planning.forecast import build_forecasts, plan
from src.planning.replay import actual_traffic, replay
from src.routing.matrices import cached_matrix, load_locations, matrix_path
from src.routing.osrm_client import OSRMProvider

STRATEGIES = ("static", "historical", "ml", "iterative", "buffered")


def run(profile: str, output: Path) -> None:
    config = yaml.safe_load(Path("configs/predict_then_optimize.yaml").read_text())
    settings = config["profiles"][profile]
    development = yaml.safe_load(Path("configs/development.yaml").read_text())
    manifest = json.loads(Path("results/experiments/traffic_ml/manifest.json").read_text())
    model = load_traffic_model()
    if (model.family, model.target_mode) != ("linear", "ratio"):
        raise ValueError("Expected the validated phase-five linear ratio model")
    source = Path("data/synthetic")
    locations = load_locations(source)
    provider = OSRMProvider(development["osrm_endpoint"])
    cache = Path(development["matrix_cache"])
    for name, path in {"matrix_sha256": matrix_path(provider, locations, cache),
                       "traffic_sha256": source / "traffic_observations.csv"}.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]:
            raise ValueError(f"Phase-five reference mismatch: {path}")
    matrix = cached_matrix(provider, locations, cache, cache_only=True)
    problem = load_fmcg(matrix, source)
    traffic = pd.read_csv(source / "traffic_observations.csv")
    splits = chronological_split(traffic, manifest["config"]["training_days"], manifest["config"]["validation_days"])
    train, validation, test = (splits[label] for label in ("train", "validation", "test"))
    history = HistoricalAverage.fit(make_features(train, model.zones), train[TARGET])
    residual_ratios = (validation[TARGET] - model.predict(validation.loc[:, list(INPUTS)])) / validation[STATIC]
    buffer = max(0., float(residual_ratios.quantile(config["buffer_quantile"])))
    if not isfinite(buffer):
        raise ValueError("Invalid validation-based traffic margin")
    days = pd.to_datetime(test.timestamp.str[:10].unique()).date[:settings["test_days"]]
    if len(days) != settings['test_days'] or len(days) == 0:
        raise ValueError('Profile must use a positive number of available test days')
    records, accuracy, plans = [], [], []
    for day in days:
        start = perf_counter()
        forecasts = build_forecasts(matrix, day, model, history)
        forecast_s = perf_counter() - start
        actual, incidents = actual_traffic(forecasts.arcs, forecasts.static, day, settings["seed"], matrix.ids, model.zones)
        # Common comparison grid: hours covered by ordinary delivery operations.
        hours = slice(7, 18)
        reference = config["reference_hour"]
        for key, prediction in {"static": forecasts.static, "historical_clock": forecasts.historical[hours],
                                "ml_clock": forecasts.ml[hours], "historical_fixed": forecasts.historical[reference],
                                "ml_fixed": forecasts.ml[reference],
                                "buffered_fixed": forecasts.ml[reference] + buffer * forecasts.static}.items():
            error = prediction - actual[hours]
            accuracy.append({"day": str(day), "strategy": key, "mae_min": np.abs(error).mean(),
                             "bias_min": error.mean(), "incident_mae_min": np.abs(error[incidents[hours]]).mean()})
        for strategy in STRATEGIES:
            start = perf_counter()
            result, report, calls = plan(problem, forecasts, strategy, config["reference_hour"],
                                         settings["solver_budget_s"], settings["iterations"], buffer)
            row = {"day": str(day), "strategy": strategy, "status": result.status,
                   "planned_feasible": int(report is not None), "planning_s": perf_counter() - start,
                   "calls": calls, "shared_forecast_s": forecast_s if strategy != "static" else 0.}
            if report is not None:
                row.update(planned_cost_mad=report.cost / UNITS_PER_MAD,
                           planned_otif=(len(problem.stops) - len(report.unserved)) / len(problem.stops),
                           vehicles=len(result.tours),
                           planned_distance_km=sum(r.distance_m for r in report.tours) / 1000,
                           weight_utilization_pct=np.mean([
                               100*r.loads[0].onboard[0]/problem.vehicles[t.vehicle].capacity[0]
                               for t, r in zip(result.tours, report.tours, strict=True)]) if report.tours else 0.)
                start = perf_counter()
                row.update(replay(problem, result.tours, forecasts.arcs, actual, incidents))
                row["replay_s"] = perf_counter() - start
                plans.append({"day": str(day), "strategy": strategy,
                              "tours": [{"vehicle": problem.vehicles[t.vehicle].id,
                                         "stops": [problem.stops[i-1].id for i in t.stops],
                                         "departure_s": t.departure_s, "service_starts_s": t.service_starts_s,
                                         "return_s": t.return_s} for t in result.tours]})
            records.append(row)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "day_metrics.csv", index=False)
    pd.DataFrame(accuracy).to_csv(output / "forecast_accuracy.csv", index=False)
    (output / "plans.json").write_text(json.dumps(plans, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "reference_checkpoint": "d0d3100", "profile": profile, "settings": settings,
        "reference_hour": config["reference_hour"], "buffer_quantile": config["buffer_quantile"],
        "buffer_ratio": buffer, "matrix_sha256": manifest["matrix_sha256"],
        "traffic_sha256": manifest["traffic_sha256"], "days": [str(day) for day in days]}, indent=2) + "\n")
    print(frame.groupby("strategy").mean(numeric_only=True).round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="development", choices=["development", "standard", "research"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = "" if args.profile == "standard" else f"_{args.profile}"
    output = args.output or Path(f"results/experiments/predict_then_optimize{suffix}")
    run(args.profile, output)


if __name__ == "__main__":
    main()
