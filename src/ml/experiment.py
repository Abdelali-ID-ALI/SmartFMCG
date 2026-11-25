"""Compare supervised traffic models offline and persist the selected pipeline."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from src.data.validation import SCHEMAS, validate_tables, validate_traffic_reference
from src.ml.features import CATEGORICAL, INPUTS, NUMERIC, STATIC, TARGET, chronological_split, make_features
from src.ml.model import HistoricalAverage, fit_model
from src.routing.matrices import cached_matrix, load_locations, matrix_path
from src.routing.osrm_client import OSRMProvider


def candidates(seed: int, threads: int) -> dict:
    """Use two modest configurations per tree family and one linear fit per target."""
    return {
        "linear": [("ols", LinearRegression())],
        "random_forest": [(f"depth_{depth}", RandomForestRegressor(
            n_estimators=100, max_depth=depth, min_samples_leaf=5, random_state=seed, n_jobs=threads))
            for depth in (5, 10)],
        "xgboost": [(f"depth_{depth}", XGBRegressor(
            n_estimators=120, max_depth=depth, learning_rate=.08, random_state=seed,
            tree_method="hist", device="cpu", n_jobs=threads)) for depth in (2, 4)],
        "lightgbm": [(f"leaves_{leaves}", LGBMRegressor(
            n_estimators=120, num_leaves=leaves, learning_rate=.08, random_state=seed,
            verbosity=-1, n_jobs=threads, deterministic=True, force_col_wise=True)) for leaves in (7, 15)],
    }


def metrics(observed, predicted) -> dict[str, float]:
    return {"mae": mean_absolute_error(observed, predicted),
            "rmse": root_mean_squared_error(observed, predicted),
            "r2": r2_score(observed, predicted) if len(observed) > 1 else float("nan"),
            "bias": float(np.mean(np.asarray(predicted) - np.asarray(observed)))}


def evaluate(predict, frame: pd.DataFrame, observed: pd.Series) -> dict[str, float]:
    predicted = predict(frame)
    runtimes = []
    for _ in range(5):
        start = perf_counter()
        predict(frame)
        runtimes.append(perf_counter() - start)
    return {**metrics(observed, predicted), "batch_ms": 1000 * float(np.median(runtimes)),
            "rows": len(frame)}


def run(root: Path, output: Path, model_path: Path) -> None:
    config = yaml.safe_load((root / "configs/traffic_ml.yaml").read_text())
    development = yaml.safe_load((root / "configs/development.yaml").read_text())
    source = root / "data/synthetic"
    tables = {name: pd.read_csv(source / f"{name}.csv") for name in SCHEMAS}
    validate_tables(tables)
    locations = load_locations(source)
    provider = OSRMProvider(development["osrm_endpoint"])
    cache = root / development["matrix_cache"]
    matrix = cached_matrix(provider, locations, cache, cache_only=True)
    traffic = tables["traffic_observations"]
    validate_traffic_reference(traffic, matrix,
                               (*tables["depots"].to_dict("records"), *tables["customers"].to_dict("records")))
    zones = dict(zip(tables["customers"].customer_id, tables["customers"].geographical_zone, strict=True))
    zones.update(dict.fromkeys(tables["depots"].depot_id, "depot"))
    splits = chronological_split(traffic, config["training_days"], config["validation_days"])
    engineered = {label: make_features(frame, zones) for label, frame in splits.items()}
    train, val, test = (splits[stage] for stage in ("train", "validation", "test"))
    fitted, trials, finalists = {}, [], []
    for mode in ("absolute", "ratio"):
        for family, alternatives in candidates(config["seed"], config["threads"]).items():
            group = []
            for settings, estimator in alternatives:
                key = f"{family}/{mode}/{settings}"
                start = perf_counter()
                model = fit_model(engineered["train"], train[TARGET], estimator, zones, mode, family)
                elapsed = perf_counter() - start
                scores = metrics(val[TARGET], model.predict_features(engineered["validation"]))
                fitted[key] = model
                record = {"model": key, "family": family, "target": mode, "training_s": elapsed, **scores}
                group.append(record)
                trials.append(record)
            finalists.append(min(group, key=lambda row: (row["mae"], row["rmse"], row["model"]))["model"])
    history = HistoricalAverage.fit(engineered["train"], train[TARGET])
    predictors = {"static_osrm": lambda frame: frame[STATIC].to_numpy(),
                  "historical_average": lambda frame: history.predict(make_features(frame, zones)),
                  **{key: fitted[key].predict for key in finalists}}
    # Freeze the choice before inspecting test scores.
    validation = [{"model": key, **metrics(val[TARGET], predictor(val.loc[:, list(INPUTS)]))}
                  for key, predictor in predictors.items()]
    best = min((row for row in validation if row["model"] in fitted), key=lambda row: (row["mae"], row["rmse"]))
    selected = fitted[best["model"]]
    raw_test = test.loc[:, list(INPUTS)]
    test_scores = [{"model": key, **evaluate(predict, raw_test, test[TARGET])}
                   for key, predict in predictors.items()]
    predicted = selected.predict(raw_test)
    evaluation = pd.DataFrame({
        "timestamp": test.timestamp, "hour": engineered["test"].hour,
        "incident": test.congestion_indicators, "actual_simulated_min": test[TARGET],
        "predicted_min": predicted, "static_osrm_min": test[STATIC],
    })
    evaluation["residual_min"] = evaluation.predicted_min - evaluation.actual_simulated_min
    evaluation["intensity"] = pd.cut(test[TARGET] / test[STATIC], [-np.inf, 1.2, 1.5, np.inf],
                                     labels=["low", "medium", "high"])
    evaluation["distance"] = pd.cut(test.road_distance, [0, 3, 8, np.inf], labels=["0–3", "3–8", ">8"])
    strata = []
    for field in ("intensity", "hour", "distance", "incident"):
        for value, rows in evaluation.groupby(field, observed=True):
            strata.append({"dimension": field, "group": str(value), "n": len(rows),
                           **metrics(rows.actual_simulated_min, rows.predicted_min)})

    # Controlled diagnosis: refit using the already selected model's configuration and target.
    from sklearn.base import clone
    ablations = []
    for label, columns in {
        "static_and_distance_only": ("road_distance", STATIC),
        "without_generator_flags": tuple(col for col in NUMERIC + CATEGORICAL
                                          if col not in {"rush", "weekday_rush", "central_zone"}),
    }.items():
        model = fit_model(engineered["train"], train[TARGET], clone(selected.pipeline["regressor"]),
                          zones, selected.target_mode, selected.family, columns)
        for stage in ("validation", "test"):
            ablations.append({"model": label, "partition": stage,
                              **metrics(splits[stage][TARGET], model.predict_features(engineered[stage]))})

    def score(_, frame, target):
        return -mean_absolute_error(target, selected.predict_features(frame))

    importance = permutation_importance(selected.pipeline, engineered["validation"], val[TARGET],
                                        scoring=score, n_repeats=4, random_state=config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, model_path, compress=3)
    if not np.array_equal(predicted, joblib.load(model_path).predict(raw_test)):
        raise RuntimeError("Reloaded model predictions changed")
    for name, records in {"validation_trials": trials, "validation_finalists": validation,
                          "test_metrics": test_scores, "test_strata": strata, "ablations": ablations,
                          "permutation_importance": {"feature": engineered["validation"].columns,
                              "mae_increase": importance.importances_mean,
                              "repeat_sd": importance.importances_std}}.items():
        pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)
    evaluation.to_csv(output / "test_residuals.csv", index=False)
    manifest = {"selected": best["model"], "config": config, "reference_checkpoint": "108d2fd",
                "matrix_sha256": hashlib.sha256(matrix_path(provider, locations, cache).read_bytes()).hexdigest(),
                "traffic_sha256": hashlib.sha256((source / "traffic_observations.csv").read_bytes()).hexdigest(),
                "partitions": {name: {"start": frame.timestamp.min()[:10], "end": frame.timestamp.max()[:10],
                                      "rows": len(frame), "days": frame.timestamp.str[:10].nunique()}
                               for name, frame in splits.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest["selected"])
    print(pd.DataFrame(test_scores).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/experiments/traffic_ml"))
    parser.add_argument("--model", type=Path, default=Path("models/traffic_model.joblib"))
    args = parser.parse_args()
    run(Path.cwd(), args.output, args.model)


if __name__ == "__main__":
    main()
