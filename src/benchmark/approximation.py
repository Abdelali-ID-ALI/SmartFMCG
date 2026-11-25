"""Matched-information comparison against the frozen compact neural network."""

from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
import json

import pandas as pd
import torch

from src.benchmark.analysis import paired, summaries
from src.benchmark.evaluation import ARCHIVE, freeze_approximation, frozen_protocol, make_plan, run_day
from src.benchmark.experiment import ROOT, save
from src.benchmark.scenarios import build_draws, fingerprint, make_case, traffic_realization
from src.model_artifacts import verified_bytes
from src.planning.forecast import build_forecasts
from src.rl.dqn import DispatchDQN
from src.rl.dqn_pilot import load_config as dispatch_config
from src.rl.rewards import RewardRates
from src.simulation.experiment import digest, load_reference
from src.simulation.state import Settings

METRICS = ("objective_mad", "otif", "regular_otif", "urgent_service_rate", "replan_attempts",
           "route_changes", "solver_calls", "inference_ms_per_decision")


def compare(output: Path) -> None:
    began = perf_counter()
    setup = json.loads((output / "approximation_protocol.json").read_text())
    protocol_path = output / "frozen_protocol.json"
    if digest(protocol_path) != setup["benchmark_protocol_sha256"]:
        raise ValueError("Benchmark protocol changed since approximation contrast freeze")
    protocol = json.loads(protocol_path.read_text())
    config, profile = protocol["config"], protocol["profile"]
    torch.set_num_threads(config["threads"])
    stable = dispatch_config()
    settings, rates = Settings(**stable["dispatch"]), RewardRates(**stable["reward"])
    _, matrix, base, model, history, _, products, priority = load_reference()
    expected_checkpoints = [row for row in frozen_protocol()["models"] if row["policy"] == "dqn_compact"]
    if setup["checkpoints"] != expected_checkpoints:
        raise ValueError("Expected the qualified compact checkpoints")
    neural = {}
    for row in setup["checkpoints"]:
        checkpoint = verified_bytes(ARCHIVE, Path(row["path"]), row["path_sha256"])
        neural[row["training_seed"]] = DispatchDQN.load(BytesIO(checkpoint), device="cpu")
    plans = {tuple(row["planning_key"]): row for row in json.loads((output / "plans.json").read_text())
             if row["planner"] == "buffered"}
    baseline = pd.read_csv(output / "scenario_metrics.csv").query("block == 'benchmark' and policy in ['B9','B10']")
    results = []
    day = None
    for draw in build_draws(config, profile, settings.latency_s):
        if draw.block != "benchmark":
            continue
        if perf_counter() - began >= setup["wall_limit_s"]:
            raise TimeoutError("Matched-information replay reached the configured time bound")
        if draw.day != day:
            day = draw.day
            forecast = build_forecasts(matrix, day, model, history)
        hour, margin = protocol["reference_hour"], protocol["buffer_ratio"]
        grid, errors = traffic_realization(base, forecast, draw, matrix.ids, model.zones, hour)
        case = make_case(base, draw, config, matrix.ids, model.zones, grid, fingerprint(grid), errors)
        key = draw.planning_key()
        plan = make_plan(case.problem, case.reserves, forecast, "buffered", hour, margin,
                         config["profiles"][profile]["solver_budget_s"], plans[(day.isoformat(), *key[1:])])
        paired_rows = baseline[baseline.case.eq(draw.key)]
        if set(paired_rows.scenario_sha256) != {case.scenario_fingerprint}:
            raise RuntimeError("Matched-information scenarios differ from the benchmark")
        for seed, net in neural.items():
            identity = {**case.metadata(), "policy": "B12", "training_seed": seed, "dispatched": plan.usable}
            if plan.usable:
                action = lambda obs, info, public: int(net.predict(obs, deterministic=True)[0])
                scores, _, _ = run_day(case, plan, action, products, priority, settings, rates, stable,
                                       input_mode="compact")
                if set(paired_rows.decision_clock_sha256) != {scores["decision_clock_sha256"]}:
                    raise RuntimeError("Matched-information decision clock differs from the benchmark")
                identity.update(scores, inference_ms_per_decision=1000 * scores["inference_wall_s"] / scores["decisions"])
            results.append(identity)
    frame = pd.DataFrame(results)
    combined = pd.concat([baseline, frame], ignore_index=True)
    differences, intervals = paired(combined, setup["comparisons"], config["bootstrap_draws"],
                                    config["bootstrap_seed"] + 2, METRICS)
    for name, table in (("approximation_cases", frame), ("approximation_pairs", differences),
                        ("approximation_intervals", intervals), ("approximation_summary", summaries(
                            combined[combined.dispatched], ["policy"], METRICS))):
        table.to_csv(output / f"{name}.csv", index=False)
    save(output / "approximation_manifest.json", {"rows": len(frame), "dispatched_rows": int(frame.dispatched.sum()),
         "plans_sha256": digest(output / "plans.json"), "wall_s": perf_counter() - began,
         "peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "final_benchmark")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    (freeze_approximation if args.freeze else compare)(args.output)


if __name__ == "__main__":
    main()
