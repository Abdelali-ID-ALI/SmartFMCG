"""Complete-case reporting, paired date/seed bootstrap and robustness summaries."""

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd

CONTRASTS = (
    ("B1", "B0", "optimization"), ("B2", "B1", "historical"), ("B3", "B2", "machine_learning"),
    ("B4", "B3", "safety_margin"), ("B5", "B4", "event_driven"), ("B6", "B4", "periodic"),
    ("B7", "B4", "threshold"), ("B8", "B4", "multicriteria"), ("B9", "B4", "tabular_adaptation"),
    ("B10", "B4", "neural_adaptation"), ("B11", "B4", "random_adaptation"),
    ("B8", "B5", "multicriteria_vs_event"), ("B9", "B8", "tabular_vs_multicriteria"),
    ("B10", "B8", "neural_vs_multicriteria"), ("B9", "B5", "tabular_vs_event"),
    ("B10", "B5", "neural_vs_event"), ("B10", "B9", "neural_vs_tabular"),
)
METRICS = (
    "otif", "regular_otif", "urgent_service_rate", "mandatory_completion_rate", "distance_km",
    "realized_travel_min", "initial_weight_utilization", "vehicles", "operating_cost_mad",
    "realized_cost_mad", "objective_mad", "delay_min", "replan_attempts", "replans_applied",
    "route_changes", "route_instability", "solver_calls", "total_solver_calls", "invalid_action_frequency",
    "safety_fallback_frequency", "physical_violations", "initial_planning_wall_s", "forecast_wall_s",
    "planning_wall_s", "inference_wall_s", "inference_ms_per_decision", "end_to_end_wall_s",
    "physical_rss_sample_peak_mib", "packaging_recovery", "recovered_crates",
)
SENSITIVITY_METRICS = ("objective_mad", "otif", "urgent_service_rate", "replan_attempts", "route_changes",
                       "operating_cost_mad", "regular_otif")
CASE_KEYS = ["case", "day", "regime"]


def complete_cases(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.case.isin(frame.groupby("case").dispatched.all().loc[lambda x: x].index)]


def summaries(frame: pd.DataFrame, groups: list[str], metrics=METRICS) -> pd.DataFrame:
    """Average seeds inside each scenario before measuring between-scenario dispersion."""
    values = frame.groupby(list(dict.fromkeys([*groups, *CASE_KEYS])), dropna=False)[list(metrics)].mean().reset_index()
    rows = []
    for keys, group in values.groupby(groups, dropna=False):
        key = dict(zip(groups, keys, strict=True))
        for metric in metrics:
            valid = group[metric].dropna()
            rows.append({**key, "metric": metric, "cases": len(valid),
                         "date_blocks": group.loc[valid.index, "day"].nunique(),
                         "mean": valid.mean(), "sd": valid.std(), "median": valid.median(),
                         "p10": valid.quantile(.1), "p90": valid.quantile(.9)})
    return pd.DataFrame(rows)


def policy_matrix(frame: pd.DataFrame, policy: str, metrics) -> pd.DataFrame:
    subset = frame[frame.policy.eq(policy) & frame.dispatched].copy()
    if subset.empty:
        return pd.DataFrame()
    subset.training_seed = subset.training_seed.fillna(-1).astype(int)
    return subset.pivot(index=CASE_KEYS, columns="training_seed", values=list(metrics))


def paired(frame: pd.DataFrame, contrasts, draws: int, seed: int, metrics=METRICS):
    """Draw shared dates and matched learned seeds, retaining all cases in each date."""
    comparisons, observations = [], []
    rng = np.random.default_rng(seed)
    for regime in ("all", *sorted(frame.regime.unique())):
        block = frame if regime == "all" else frame[frame.regime.eq(regime)]
        matrices = {policy: policy_matrix(block, policy, metrics) for policy in {key for l, r, _ in contrasts for key in (l, r)}}
        for left, right, contrast in contrasts:
            if matrices[left].empty or matrices[right].empty:
                continue
            for metric in metrics:
                lhs, rhs = matrices[left][metric], matrices[right][metric]
                common = lhs.dropna().index.intersection(rhs.dropna().index)
                if common.empty:
                    continue
                a, b = lhs.loc[common], rhs.loc[common]
                av, bv = a.mean(axis=1), b.mean(axis=1)
                difference = av - bv
                relative = difference / bv.replace(0., np.nan) * 100
                sample = pd.DataFrame({"left_value": av, "right_value": bv, "difference": difference,
                                       "relative_pct": relative}).reset_index()
                if regime == "all":
                    observations.extend(sample.assign(left=left, right=right, contrast=contrast, metric=metric).to_dict("records"))
                days = sorted(a.index.get_level_values("day").unique())
                dates = rng.integers(len(days), size=(draws, len(days)))
                counts = a.groupby(level="day").size().loc[days].to_numpy()
                a_sum = a.groupby(level="day").sum().loc[days].to_numpy()[dates].sum(axis=1)
                b_sum = b.groupby(level="day").sum().loc[days].to_numpy()[dates].sum(axis=1)
                if a.shape[1] > 1:
                    a_idx = rng.integers(a.shape[1], size=(draws, a.shape[1]))
                    a_draw = np.take_along_axis(a_sum, a_idx, axis=1).mean(axis=1)
                else:
                    a_draw = a_sum[:, 0]
                if b.shape[1] > 1:
                    b_idx = a_idx if a.columns.equals(b.columns) else rng.integers(b.shape[1], size=(draws, b.shape[1]))
                    b_draw = np.take_along_axis(b_sum, b_idx, axis=1).mean(axis=1)
                else:
                    b_draw = b_sum[:, 0]
                limits = np.quantile((a_draw - b_draw) / counts[dates].sum(axis=1), [.025, .975])
                comparisons.append({"left": left, "right": right, "contrast": contrast, "regime": regime,
                                    "metric": metric, "cases": len(a), "date_blocks": len(days),
                                    "difference": difference.mean(), "sd_difference": difference.std(),
                                    "median_difference": difference.median(), "relative_pct": relative.mean(),
                                    "relative_cases": relative.count(), "ci_low": limits[0], "ci_high": limits[1]})
    return pd.DataFrame(observations), pd.DataFrame(comparisons)


def sensitivity(frame: pd.DataFrame, draws: int, seed: int):
    individual, aggregate = [], []
    for name, group in frame.groupby("policy"):
        data = group.assign(policy=group.variant, case=group.day)
        contrasts = [(variant, "reference", variant) for variant in group.variant.unique() if variant != "reference"]
        effects, intervals = paired(data, contrasts, draws, seed, SENSITIVITY_METRICS)
        individual.append(effects.assign(policy=name))
        aggregate.append(intervals[intervals.regime.eq("all")].assign(policy=name))
    ranks = summaries(complete_cases(frame), ["variant", "policy"], SENSITIVITY_METRICS)
    ranks["rank"] = ranks.groupby(["variant", "metric"])["mean"].rank(method="min", ascending=True)
    maximize = ranks.metric.isin({"otif", "regular_otif", "urgent_service_rate"})
    ranks.loc[maximize, "rank"] = ranks[maximize].groupby(["variant", "metric"])["mean"].rank(
        method="min", ascending=False)
    return pd.concat(individual, ignore_index=True), pd.concat(aggregate, ignore_index=True), ranks


def pareto(summary: pd.DataFrame) -> pd.DataFrame:
    selected = ["objective_mad", "otif", "route_changes", "end_to_end_wall_s", "urgent_service_rate"]
    frame = summary[summary.metric.isin(selected)].pivot(index="policy", columns="metric", values="mean")
    costs, service = frame.objective_mad, frame.otif
    frame["pareto_loss_otif"] = [not ((costs <= cost) & (service >= rate)
                                     & ((costs < cost) | (service > rate))).any() for cost, rate in zip(costs, service)]
    return frame.reset_index()


def prediction_link(predictions: pd.DataFrame, differences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = predictions.block.eq("benchmark") & predictions.traffic_group.eq("all")
    errors = predictions[mask].pivot(index=CASE_KEYS, columns="prediction", values="mae_min")
    routing = differences[differences.contrast.eq("machine_learning")].pivot(
        index=CASE_KEYS, columns="metric", values="difference")
    link = errors.join(routing[["objective_mad", "otif", "regular_otif"]], how="inner")
    link["fixed_mae_difference_min"] = link.ml_fixed - link.historical_fixed
    link["hourly_mae_difference_min"] = link.ml_hourly - link.historical_hourly
    rows = []
    for error in ("fixed_mae_difference_min", "hourly_mae_difference_min"):
        for outcome in ("objective_mad", "otif", "regular_otif"):
            rows.append({"prediction_metric": error, "outcome_metric": outcome, "cases": len(link),
                         "spearman": link[error].corr(link[outcome], method="spearman"),
                         "lower_mae_days": int(link[error].lt(0).sum()),
                         "lower_mae_and_better_outcome_days": int(
                             (link[error].lt(0) & (link[outcome].lt(0) if outcome == "objective_mad"
                                                  else link[outcome].gt(0))).sum())})
    return link.reset_index(), pd.DataFrame(rows)


def analyze(output: Path) -> None:
    config = json.loads((output / "frozen_protocol.json").read_text())["config"]
    frame = pd.read_csv(output / "scenario_metrics.csv")
    bench = frame[frame.block.eq("benchmark")]
    robust = frame[frame.block.eq("sensitivity")]
    coverage = frame.groupby(["block", "variant", "regime", "policy"], as_index=False).agg(
        rows=("case", "size"), planned=("dispatched", "sum"), cases=("case", "nunique"))
    common = complete_cases(bench)
    summary = summaries(common, ["policy"])
    regimes = summaries(common, ["regime", "policy"])
    seed_rows = common[common.training_seed.notna()]
    seeds = summaries(seed_rows, ["policy", "training_seed"])
    contrasts = [pair for pair in CONTRASTS if set(pair[:2]) <= set(bench.policy)]
    differences, intervals = paired(bench, contrasts, config["bootstrap_draws"], config["bootstrap_seed"])
    effects, uncertainty, rankings = sensitivity(robust, config["bootstrap_draws"], config["bootstrap_seed"] + 1)
    prediction, link = prediction_link(pd.read_csv(output / "prediction_errors.csv"), differences)
    for name, table in (("coverage", coverage), ("summary", summary), ("by_regime", regimes), ("by_seed", seeds),
                        ("paired_scenario", differences), ("paired_comparisons", intervals), ("pareto", pareto(summary)),
                        ("sensitivity_scenario", effects), ("sensitivity_effects", uncertainty),
                        ("sensitivity_ranks", rankings), ("prediction_link", prediction), ("prediction_summary", link)):
        table.to_csv(output / f"{name}.csv", index=False)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/experiments/final_benchmark"))
    args = parser.parse_args()
    analyze(args.output)


if __name__ == "__main__":
    main()
