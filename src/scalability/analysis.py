"""Summarize paired scalability runs without treating model seeds as independent cases."""

from argparse import ArgumentParser
from pathlib import Path
import json
import pstats

import pandas as pd


def read_runs(root: Path, filename: str) -> pd.DataFrame:
    files = sorted(root.glob(f"*/{filename}"))
    if filename.endswith(".json"):
        return pd.DataFrame(json.loads(path.read_text()) for path in files)
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True) if files else pd.DataFrame()


def aggregate(root: Path, profile_path: Path | None = None):
    plans = read_runs(root, "planning.csv")
    dispatch = read_runs(root, "dispatch.csv")
    predictions = read_runs(root, "prediction.csv")
    matrices = read_runs(root, "matrix.json").drop(columns=["locations", "provenance"])
    resources = read_runs(root, "resources.json")
    observations = read_runs(root, "observations.csv.gz")
    decisions = read_runs(root, "decisions.csv")
    has_ablation = "ablation" in dispatch
    clean = dispatch[dispatch.ablation.isna()].copy() if has_ablation else dispatch.copy()
    fields = ("profile", "geography", "customers", "regime", "planner", "budget_s")
    plans["reference_sizing_plus_plan_wall_s"] = plans.planning_wall_s + plans.fleet_sizing_wall_s
    planning_summary = plans.groupby(list(fields), dropna=False).agg(
        cases=("case", "nunique"), feasible=("usable", "sum"),
        timeouts=("status", lambda x: (x == "timeout").sum()),
        proven_infeasible=("status", lambda x: (x == "infeasible").sum()),
        other_without_plan=("status", lambda x: (~x.isin(["timeout", "infeasible", "feasible", "optimal"])).sum()),
        median_first_incumbent_s=("first_incumbent_s", "median"),
        median_planning_wall_s=("planning_wall_s", "median"), max_planning_wall_s=("planning_wall_s", "max"),
        median_cost_mad=("cost_mad", "median"), median_planned_otif=("planned_otif", "median"),
        median_vehicles_used=("vehicles_used", "median"), median_distance_km=("distance_km", "median"),
        max_planning_process_mib=("planning_process_peak_mib", "max"),
        median_sizing_plus_plan_wall_s=("reference_sizing_plus_plan_wall_s", "median")).reset_index()

    metrics = ["objective_mad", "otif", "urgent_service_rate", "replay_wall_s", "planning_wall_s",
               "route_changes", "replan_attempts", "replans_applied", "candidate_limit_hits",
               "candidates", "mean_policy_inference_ms", "solver_calls", "safety_fallbacks",
               "unserved_mandatory", "probe_exhaustions", "modeled_latency_s"]
    keys = ["profile", "geography", "customers", "regime", "case", "policy", "candidate_limit"]
    # Average the frozen training seeds within each scenario before comparing two policies.
    case_policy = clean.groupby(keys)[metrics].mean().reset_index()
    baseline = case_policy[case_policy.policy.eq("B4")].drop(columns=["policy", "candidate_limit"])
    case_policy = case_policy.merge(baseline[[*keys[:5], "objective_mad", "otif", "urgent_service_rate"]],
                                    on=keys[:5], suffixes=("", "_b4"), validate="many_to_one")
    case_policy["paired_loss_mad"] = case_policy.objective_mad - case_policy.objective_mad_b4
    case_policy["paired_loss_per_customer_mad"] = case_policy.paired_loss_mad / case_policy.customers
    case_policy["paired_otif_pp"] = 100 * (case_policy.otif - case_policy.otif_b4)
    case_policy["paired_urgent_pp"] = 100 * (case_policy.urgent_service_rate - case_policy.urgent_service_rate_b4)
    summary = case_policy.groupby([*keys[:4], "policy", "candidate_limit"]).agg(
        cases=("case", "nunique"), loss_mad=("objective_mad", "mean"),
        loss_delta_mad=("paired_loss_mad", "mean"), loss_delta_min_mad=("paired_loss_mad", "min"),
        loss_delta_max_mad=("paired_loss_mad", "max"),
        loss_delta_per_customer_mad=("paired_loss_per_customer_mad", "mean"),
        otif=("otif", "mean"), otif_min=("otif", "min"), otif_max=("otif", "max"),
        otif_delta_pp=("paired_otif_pp", "mean"), urgent_service=("urgent_service_rate", "mean"),
        urgent_delta_pp=("paired_urgent_pp", "mean"), replan_wall_s=("planning_wall_s", "mean"),
        mean_inference_ms=("mean_policy_inference_ms", "mean"), solver_calls=("solver_calls", "mean"),
        safety_fallbacks=("safety_fallbacks", "sum"), candidates=("candidates", "mean"),
        limit_hits=("candidate_limit_hits", "mean"), route_changes=("route_changes", "mean"),
        replans_applied=("replans_applied", "mean"), probe_exhaustions=("probe_exhaustions", "mean")).reset_index()
    action_stats = (observations[observations.get("ablation", pd.Series(index=observations.index, dtype=str)).isna()]
                    .groupby([*keys, "training_seed", "eligible", "action"], dropna=False)
                    .size().rename("decisions").reset_index())
    eligible = observations["eligible"].fillna(False)
    ood = observations[eligible & observations.policy.isin(["B9", "B10"])].copy()
    ood_counts = ood.groupby([*keys, "training_seed"]).agg(
        eligible=("eligible", "size"), outside_box=("ood_box", lambda x: x.sum(min_count=1)),
        unseen_state=("unseen_table_state", lambda x: x.sum(min_count=1)),
        unseen_actions=("no_trained_available_action", lambda x: x.sum(min_count=1))).reset_index()
    outside = (ood[ood.policy.eq("B10")].assign(
        feature=lambda x: x.outside_features.str.split("|")).explode("feature").dropna(subset=["feature"]))
    feature_ood = (outside.groupby(["profile", "customers", "regime", "training_seed", "feature"])
                   .size().rename("out_of_range_count").reset_index())
    if "ablation" in decisions:
        decisions = decisions[decisions.ablation.isna()]
    attempted = decisions[decisions.attempted].copy()
    attempted["planning_wall_ms"] = 1000 * attempted.planning_wall_s
    by_action = attempted.groupby(["profile", "customers", "candidate_limit", "requested"]).agg(
        attempts=("case", "size"), affected_cases=("case", "nunique"),
        median_wall_ms=("planning_wall_ms", "median"),
        p95_wall_ms=("planning_wall_ms", lambda x: x.quantile(.95)),
        median_search_items=("search_items", "median"), solver_calls=("solver_calls", "sum"),
        applications=("applied_s", "count")).reset_index()
    output = {"planning_summary": planning_summary, "paired_case_policy": case_policy, "dispatch_summary": summary,
              "prediction_profile": predictions, "matrix_profile": matrices,
              "resource_profile": resources, "action_counts": action_stats,
              "observation_support": ood_counts, "ood_features": feature_ood, "replanning_action_profile": by_action}
    if has_ablation:
        ablated = dispatch[dispatch.ablation.eq("zero_reserve")]
        comparison = ["case", "policy", "candidate_limit"]
        affected = clean.merge(ablated, on=comparison, suffixes=("_base", "_no_reserve"),
                               validate="many_to_one")
        for field in ("objective_mad", "otif", "urgent_service_rate", "route_changes"):
            affected[f"delta_{field}"] = affected[f"{field}_no_reserve"] - affected[f"{field}_base"]
        selected = [*comparison, "customers_base", "loaded_reserve_crates_base", "loaded_reserve_crates_no_reserve"]
        output["reserve_ablation"] = affected[selected + [c for c in affected if c.startswith("delta_")]]
    for name, frame in output.items():
        frame.to_csv(root / f"{name}.csv", index=False)
    if profile_path is not None:
        stats = pstats.Stats(str(profile_path))
        calls = [{"function": f"{Path(file).name}:{line}:{fn}", "primitive_calls": record[0],
                  "total_calls": record[1], "self_s": record[2], "cumulative_s": record[3]}
                 for (file, line, fn), record in stats.stats.items()]
        pd.DataFrame(calls).sort_values("self_s", ascending=False).head(80).to_csv(
            root / "python_profile_top.csv", index=False)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cprofile", type=Path)
    args = parser.parse_args()
    aggregate(args.output, args.cprofile)


if __name__ == "__main__":
    main()
