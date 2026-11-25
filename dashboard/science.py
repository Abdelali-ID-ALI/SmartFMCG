"""Interactive views of frozen scientific artifacts and their evaluation protocols."""

import base64
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ARCHIVE = Path("results/experiments")
DQN = "dqn_evaluation"
FINAL = "final_benchmark"
SCALE = "scalability"
ML = "traffic_ml"
PTO = "predict_then_optimize"
LABELS = {"objective_mad": "Operational loss (MAD)", "otif": "OTIF (%)",
          "urgent_service_rate": "Urgent orders served (%)", "operating_cost_mad": "Operating cost (MAD)",
          "replan_attempts": "Replanning attempts", "median_planning_wall_s": "Median planning wall time (s)",
          "process_peak_mib": "Process peak RSS (MiB)", "loss_mad": "Operational loss (MAD)",
          "urgent_service": "Urgent orders served (%)", "replan_wall_s": "Replanning wall time (s)"}


@st.cache_data(max_entries=45, show_spinner=False)
def read_csv(path: Path, modified_ns: int):
    return pd.read_csv(path)


def table(experiment, name):
    path = ARCHIVE / experiment / f"{name}.csv"
    return read_csv(path, path.stat().st_mtime_ns)


@st.cache_data(max_entries=1, show_spinner=False)
def shap_images(path: Path, modified_ns: int):
    notebook = json.loads(path.read_text())
    cell = next(cell for cell in notebook["cells"] if "shap.plots.beeswarm" in "".join(cell["source"]))
    return tuple(base64.b64decode(out["data"]["image/png"]) for out in cell["outputs"]
                 if "image/png" in out.get("data", {}))


def chart(figure, key):
    figure.update_layout(margin=dict(l=12, r=15, t=30, b=16), legend_title_text="")
    st.plotly_chart(figure, key=key)


def traffic_page():
    st.title("Travel-time prediction")
    st.caption("Archived Phase 5 · Chronological held-out test, 11 days and 1,966 synthetic observations")
    metrics = table(ML, "test_metrics")
    selected = metrics.loc[metrics.model.eq("linear/ratio/ols")].iloc[0]
    static = metrics.loc[metrics.model.eq("static_osrm")].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Mean absolute error", f"{selected.mae:.3f} min",
             f"{selected.mae - static.mae:+.3f} vs static", delta_color="inverse", border=True)
    b.metric("Root mean squared error", f"{selected.rmse:.3f} min", border=True)
    c.metric("R²", f"{selected.r2:.3f}", border=True)
    st.markdown("Linear regression on the ratio to static OSRM travel time was selected on validation "
                "(MAE 1.259 min). On the test set it improved MAE by 8.67% over historical averages. "
                "Tree ensembles gave similar errors on the ratio target.")
    models, residuals, explain, impact = st.tabs(("Model comparison", "Residuals", "Explainability",
                                                   "Planning impact"))
    with models:
        chart(px.bar(metrics.sort_values("mae", ascending=False), x="mae", y="model", orientation="h",
                     labels={"mae": "Test MAE (min)", "model": "Forecast model"},
                     hover_data=["rmse", "r2", "bias", "batch_ms"]), "ml_family")
        st.dataframe(metrics.rename(columns={"mae": "MAE (min)", "rmse": "RMSE (min)", "r2": "R²",
                                             "bias": "Bias (min)", "batch_ms": "Batch time (ms)"}),
                     hide_index=True)
    with residuals:
        rows = table(ML, "test_residuals")
        group = st.radio("Traffic observations", ("All", "Ordinary", "Incidents"), horizontal=True)
        if group != "All":
            rows = rows[rows.incident.eq("incident" if group == "Incidents" else "normal")]
        st.caption(f"{len(rows):,} rows · Residual = predicted − simulated actual")
        labels = {"actual_simulated_min": "Simulated actual (min)", "predicted_min": "Predicted (min)",
                  "residual_min": "Residual (min)", "hour": "Hour of day"}
        left, right = st.columns(2)
        scatter = px.scatter(rows, x="actual_simulated_min", y="predicted_min", labels=labels,
                             opacity=.7, hover_data=["timestamp"])
        if rows.empty:
            st.info("No observations match this filter.")
        else:
            ceiling = max(rows.actual_simulated_min.max(), rows.predicted_min.max())
            scatter.add_shape(type="line", x0=0, y0=0, x1=ceiling, y1=ceiling, line=dict(dash="dot", color="#627589"))
        with left:
            chart(scatter, "ml_fit")
        with right:
            chart(px.histogram(rows, x="residual_min", labels=labels), "ml_residual")
        st.caption("The 75 held-out incidents show a mean residual of −7.168 min. "
                   "These forecasts model synthetic traffic under the archived generation assumptions.")
        st.dataframe(table(ML, "test_strata"), hide_index=True)
    with explain:
        perm = table(ML, "permutation_importance").sort_values("mae_increase")
        chart(px.bar(perm, x="mae_increase", y="feature", error_x="repeat_sd", orientation="h",
                     labels={"mae_increase": "Validation MAE increase (min)", "feature": "Feature"}), "ml_perm")
        st.caption("Error bars show spread across permutations. Correlated inputs share information.")
        with st.expander("Saved SHAP analysis"):
            path = Path("notebooks/05_traffic_prediction.ipynb")
            for image, title in zip(shap_images(path, path.stat().st_mtime_ns),
                                    ("Global SHAP beeswarm", "One held-out prediction"), strict=True):
                st.image(image, caption=title)
            st.caption("Archived ratio-output explanation. 24 training background rows and 40 test rows.")
    with impact:
        accuracy = table(PTO, "forecast_accuracy")
        plans = table(PTO, "day_metrics")
        st.caption("Phase 6 · Matched synthetic test dates and available information. Forecast error uses "
                   "minutes. Replayed plan results use travel-time realizations and service windows.")
        chart(px.box(accuracy, x="strategy", y="mae_min",
                     labels={"strategy": "Forecast", "mae_min": "Daily MAE (min)"}), "ml_pto_error")
        chart(px.scatter(plans.assign(otif_pct=plans.otif * 100), x="realized_cost_mad", y="otif_pct",
                         color="strategy", hover_data=["day", "delay_min", "planned_cost_mad"],
                         labels={"realized_cost_mad": "Realized cost (MAD)", "otif_pct": "Realized OTIF (%)"}),
              "ml_pto_plan")
        st.caption("Robust ML applies an 8.87% margin calibrated on validation. A lower average prediction "
                   "error can still produce a worse route when a few late arrivals cross hard service windows.")


def learning_page():
    st.title("Learning policies")
    st.caption("Archived Phase 9B · 36 held-out cases, 18 per fleet size · Three frozen training seeds per fitted family")
    st.markdown("Policies choose a dispatch action at each public decision epoch. The simulator enforces "
                "feasibility and only reveals events as they occur. Q-Learning estimates a value for each "
                "discrete state–action pair. DQN estimates action values with a neural network using 22 "
                "continuous indicators or the compact encoding. Rules use explicit event and risk triggers.")
    with st.expander("Actions and learning objective"):
        st.markdown("""
- **Keep plan** preserves current commitments.
- **Local heuristic** searches nearby feasible suffix changes.
- **Local reoptimization** invokes the bounded CP-SAT suffix solver.
- **Global reoptimization** searches the editable suffixes across available vehicles.

The reward is the negative incremental operational loss. The episode ends after the simulated day and
final delivery accounting. The action mask and independent validation constrain proposals.
A pending replan can still fail validation at application time.
""")
    family_tab, policy_tab, transfer_tab = st.tabs(("Learning curves", "Held-out performance", "Transfer"))
    with family_tab:
        families = ("tabular_A", "tabular_B", "tabular_D", "dqn_continuous", "dqn_compact")
        family = st.selectbox("Training family", families, index=3)
        seeds = (2917, 2927, 2953)
        parts = [table(DQN, f"fit/{family}_{seed}/validation") for seed in seeds]
        validation = pd.concat(parts, ignore_index=True)
        averaged = validation.groupby(["training_seed", "step"], as_index=False).objective_mad.mean()
        averaged["Seed"] = averaged.training_seed.astype(str)
        chart(px.line(averaged, x="step", y="objective_mad", color="Seed", markers=True,
                      labels={"step": "Training interactions", "objective_mad": "Validation loss (MAD)"}), "rl_validation")
        traces = []
        for seed in seeds:
            log = table(DQN, f"fit/{family}_{seed}/episodes")
            log = log.assign(Seed=str(seed))
            log["Trailing training loss (MAD)"] = log.objective_mad.rolling(10, min_periods=1).mean()
            traces.append(log)
        episodes = pd.concat(traces, ignore_index=True)
        chart(px.line(episodes, x="global_step", y="Trailing training loss (MAD)", color="Seed",
                      labels={"global_step": "Training interactions"}), "rl_training")
        st.caption("Validation uses 12 cases. The training plot shows a trailing mean over up to 10 episodes.")
    with policy_tab:
        fleet = st.radio("Available vehicles", (3, 4), horizontal=True)
        summary = table(DQN, "final_summary")
        subset = summary[summary.fleet.eq(fleet)].copy()
        subset["OTIF (%)"] = subset.otif * 100
        subset["Urgent service (%)"] = subset.urgent_service_rate * 100
        chart(px.scatter(subset, x="objective_mad", y="OTIF (%)", hover_name="policy",
                         size="replan_attempts", hover_data=["Urgent service (%)", "replans_applied"],
                         labels={"objective_mad": LABELS["objective_mad"]}), "rl_results")
        st.dataframe(subset[["policy", "objective_mad", "OTIF (%)", "Urgent service (%)", "replan_attempts",
                             "replans_applied", "solver_calls"]].rename(columns=LABELS), hide_index=True)
        dist = table(DQN, "final_action_distribution")
        policy = st.selectbox("Action distribution policy", sorted(dist.policy.unique()), key="archive_action_policy")
        actions = dist[dist.fleet.eq(fleet) & dist.policy.eq(policy)]
        chart(px.bar(actions, x="action", y=["requested_when_eligible", "executed_when_eligible"],
                     barmode="group", labels={"value": "Eligible decisions", "action": "Dispatch action"}),
              "rl_actions")
        with st.expander("Paired bootstrap intervals"):
            st.dataframe(table(DQN, "paired_intervals"), hide_index=True)
    with transfer_tab:
        st.markdown("The event rule retained lower mean loss than continuous DQN in Phase 9B. At 100 customers, "
                    "Q-Learning improved on the fixed reference in the archived cases. Policy ordering may shift "
                    "with the scenario, and tabular state coverage can hide distinct physical situations.")
        observations = table(SCALE, "observation_support")
        base = observations.query("profile == 'standard' and candidate_limit == 400")
        grouped = base.groupby(["policy", "customers"], as_index=False)[
            ["eligible", "outside_box", "unseen_state"]].sum(min_count=1)
        grouped["Outside / unseen (%)"] = (grouped.outside_box.fillna(grouped.unseen_state)
                                           / grouped.eligible * 100)
        chart(px.bar(grouped, x="customers", y="Outside / unseen (%)", color="policy", barmode="group",
                     hover_data=["eligible", "outside_box", "unseen_state"], labels={"customers": "Customers"}),
              "rl_transfer")
        st.caption("B10 counts observations outside at least one archived training range. B9 counts unseen "
                   "tabular states. These diagnostics answer different questions and neither estimates failure risk.")
        st.dataframe(grouped, hide_index=True)


def benchmark_page():
    st.title("Scientific benchmark")
    st.caption("Archived experiments · Synthetic operations and economic assumptions in MAD")
    unified, paired, scale = st.tabs(("Unified comparison", "Ablations & sensitivity", "Scalability"))
    with unified:
        st.markdown("Phase 10 · The comparison uses **35 common cases over 7 date blocks**. Learned-policy "
                    "seeds are averaged within a case. Whiskers show between-case standard deviation.")
        rows = table(FINAL, "summary")
        metric = st.selectbox("Outcome", ("objective_mad", "otif", "urgent_service_rate", "operating_cost_mad",
                                          "replan_attempts"), format_func=LABELS.get)
        shown = rows[rows.metric.eq(metric)].copy()
        if "%" in LABELS[metric]:
            shown[["mean", "sd"]] *= 100
        chart(px.bar(shown, x="policy", y="mean", error_y="sd", hover_data=["cases", "date_blocks"],
                     labels={"mean": LABELS[metric], "policy": "Policy"}), "benchmark_unified")
        frontier = table(FINAL, "pareto").assign(otif_pct=lambda x: x.otif * 100)
        chart(px.scatter(frontier, x="objective_mad", y="otif_pct", hover_name="policy", color="pareto_loss_otif",
                         hover_data=["end_to_end_wall_s", "urgent_service_rate", "route_changes"],
                         labels={"objective_mad": LABELS["objective_mad"], "otif_pct": LABELS["otif"],
                                 "pareto_loss_otif": "Loss–OTIF frontier"}), "benchmark_frontier")
        with st.expander("Matched information comparison · B9, B10 and B12"):
            st.dataframe(table(FINAL, "approximation_summary"), hide_index=True)
            st.dataframe(table(FINAL, "approximation_intervals"), hide_index=True)
        with st.expander("Coverage and resource profile"):
            st.dataframe(table(FINAL, "coverage"), hide_index=True)
            st.dataframe(table(FINAL, "resource_profile"), hide_index=True)
    with paired:
        st.caption("Paired difference = left minus right. Error bars are descriptive 95% date-block bootstrap intervals.")
        pair_metric = st.radio("Paired outcome", ("objective_mad", "otif", "urgent_service_rate"),
                               format_func=LABELS.get, horizontal=True)
        data = table(FINAL, "paired_comparisons")
        regime = st.selectbox("Disruption regime", sorted(data.regime.unique()), index=0)
        differences = data[data.regime.eq(regime) & data.metric.eq(pair_metric)].copy()
        multiplier = 100 if pair_metric != "objective_mad" else 1
        differences["Difference"] = differences.difference * multiplier
        differences["Upper"] = (differences.ci_high - differences.difference).clip(lower=0) * multiplier
        differences["Lower"] = (differences.difference - differences.ci_low).clip(lower=0) * multiplier
        differences["Comparison"] = differences.left + " − " + differences.right + " · " + differences.contrast
        figure = px.scatter(differences, x="Difference", y="Comparison", error_x="Upper", error_x_minus="Lower",
                            hover_data=["cases", "date_blocks", "ci_low", "ci_high"],
                            labels={"Difference": "Difference (MAD)" if multiplier == 1 else "Difference (pp)"})
        figure.add_vline(x=0, line_dash="dot")
        figure.update_layout(height=530)
        chart(figure, "benchmark_paired")
        st.subheader("Sensitivity tests")
        st.caption("Scenario counts and uncertainty vary by variant. Each row compares a variant with its own reference.")
        effects = table(FINAL, "sensitivity_effects")
        st.dataframe(effects[effects.metric.eq(pair_metric)], hide_index=True)
    with scale:
        scalability()


def scalability():
    st.markdown("Phase 10.1 · End-to-end OSRM replays cover 30, 50 and 100 customers, with up to 16 "
                "available vehicles. Haversine stress tests at 250 and 500 customers cover partial workloads.")
    data = table(SCALE, "dispatch_summary")
    size = st.selectbox("OSRM customers", (30, 50, 100), index=2)
    group = st.radio("Scenario regime", ("controlled", "stress"), horizontal=True)
    part = data[data.profile.eq("standard") & data.customers.eq(size) & data.regime.eq(group)]
    budgets = sorted(part.candidate_limit.unique())
    budget = st.selectbox("Candidate ceiling", budgets, index=budgets.index(400))
    subset = part[part.candidate_limit.eq(budget)].assign(
        **{"OTIF (%)": lambda df: df.otif * 100, "Urgents served (%)": lambda df: df.urgent_service * 100})
    chart(px.scatter(subset, x="loss_mad", y="OTIF (%)", hover_name="policy",
                     hover_data=["Urgents served (%)", "cases", "loss_delta_min_mad", "loss_delta_max_mad",
                                 "replan_wall_s", "solver_calls", "limit_hits"], labels=LABELS), "scale_dispatch")
    st.dataframe(subset, hide_index=True)
    planning = table(SCALE, "planning_summary")
    with st.expander("Solver budget and feasibility"):
        eligible = planning[planning.profile.eq("standard") & planning.customers.eq(size) &
                            planning.regime.eq(group)]
        chart(px.line(eligible.dropna(subset=["budget_s"]), x="budget_s", y="median_planning_wall_s",
                      color="planner", markers=True, hover_data=["feasible", "timeouts", "cases"],
                      labels={**LABELS, "budget_s": "Solver limit (s)"}), "scale_solver")
        st.dataframe(eligible, hide_index=True)
        st.dataframe(table(SCALE, "replanning_action_profile").query("profile == 'standard' and customers == @size"),
                     hide_index=True)
    with st.expander("Computational profiles and partial stress tests"):
        resources = table(SCALE, "resource_profile")
        chart(px.bar(resources, x="customers", y="process_peak_mib", color="geography", barmode="group",
                     hover_data=["profile", "wall_s", "cases", "replays"], labels=LABELS), "scale_memory")
        st.dataframe(resources, hide_index=True)
        st.dataframe(table(SCALE, "prediction_profile"), hide_index=True)
    with st.expander("Reserve inventory ablation"):
        st.dataframe(table(SCALE, "reserve_ablation"), hide_index=True)
    st.caption("Intervals and ranges reflect the archived synthetic cases. The 100-customer study has four "
               "standard cases. Transfer across geography or real operations remains unmeasured.")
