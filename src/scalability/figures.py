"""Render scalable, publication-ready figures from the archived measurements."""

from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from src.scalability.analysis import read_runs

COLORS = dict(B4="#48596A", B5="#006D77", B8="#A54C20", B9="#664DA1", B10="#243F93")
PLANS = {"witness": ("Témoin", "#48596A"), "greedy": ("Glouton", "#006D77"),
         "ortools_cold": ("OR-Tools à froid", "#A54C20"), "ortools_warm": ("OR-Tools à chaud", "#243F93")}


def plot(root: Path):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.labelsize": 11, "font.size": 10, "figure.dpi": 130,
                         "figure.constrained_layout.use": True})
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    protocol = json.loads(next(root.glob("standard_*/protocol.json")).read_text())["config"]
    plot_limit = protocol["reference_candidates"]
    all_plans = read_runs(root, "planning.csv")
    plans = all_plans.query("profile == 'standard' and regime != 'no_reefer'").copy()
    dispatch = read_runs(root, "dispatch.csv").query("profile == 'standard'").copy()
    dispatch = dispatch[dispatch.ablation.isna()] if "ablation" in dispatch else dispatch
    standard = dispatch[dispatch.candidate_limit.eq(plot_limit)]
    predictions = pd.read_csv(root / "prediction_profile.csv")
    resources = pd.read_csv(root / "resource_profile.csv")
    paired = pd.read_csv(root / "paired_case_policy.csv").query("profile == 'standard'")
    size = sorted(plans.customers.unique())

    def figure(name, fig):
        fig.savefig(destination / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(destination / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def axis_x(ax, ticks=size, name="Clients réguliers"):
        ax.set_xlabel(name)
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.grid(alpha=.20)

    def values(ax, frame, x, y, label, color, *, style="-", spread=False):
        summary = frame.groupby(x)[y].agg(["median", "min", "max"]).dropna()
        if summary.empty:
            return
        ax.plot(summary.index, summary["median"], marker="o", linestyle=style, color=color, label=label,
                markersize=4)
        if spread:
            ax.vlines(summary.index, summary["min"], summary["max"], color=color, alpha=.28, linewidth=4)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    display = plans[plans.budget_s.eq(plans.budget_s.max()) | plans.planner.isin(["greedy", "witness"])]
    for key, (label, color) in PLANS.items():
        rows = display[display.planner.eq(key)]
        if key == "witness":
            values(left, rows, "customers", "fleet_sizing_wall_s", "Dimensionnement + témoin", color, spread=True)
        else:
            values(left, rows, "customers", "planning_wall_s", label, color, spread=True)
            values(right, rows, "customers", "first_incumbent_s", label, color, spread=True)
    missing = display[~display.usable]
    if len(missing):
        for status, rows in missing.groupby("status"):
            right.scatter(rows.customers, rows.planning_wall_s, marker="x", color="#902F42",
                          label={"timeout": "Temps écoulé", "no_solution": "Terminé sans plan"}.get(status, status))
    for ax, title in ((left, "Temps de planification"), (right, "Premier plan natif ou glouton terminé")):
        axis_x(ax)
        ax.set_title(title)
        ax.set_ylabel("Secondes, échelle log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
    figure("01_initial_routing", fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for label, source, color in (("OSRM · expérience complète", "standard", "#006D77"),
                                 ("Haversine · charge isolée", "proxy", "#A54C20")):
        rows = resources[resources.profile.eq(source)]
        values(ax, rows, "customers", "process_peak_mib", label, color, style="--" if source == "proxy" else "-")
    axis_x(ax, sorted(resources.customers.unique()))
    ax.set_ylabel("Pic de RSS du processus (Mio)")
    ax.set_title("Mémoire du processus par taille")
    ax.legend(frameon=False)
    workload = predictions[predictions.scope.eq("matrix_and_prediction_only")]
    if len(workload):
        n = int(workload.customers.max())
        ax.text(.98, .04, f"{n} clients · matrice et prévision", transform=ax.transAxes, ha="right", fontsize=8)
    figure("02_peak_memory", fig)

    proxy = all_plans[all_plans.profile.eq("proxy") & all_plans.customers.gt(max(size))]
    quality = pd.concat([plans, proxy], ignore_index=True)
    witness = quality[quality.planner.eq("witness")].set_index("case").cost_mad
    quality["witness_savings"] = 100 * (1 - quality.cost_mad / quality.case.map(witness))
    frames = list(quality.groupby(["profile", "customers"], sort=False))
    fig, panels = plt.subplots(1, len(frames), figsize=(12, 4.0), sharey=True)
    for ax, ((profile, n), frame) in zip(panels, frames, strict=True):
        budget_ticks = sorted(frame.budget_s.dropna().unique())
        for mode in ("ortools_cold", "ortools_warm"):
            style, color = PLANS[mode]
            rows = frame[frame.planner.eq(mode)]
            values(ax, rows, "budget_s", "witness_savings", style, color, spread=True)
            for (budget, status), count in rows[~rows.usable].groupby(["budget_s", "status"]).size().items():
                state = "délai" if status == "timeout" else "sans plan"
                ax.annotate(f"×{count} {state}", (budget, .92 if mode == "ortools_cold" else .82),
                            xycoords=ax.get_xaxis_transform(), xytext=(3, 0), textcoords="offset points",
                            color=color, fontsize=8)
        ax.axhline(0, color="0.65", linewidth=.8)
        if len(budget_ticks) > 1:
            ax.set_xlim(budget_ticks[0] / 1.17, budget_ticks[-1] * 1.17)
        axis_x(ax, budget_ticks, "Budget solveur (s)")
        ax.set_title(f"{n} clients · {'OSRM' if profile == 'standard' else 'Haversine'}")
    panels[0].set_ylabel("Réduction du coût vs témoin (%)")
    panels[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Qualité du plan selon le budget OR-Tools")
    figure("03_solution_quality", fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.7, 3.8))
    for scope, style in (("standard", "-"), ("proxy", "--")):
        rows = predictions[predictions.profile.eq(scope)]
        for key, name, color in (("feature_wall_s", "Variables", "#006D77"),
                                 ("prediction_wall_s", "Inférence", "#A54C20"),
                                 ("forecast_wall_s", "Total + historique", "#243F93")):
            values(left, rows, "customers", key, name if scope == "standard" else "_nolegend_", color, style=style)
    left.set_title("Prévision sur 24 heures")
    left.text(.04, .96, "Trait plein OSRM  ·  tirets Haversine", transform=left.transAxes, va="top", fontsize=8)
    left.legend(frameon=False, fontsize=8, loc="lower right")
    acquisition = root / "geography_acquisition.csv"
    if acquisition.exists():
        cols = pd.read_csv(acquisition)
        values(right, cols, "customers", "matrix_reload_s", "Cache OSRM", "#006D77")
        values(right, cols, "customers", "network_build_s", "Requêtes OSRM", "#A54C20")
    matrix = pd.read_csv(root / "matrix_profile.csv").query("profile == 'proxy'")
    values(right, matrix, "customers", "matrix_wall_s", "Haversine local", "#243F93", style="--")
    right.set_title("Construction et relecture de la matrice")
    right.legend(frameon=False, fontsize=8)
    for ax in (left, right):
        axis_x(ax, sorted(predictions.customers.unique()))
        ax.set_yscale("log")
        ax.set_ylabel("Secondes, échelle log")
    figure("04_prediction_and_matrices", fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.3, 3.8))
    for policy in ("B5", "B8", "B9", "B10"):
        rows = standard[standard.policy.eq(policy)]
        values(left, rows, "customers", "planning_wall_s", policy, COLORS[policy], spread=True)
        values(right, rows, "customers", "replan_p95_ms", policy, COLORS[policy], spread=True)
    for ax, title, name in ((left, "Calcul cumulé par journée", "Secondes murales"),
                            (right, "P95 des tentatives actives", "Millisecondes murales")):
        axis_x(ax)
        ax.set_title(title)
        ax.set_ylabel(name)
        ax.legend(frameon=False)
    right.set_yscale("log")
    figure("05_replanning_time", fig)

    stressed = paired[paired.regime.eq("stress") & paired.customers.isin(protocol["sweep_sizes"])
                      & paired.policy.isin(protocol["sweep_policies"])]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.7), sharey=True)
    for ax, (n, rows) in zip((left, right), stressed.groupby("customers"), strict=True):
        for policy, group in rows.groupby("policy"):
            values(ax, group, "candidate_limit", "paired_loss_per_customer_mad", policy, COLORS[policy], spread=True)
        axis_x(ax, sorted(stressed.candidate_limit.unique()), "Candidats par tentative")
        ax.axhline(0, color="0.60", linewidth=.8)
        ax.set_title(f"{n} clients, stress")
        ax.legend(frameon=False)
    left.set_ylabel("Écart de perte vs B4 (MAD/client)")
    figure("06_candidate_budgets", fig)

    paired_standard = paired[paired.candidate_limit.eq(plot_limit)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, regime in ((left, "controlled"), (right, "stress")):
        rows = paired_standard[paired_standard.regime.eq(regime)]
        for policy, group in rows.groupby("policy"):
            values(ax, group.assign(otif_pct=group.otif * 100), "customers", "otif_pct", policy,
                   COLORS[policy], spread=True)
        axis_x(ax)
        ax.set_title("Perturbations modérées" if regime == "controlled" else "Stress")
    left.set_ylabel("OTIF total (%)")
    right.legend(frameon=False, ncol=2)
    figure("07_otif", fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for policy in COLORS:
        rows = paired_standard[paired_standard.policy.eq(policy)]
        values(left, rows, "customers", "paired_loss_per_customer_mad", policy, COLORS[policy], spread=True)
        values(right, rows.assign(urgent_pct=100*rows.urgent_service_rate), "customers", "urgent_pct", policy,
               COLORS[policy], spread=True)
    left.axhline(0, color="0.65", linewidth=.8)
    left.set_ylabel("Écart de perte vs B4 (MAD/client)")
    right.set_ylabel("Urgences servies (%)")
    for ax in (left, right):
        axis_x(ax)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Politiques appariées. Médianes et étendues observées")
    figure("08_policy_transfer", fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    keys = ["planning_wall_s", "simulation_with_probes_wall_s", "inference_wall_s"]
    layers = [("Replan", "#A54C20"), ("Moteur + observations", "#006D77"), ("Inférence politique", "#A8B8D4")]
    groups = standard[standard.policy.isin(["B5", "B8", "B9", "B10"])].groupby(["customers", "policy"])[keys].median()
    heights = np.zeros(len(groups))
    for key, (label, color) in zip(keys, layers, strict=True):
        ax.bar(np.arange(len(groups)), groups[key], bottom=heights, label=label, color=color)
        heights += groups[key].to_numpy()
    ax.set_xticks(np.arange(len(groups)), [f"{n}\n{p}" for n, p in groups.index], fontsize=8)
    ax.set_ylabel("Secondes médianes par relecture")
    ax.set_title("Décomposition du calcul au moment de l’exécution")
    ax.grid(axis="y", alpha=.20)
    ax.legend(frameon=False)
    figure("09_runtime_breakdown", fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args().output)


if __name__ == "__main__":
    main()
