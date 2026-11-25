"""Publication figures generated from the saved Phase 10 results."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "B0": "Glouton statique", "B1": "OR-Tools statique", "B2": "Historique", "B3": "ML sans marge",
    "B4": "ML avec marge", "B5": "Événementiel", "B6": "Périodique", "B7": "À seuil", "B8": "Multicritère",
    "B9": "Q-Learning", "B10": "DQN continu", "B11": "Aléatoire masqué",
}
REGIMES = {
    "normal": "Normal", "moderate": "Modéré", "severe": "Sévère",
    "urgent_pressure": "Urgences nombreuses", "limited_resources": "Ressources limitées",
}
FACTORS = {
    "penalty_low": "Omission · 40 MAD", "penalty_high": "Omission · 120 MAD",
    "latency_low": "Diffusion · 5 s", "latency_high": "Diffusion · 60 s",
    "reserve_low": "Réserve · 3 caisses", "reserve_high": "Réserve · 14 caisses",
    "congestion_low": "Congestion · ×1,15", "congestion_high": "Congestion · ×2,05",
    "service_low": "Surcroît de service · 4 min", "service_high": "Surcroît de service · 19 min",
    "windows_tight": "Fermeture · −15 min", "windows_wide": "Fermeture · +15 min",
    "fleet_light": "Sans V01, utilitaire", "fleet_heavy": "Sans V05, poids lourd",
}
COMPARISONS = {
    "optimization": "OR-Tools / glouton", "historical": "Historique / OSRM",
    "machine_learning": "ML / historique", "safety_margin": "Marge / ML",
    "event_driven": "Événements / marge seule", "multicriteria": "Multicritère / marge seule",
    "tabular_adaptation": "Q-Learning / marge seule", "neural_adaptation": "DQN / marge seule",
    "neural_vs_tabular": "DQN / Q-Learning",
}
COLORS = {**dict.fromkeys(("B0", "B1"), "#64748b"), **dict.fromkeys(("B2", "B3", "B4"), "#167e80"),
          **dict.fromkeys(("B5", "B6", "B7", "B8", "B11"), "#255ba3"),
          **dict.fromkeys(("B9", "B10"), "#a04a25")}


def save(fig, where: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(where / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heatmap(ax, frame, title, label, *, scheme="RdBu", center=True, digits=1):
    values = frame.to_numpy(dtype=float)
    shown = np.ma.masked_invalid(values)
    if center:
        extent = max(.01, np.abs(shown).max())
        image = ax.imshow(shown, aspect="auto", cmap=scheme, vmin=-extent, vmax=extent)
    else:
        image = ax.imshow(shown, aspect="auto", cmap=scheme)
    ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(frame.index)), frame.index)
    ax.set_title(title, loc="left")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            shade = image.cmap(image.norm(value))[:3] if np.isfinite(value) else (1, 1, 1)
            color = "black" if np.dot(shade, [.2126, .7152, .0722]) > .50 else "white"
            ax.text(j, i, f"{value:.{digits}f}" if np.isfinite(value) else "–",
                    ha="center", va="center", color=color, fontsize=8)
    ax.figure.colorbar(image, ax=ax, label=label, shrink=.84)


def render(output: Path) -> None:
    where = output / "figures"
    where.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "pdf.fonttype": 42, "figure.facecolor": "white"})
    summary = pd.read_csv(output / "summary.csv")
    frame = summary.pivot(index="policy", columns="metric", values="mean")
    order = [key for key in LABELS if key in frame.index]
    frame = frame.loc[order]
    palette = [COLORS[key] for key in order]
    names = [f"{key} · {LABELS[key]}" for key in order]
    indices = np.arange(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 6.1), sharey=True, layout="constrained")
    for ax, metric, scale, title, unit in zip(axes, ("objective_mad", "otif", "urgent_service_rate"),
                                               (1, 100, 100), ("Perte opérationnelle", "OTIF total", "Urgences servies"),
                                               ("MAD par scénario", "% de toutes les commandes",
                                                "% par scénario avec urgence"), strict=True):
        values = frame[metric] * scale
        ax.barh(indices, values, color=palette, height=.67)
        ax.bar_label(ax.containers[0], [f"{v:.1f}" for v in values], padding=3, fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_xlabel(unit)
        ax.set_xlim(0, max(values.max() * 1.2, 1.))
        ax.grid(axis="x", alpha=.22)
    axes[0].set_yticks(indices, names)
    axes[0].invert_yaxis()
    fig.suptitle("SmartFMCG · Comparaison de bout en bout", fontsize=15)
    save(fig, where, "01_end_to_end")

    frontier = pd.read_csv(output / "pareto.csv").set_index("policy")
    fig, (overview, detail) = plt.subplots(1, 2, figsize=(11.4, 5.1), layout="constrained",
                                          gridspec_kw={"width_ratios": (1.35, 1)})
    offsets = {"B4": (5, -10), "B5": (5, 6), "B6": (7, -10), "B7": (6, 4),
               "B8": (-15, 9), "B9": (-10, 9), "B10": (5, -12), "B11": (5, 9)}
    efficient = frontier[frontier.pareto_loss_otif].sort_values("objective_mad")
    for ax, keys in ((overview, order), (detail, [key for key in order if key in offsets])):
        ax.plot(efficient.objective_mad, efficient.otif * 100, color="#18304b", linestyle="--",
                linewidth=1.2, label="Front descriptif coût–service")
        for key in keys:
            row = frontier.loc[key]
            ax.scatter(row.objective_mad, row.otif * 100, color=COLORS[key], s=44, zorder=3)
            if ax is detail or key not in offsets:
                ax.annotate(key, (row.objective_mad, row.otif * 100), xytext=offsets.get(key, (4, 5)),
                            textcoords="offset points", fontsize=9)
        ax.set(xlabel="Perte opérationnelle (MAD par scénario)", ylabel="OTIF total (%)")
        ax.grid(alpha=.24)
    overview.set_title("Ensemble des configurations", loc="left")
    detail.set_title("Détail des politiques B4 à B11", loc="left")
    detail.margins(x=.19, y=.22)
    overview.legend(loc="best")
    fig.suptitle("Compromis entre coût et service", fontsize=14)
    save(fig, where, "02_cost_service_pareto")

    pairs = pd.read_csv(output / "paired_comparisons.csv").query("regime == 'all'")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), layout="constrained", sharey=True)
    for ax, metric, factor, title in zip(axes, ("objective_mad", "otif"), (1, 100),
                                         ("Écart de perte (MAD)", "Écart d’OTIF (points)"), strict=True):
        series = pairs[pairs.metric.eq(metric)].set_index("contrast").loc[list(COMPARISONS)]
        y = np.arange(len(series))
        ax.hlines(y, series.ci_low * factor, series.ci_high * factor, color="#255ba3", linewidth=1.6)
        ax.scatter(series.difference * factor, y, s=27, color="#255ba3")
        ax.axvline(0, color="#5c6570", linestyle="--", linewidth=1)
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=.22)
    axes[0].set_yticks(y, list(COMPARISONS.values()))
    axes[0].invert_yaxis()
    fig.suptitle("Contributions des composants · Écarts appariés et intervalles à 95 %", fontsize=13)
    save(fig, where, "03_component_ablation")

    regimes = pd.read_csv(output / "by_regime.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.0), layout="constrained")
    a = regimes[regimes.metric.eq("objective_mad")].pivot(index="policy", columns="regime", values="mean")
    a = a.subtract(a.loc["B4"], axis="columns")
    b = regimes[regimes.metric.eq("otif")].pivot(index="policy", columns="regime", values="mean") * 100
    for ax, matrix, title, unit, center in zip(axes, (a, b), ("Perte relative à B4", "OTIF moyen"),
                                               ("MAD par scénario", "%"), (True, False), strict=True):
        heatmap(ax, matrix.loc[order, list(REGIMES)].rename(columns=REGIMES), title, unit,
                scheme="RdBu_r" if center else "YlGnBu", center=center)
    save(fig, where, "04_disruption_regimes")

    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), sharey=True, layout="constrained")
    for label, metric, color in (("Tentatives", "replan_attempts", "#8da6c3"),
                                  ("Replans appliqués", "replans_applied", "#255ba3"),
                                  ("Prédécesseurs changés", "route_changes", "#a04a25")):
        offset = {"replan_attempts": -.22, "replans_applied": 0., "route_changes": .22}[metric]
        axes[0].barh(indices + offset, frame[metric], height=.2, color=color, label=label)
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Événements par scénario")
    axes[0].set_title("Stabilité des tournées", loc="left")
    axes[1].barh(indices, frame.end_to_end_wall_s, color=palette, height=.65)
    axes[1].set_xlabel("Secondes par scénario")
    axes[1].set_title("Planification et rejeu", loc="left")
    axes[2].scatter(frame.inference_ms_per_decision, indices, c=palette, s=33)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("ms par décision · échelle log")
    axes[2].set_title("Inférence", loc="left")
    for ax in axes:
        ax.grid(axis="x", alpha=.22)
    axes[0].set_yticks(indices, names)
    axes[0].invert_yaxis()
    save(fig, where, "05_stability_and_computing")

    seeds = pd.read_csv(output / "by_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), layout="constrained")
    for ax, metric, factor, label in zip(axes, ("objective_mad", "urgent_service_rate"), (1, 100),
                                        ("Perte moyenne (MAD)", "Urgences servies (%)"), strict=True):
        for training_seed, sub in seeds[seeds.metric.eq(metric)].groupby("training_seed"):
            data = sub.set_index("policy")["mean"].reindex(["B9", "B10"])
            ax.plot([0, 1], data * factor, marker="o", label=str(int(training_seed)))
        ax.set_xticks([0, 1], ["Q-Learning", "DQN continu"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.22)
    axes[1].legend(title="Graine", loc="best")
    fig.suptitle("Variabilité des checkpoints gelés", fontsize=13)
    save(fig, where, "06_learned_seed_variability")

    robust = pd.read_csv(output / "sensitivity_effects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.6), layout="constrained")
    for ax, metric, factor, label, scheme in zip(axes, ("objective_mad", "otif"), (1, 100),
                                                 ("Δ perte (MAD)", "Δ OTIF (points)"),
                                                 ("RdBu_r", "RdBu"), strict=True):
        table = robust[robust.metric.eq(metric)].pivot(index="contrast", columns="policy", values="difference") * factor
        table = table.reindex(list(FACTORS)).rename(index=FACTORS)
        heatmap(ax, table, label, label, scheme=scheme)
    fig.suptitle("Sensibilité au contexte · Un seul facteur varie", fontsize=13)
    save(fig, where, "07_sensitivity")

    link = pd.read_csv(output / "prediction_link.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for ax, metric, factor, label in zip(axes, ("objective_mad", "otif"), (1, 100),
                                        ("Δ perte (MAD)", "Δ OTIF (points)"), strict=True):
        ax.scatter(link.fixed_mae_difference_min, link[metric] * factor, color="#167e80", alpha=.78)
        ax.axvline(0, color="#5c6570", linestyle="--", linewidth=1)
        ax.axhline(0, color="#5c6570", linestyle="--", linewidth=1)
        ax.set(xlabel="Δ MAE de la prévision fixe (min)", ylabel=label)
    fig.suptitle("ML fixe moins historique · Erreur et résultat logistique", fontsize=13)
    save(fig, where, "08_accuracy_and_operations")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/experiments/final_benchmark"))
    render(parser.parse_args().output)


if __name__ == "__main__":
    main()
