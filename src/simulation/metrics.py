"""Terminal KPIs and a business loss independent of the simulator and policies."""

from dataclasses import dataclass

from src.optimization.fmcg_validator import UNITS_PER_MAD, operating_cost
from src.simulation.simulator import Simulator


@dataclass(frozen=True)
class Costs:
    fixed_mad: float
    distance_mad: float
    time_mad: float
    omissions_mad: float
    delay_min: float
    overtime_min: float
    changed_predecessors: int

    @property
    def operating_mad(self) -> float:
        return self.fixed_mad + self.distance_mad + self.time_mad

    @property
    def realized_mad(self) -> float:
        return self.operating_mad + self.omissions_mad


def reward_delta(before: Costs, after: Costs, *, delay_mad_per_min: float, overtime_mad_per_min: float,
                 change_mad: float) -> float:
    """Return the negative incremental business loss for any accounting interval."""
    return -(after.realized_mad - before.realized_mad
             + delay_mad_per_min * (after.delay_min - before.delay_min)
             + overtime_mad_per_min * (after.overtime_min - before.overtime_min)
             + change_mad * (after.changed_predecessors - before.changed_predecessors))


def evaluate(sim: Simulator) -> dict[str, float]:
    orders, finished = sim.orders, sim.completed
    delays = {key: max(0, end - orders[key].stop.window[1]) for key, end in finished.items()}
    urgent = orders.keys() - sim.products.keys()
    regular = orders.keys() - urgent
    priority = {key for key, order in orders.items() if order.priority}
    on_time = {key for key, delay in delays.items() if not delay}
    missed = orders.keys() - finished.keys()
    departures = {entry.vehicle: entry for entry in sim.movements if entry.kind == "depart"}
    returns = {entry.vehicle: entry for entry in sim.movements if entry.kind == "unload"}
    rows = [(sim.problem.vehicles[i], end.distance_m, end.time_s - departures[i].time_s)
            for i, end in returns.items()]
    totals = tuple(map(sum, zip(*(operating_cost(vehicle, meters, duration)
                                  for vehicle, meters, duration in rows), strict=True))) if rows else (0, 0, 0)
    overtime = sum(max(0, entry.time_s - sim.problem.operating_window(sim.problem.vehicles[i])[1])
                   for i, entry in returns.items())
    missed_mandatory = sum(orders[key].stop.penalty_cents is None for key in missed)
    changed = sum(d.changes for d in sim.decisions)
    ready = {tour.vehicle: tour.departure_s for tour in sim.tours}
    hold_s = 0
    predeparture_hold_s = 0
    for entry in sim.movements:
        if entry.kind == "depart":
            predeparture_hold_s += entry.time_s - ready.pop(entry.vehicle)
        if entry.kind == "service_end":
            ready[entry.vehicle] = entry.time_s
        if entry.kind == "leg" and entry.vehicle in ready:
            hold_s += entry.time_s - ready.pop(entry.vehicle)
    costs = Costs(*(total / UNITS_PER_MAD for total in totals),
                  sum(orders[key].stop.penalty_cents or 0 for key in missed) / 100,
                  sum(delays.values()) / 60, overtime / 60, changed)

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.

    recoverable = sum(order.stop.pickup[2] for order in orders.values())
    recovered = sum(orders[key].stop.pickup[2] for key in finished)
    return {
        "realized_cost_mad": costs.realized_mad, "operating_cost_mad": costs.operating_mad,
        "business_loss_mad": -reward_delta(Costs(0., 0., 0., 0., 0., 0., 0), costs,
                                           delay_mad_per_min=sim.settings.delay_cents_per_min / 100,
                                           overtime_mad_per_min=sim.settings.overtime_cents_per_min / 100,
                                           change_mad=sim.settings.change_cents / 100),
        "fixed_mad": costs.fixed_mad, "distance_mad": costs.distance_mad, "time_mad": costs.time_mad,
        "omission_mad": costs.omissions_mad, "distance_km": sum(meters for _, meters, _ in rows) / 1000,
        "vehicle_duration_min": sum(duration for _, _, duration in rows) / 60,
        "vehicles": len(rows), "initial_weight_utilization": ratio(sum(
            sum(load[0] for load in entry.holdings) / max(1, sim.problem.vehicles[i].capacity[0])
            for i, entry in departures.items()), len(rows)),
        "otif": ratio(len(on_time), len(orders)), "regular_otif": ratio(len(on_time & regular), len(regular)),
        "urgent_otif": ratio(len(on_time & urgent), len(urgent)),
        "priority_otif": ratio(len(on_time & priority), len(priority)),
        "urgent_total": len(urgent), "urgent_served": len(finished.keys() & urgent),
        "late": len(set(delays) - on_time), "delay_min": costs.delay_min,
        "max_delay_min": max(delays.values(), default=0) / 60, "overtime_min": costs.overtime_min,
        "unserved": len(missed), "unserved_mandatory": missed_mandatory,
        "realized_feasible": int(not costs.delay_min and not overtime and not missed_mandatory),
        "recovered_crates": recovered, "packaging_recovery": ratio(recovered, recoverable),
        "opportunities": len(sim.decisions), "replan_attempts": sum(d.modeled_latency_s > 0 for d in sim.decisions),
        "replans_applied": sum(d.executed != "KEEP" for d in sim.decisions),
        "route_changes": changed, "candidates": sum(d.candidates for d in sim.decisions),
        "planning_wall_s": sum(d.planning_wall_s for d in sim.decisions),
        "modeled_latency_s": sum(d.modeled_latency_s for d in sim.decisions),
        "dispatch_hold_s": hold_s, "predeparture_hold_s": predeparture_hold_s,
    }
