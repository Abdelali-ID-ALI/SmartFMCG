"""Dispatch actions and a CP-SAT suffix optimizer for one affected vehicle."""

from enum import IntEnum
from functools import partial

from ortools.sat.python import cp_model

from src.optimization.fmcg_problem import ZERO
from src.optimization.fmcg_validator import UNITS_PER_CENT
from src.simulation.policies import Action
from src.simulation.replanning import Forecast, Proposal, project, route_changes
from src.simulation.state import PRODUCTS, Settings, VehicleView, minus, plus, within


class DispatchAction(IntEnum):
    KEEP_PLAN = 0
    LOCAL_HEURISTIC = 1
    LOCAL_REOPTIMIZATION = 2
    GLOBAL_REOPTIMIZATION = 3


ENGINE_ACTIONS = (Action.KEEP, Action.LOCAL, Action.LOCAL_REOPTIMIZATION, Action.GLOBAL)
ACTION_NAMES = {action.name: dispatch.name for action, dispatch in zip(ENGINE_ACTIONS, DispatchAction, strict=True)}


def usable_stock(forecast: Forecast, view: VehicleView, product: str):
    stock = view.reserves[PRODUCTS.index(product)]
    for key in (*((view.committed,) if view.committed else ()), *view.tail):
        order = forecast.orders[key]
        if order.owner is None and order.product == product:
            stock = minus(stock, order.stop.delivery)
    return stock


def can_supply(forecast: Forecast, view: VehicleView, key: str) -> bool:
    order = forecast.orders[key]
    return (view.mutable and forecast.problem.vehicles[view.index].compatible(order.stop)
            and within(order.stop.delivery, usable_stock(forecast, view, order.product)))


def optimizer(budget: float):
    return partial(local_reoptimization, budget=budget)


def local_reoptimization(forecast: Forecast, views: tuple[VehicleView, ...], waiting: tuple[str, ...],
                         action: Action, release_s: int, settings: Settings, *, budget: float) -> Proposal:
    original = tuple(view.tail for view in views)
    candidates = []
    for view in views:
        if view.mutable:
            urgent = next((key for key in waiting if can_supply(forecast, view, key)), None)
            if not view.tail and urgent is None:
                continue
            incumbent = project(forecast, view, view.tail, release_s)
            pressure = sum(delay for _, delay in incumbent.delays) + incumbent.overtime_s
            candidates.append((urgent is not None, pressure, -view.index, view, urgent, incumbent))
    if not candidates:
        return Proposal(original, 0, solver_status="no_local_work")
    _, _, _, view, urgent, incumbent = max(candidates, key=lambda row: row[:3])
    if len(view.tail) < 2 and urgent is None:
        return Proposal(original, 0, solver_status="too_few_stops")

    keys = (*view.tail, *((urgent,) if urgent else ()))
    truck = forecast.problem.vehicles[view.index]
    start = max(view.anchor_s, release_s)
    close = forecast.problem.operating_window(truck)[1]
    if start > close:
        return Proposal(original, 0, solver_status="shift_ended")

    model = cp_model.CpModel()
    end = len(keys) + 1
    locations = (view.anchor_node, *(forecast.orders[key].node for key in keys), 0)
    stops = [forecast.orders[key].stop for key in keys]
    times = [model.new_constant(start), *(model.new_int_var(0, close, f"finish_{i}") for i in range(1, end + 1))]
    initial = view.onboard
    if view.committed:
        stop = forecast.orders[view.committed].stop
        initial = plus(minus(initial, stop.delivery), stop.pickup)
    loads = [[model.new_constant(q) for q in initial]]
    loads.extend([model.new_int_var(0, limit, f"load_{i}_{d}") for d, limit in enumerate(truck.capacity)]
                 for i in range(1, end))
    fixed = model.new_constant(1)
    circuit = [(end, 0, fixed)]
    arcs = {}
    objective = [times[end] * truck.cents_per_hour * 5]
    if urgent:
        skip = model.new_bool_var("skip_urgent")
        circuit.append((end - 1, end - 1, skip))
        objective.append(skip * forecast.orders[urgent].stop.penalty_cents * UNITS_PER_CENT)
    for j, stop in enumerate(stops, 1):
        if not truck.compatible(stop) or stop.window[0] + forecast.service(keys[j - 1]) > stop.window[1]:
            if urgent and j == end - 1:
                model.add(skip == 1)
            else:
                return Proposal(original, 0, solver_status="infeasible_window")
        else:
            lower = model.add(times[j] >= stop.window[0] + forecast.service(keys[j - 1]))
            upper = model.add(times[j] <= stop.window[1])
            if urgent and j == end - 1:
                lower.only_enforce_if(~skip)
                upper.only_enforce_if(~skip)
    old_predecessor = dict(zip(view.tail, (None, *view.tail)))
    for i in range(end):
        for j in range(1, end + 1):
            if i == j:
                continue
            origin, destination = locations[i], locations[j]
            drive = forecast.travel(origin, destination, start)
            if drive is None:
                continue
            edge = model.new_bool_var(f"arc_{i}_{j}")
            arcs[i, j] = edge
            circuit.append((i, j, edge))
            duration = 0 if j == end else forecast.service(keys[j - 1])
            # Announced factors are nonincreasing after the solve anchor. This
            # freezes a conservative transit duration over the remaining shift.
            model.add(times[j] >= times[i] + drive + duration).only_enforce_if(edge)
            distance = forecast.problem.distance_m[origin][destination]
            change = 0
            if j != end:
                before = None if i == 0 else keys[i - 1]
                change = keys[j - 1] not in old_predecessor or old_predecessor[keys[j - 1]] != before
                delta = minus(stops[j - 1].pickup, stops[j - 1].delivery)
                for d, quantity in enumerate(delta):
                    model.add(loads[j][d] == loads[i][d] + quantity).only_enforce_if(edge)
            objective.append(edge * (distance * truck.cents_per_km * 18
                                     + change * settings.change_cents * UNITS_PER_CENT))
    model.add_circuit(circuit)
    model.minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_deterministic_time = budget
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Proposal(original, len(arcs), 1, status_name)
    successors = {i: j for (i, j), edge in arcs.items() if solver.boolean_value(edge)}
    route, cursor = [], successors[0]
    while cursor != end:
        route.append(keys[cursor - 1])
        cursor = successors[cursor]
    tail = tuple(route)
    proposal = project(forecast, view, tail, release_s)

    def score(report, sequence):
        return (report.cost_units + (settings.delay_cents_per_min * sum(d for _, d in report.delays)
                                    + settings.overtime_cents_per_min * report.overtime_s) * (UNITS_PER_CENT // 60)
                + (settings.change_cents * route_changes(view.tail, sequence)
                   - (forecast.orders[urgent].stop.penalty_cents if urgent and urgent in sequence else 0))
                * UNITS_PER_CENT)

    if proposal is None or score(proposal, tail) >= score(incumbent, view.tail):
        return Proposal(original, len(arcs), 1, status_name)
    changed = (*original[:view.index], tail, *original[view.index + 1:])
    return Proposal(changed, len(arcs), 1, status_name)
