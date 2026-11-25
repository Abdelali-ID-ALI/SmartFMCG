"""Event-driven distribution execution with an explicit decision/step boundary."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from types import MappingProxyType
from time import perf_counter

from src.optimization.fmcg_problem import FMCGProblem, Load, ZERO, nonnegative_integer
from src.optimization.fmcg_validator import Tour, validate_fmcg
from src.simulation.policies import Action
from src.simulation.replanning import Forecast, Proposal, Routes, affected, project, propose, route_changes
from src.simulation.state import (PRODUCTS, Congestion, Event, Observation, Order, Scenario, ServiceDelay,
                                  Settings, UrgentOrder, VehicleView, congested_seconds, minus, plus)
from src.simulation.validation import Holdings, Movement, validate_movements, validate_routes


@dataclass
class VehicleState:
    phase: str = "unused"
    node: int = 0
    destination: int | None = None
    entered_s: int = 0
    due_s: int = 0
    departure_s: int | None = None
    predicted_arrival_s: int = 0
    committed: str | None = None
    tail: tuple[str, ...] = ()
    holdings: Holdings = (ZERO, ZERO, ZERO, ZERO)
    distance_m: int = 0


@dataclass
class Decision:
    time_s: int
    events: tuple[str, ...]
    tick: bool
    requested: str
    executed: str = "KEEP"
    status: str = "kept"
    planning_wall_s: float = 0.
    modeled_latency_s: int = 0
    candidates: int = 0
    changes: int = 0
    applied_s: int | None = None
    predicted_delay_s: int = 0
    waiting_urgent: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pending:
    routes: Routes
    decision: Decision


@dataclass
class Simulator:
    problem: FMCGProblem
    tours: tuple[Tour, ...]
    products: Mapping[str, str]
    priority: frozenset[str]
    starting_reserves: tuple[tuple[Load, Load], ...]
    scenario: Scenario
    settings: Settings
    movements: list[Movement] = field(default_factory=list, init=False)
    decisions: list[Decision] = field(default_factory=list, init=False)
    planning_details: dict[int, tuple[int, str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        validate_fmcg(self.problem, self.tours)
        self.now_s, self._close_s = self.problem.depot_window
        self._queue = []
        self._serial = 0
        self._pending = None
        self._observation = None
        self._last_replan = None
        self._alerts = ()
        self._delays = {}
        self._completed = {}
        self._released = {}
        self._fleet = [VehicleState() for _ in self.problem.vehicles]
        self._orders = {
            stop.id: Order(stop, i, self.products[stop.id], stop.id in self.priority)
            for i, stop in enumerate(self.problem.stops, 1)
        }
        if len(self.starting_reserves) != len(self._fleet):
            raise ValueError("Expected one reserve per vehicle")
        for tour in self.tours:
            tail = tuple(self.problem.stops[i - 1].id for i in tour.stops)
            assigned = [self._orders[key].stop.delivery for key in tail]
            regular = tuple(map(sum, zip(*assigned, strict=True)))
            state = self._fleet[tour.vehicle]
            state.phase, state.tail, state.due_s = "staged", tail, tour.departure_s
            state.entered_s = self.now_s
            state.holdings = (regular, *self.starting_reserves[tour.vehicle], ZERO)
            for key in tail:
                self._orders[key] = replace(self._orders[key], owner=tour.vehicle)
            self._push(tour.departure_s, "ready", tour.vehicle)
        if any(state.phase == "unused" and reserves != (ZERO, ZERO)
               for state, reserves in zip(self._fleet, self.starting_reserves, strict=True)):
            raise ValueError("Reserves must be loaded on an active initial tour")
        self._initial = tuple(state.holdings for state in self._fleet)
        known = set(self._orders)
        urgent = set()
        for event in self.scenario.events:
            if not self.now_s <= event.at_s <= self._close_s:
                raise ValueError("External event outside the distribution day")
            if isinstance(event, Congestion):
                if any(not nonnegative_integer(i) or i > len(self.problem.stops) for i in event.nodes):
                    raise ValueError("Unknown congestion location")
            elif isinstance(event, ServiceDelay):
                if event.order_id not in known:
                    raise ValueError("Unknown regular order in service delay")
            else:
                key = event.order.stop.id
                if key in known | urgent or event.order.node > len(self.problem.stops):
                    raise ValueError("Unknown urgent location or duplicate order ID")
                if self.problem.stops[event.order.node - 1].no_heavy and not event.order.stop.no_heavy:
                    raise ValueError("Urgent order must inherit site access restrictions")
                urgent.add(key)
            self._push(event.at_s, "external", event)
        if len(self.scenario.actual_s) != 24:
            raise ValueError("Expected 24 hourly grids of actual travel times")
        for grid in self.scenario.actual_s:
            if len(grid) != len(self.problem.travel_s):
                raise ValueError("Invalid realization dimensions")
            for row, nominal in zip(grid, self.problem.travel_s, strict=True):
                if len(row) != len(nominal) or any(
                    (x is None) != (y is None) or (x is not None and (
                        not nonnegative_integer(x) or (x > 0) != (y > 0)
                    )) for x, y in zip(row, nominal, strict=True)
                ):
                    raise ValueError("Invalid realized travel duration or connectivity")
        for tick in range(self.now_s, self._close_s + 1, self.settings.tick_s):
            self._push(tick, "tick")
        validate_movements(self.problem, self._initial, (), {}, {})
        views = self._views()
        validate_routes(self._forecast(), views, tuple(view.tail for view in views),
                        self.now_s, frozenset(), check_windows=False)

    def _push(self, when: int, kind: str, payload=None) -> None:
        self._serial += 1
        heappush(self._queue, (when, self._serial, kind, payload))

    def _forecast(self) -> Forecast:
        return Forecast(self.problem, MappingProxyType(self._orders), self._alerts, MappingProxyType(self._delays))

    def dispatch_forecast(self) -> Forecast:
        """Snapshot forecasts and announcements known now, independent of realized traffic."""
        return Forecast(self.problem, MappingProxyType(dict(self._orders)), self._alerts,
                        MappingProxyType(dict(self._delays)))

    def _views(self) -> tuple[VehicleView, ...]:
        views = []
        forecast = self._forecast()
        for i, state in enumerate(self._fleet):
            node, when = state.node, self.now_s
            if state.phase == "staged":
                when = max(when, state.due_s)
            if state.committed is not None:
                key = state.committed
                stop = self._orders[key].stop
                node = self._orders[key].node
                if state.phase == "service":
                    when = max(when, state.entered_s + forecast.service(key))
                else:
                    arrival = max(when, state.predicted_arrival_s) if state.phase == "travel" else when
                    when = max(arrival, stop.window[0]) + forecast.service(key)
            regular, ambient, chilled, collected = state.holdings
            views.append(VehicleView(i, state.phase, state.node, state.destination, state.entered_s,
                                     state.committed, state.tail, node, when,
                                     regular, (ambient, chilled), collected))
        return tuple(views)

    def _waiting_urgent(self) -> tuple[str, ...]:
        allocated = {key for state in self._fleet for key in (*state.tail, state.committed)}
        return tuple(key for key in self._released if key not in self._completed and key not in allocated
                     and self._orders[key].stop.window[1] >= self.now_s)

    def _record(self, i: int, kind: str, order_id: str | None = None) -> None:
        state = self._fleet[i]
        self.movements.append(Movement(i, kind, self.now_s, state.node, order_id, state.holdings,
                                       state.distance_m, state.destination))

    def _dispatch(self) -> None:
        if self._pending:
            return
        for i, state in enumerate(self._fleet):
            if state.phase == "staged" and state.due_s <= self.now_s:
                state.departure_s = self.now_s
                state.phase = "idle"
                self._record(i, "depart")
            if state.phase != "idle":
                continue
            key = state.tail[0] if state.tail else None
            node = self._orders[key].node if key is not None else 0
            base = self.scenario.actual_s[min(23, self.now_s // 3600)][state.node][node]
            factor = max((alert.factor for alert in self._alerts if affected(alert, state.node, node, self.now_s)),
                         default=1.)
            # The realized completion remains private. A released alert affects new legs only.
            state.due_s = self.now_s + congested_seconds(base, factor)
            state.predicted_arrival_s = self.now_s + self._forecast().travel(state.node, node, self.now_s)
            state.destination, state.committed, state.entered_s = node, key, self.now_s
            if key is not None:
                state.tail = state.tail[1:]
            state.phase = "returning" if key is None else "travel"
            self._record(i, "leg", key)
            self._push(state.due_s, "arrive", i)

    def _transition(self, i: int, kind: str) -> None:
        state = self._fleet[i]
        key = state.committed
        if kind == "arrive":
            state.distance_m += self.problem.distance_m[state.node][state.destination]
            state.node, state.destination = state.destination, None
            self._record(i, kind, key)
            state.entered_s = self.now_s
            if key is None:
                state.holdings = (ZERO, ZERO, ZERO, ZERO)
                state.phase = "returned"
                self._record(i, "unload")
            else:
                state.phase = "waiting"
                start = max(self.now_s, self._orders[key].stop.window[0])
                self._push(start, "service_start", i)
        elif kind == "service_start":
            state.phase, state.entered_s = "service", self.now_s
            state.due_s = self.now_s + self._forecast().service(key)
            self._record(i, kind, key)
            self._push(state.due_s, "service_end", i)
        elif state.phase == "service" and state.due_s == self.now_s:
            order = self._orders[key]
            compartment = 0 if order.owner is not None else 1 + PRODUCTS.index(order.product)
            cargo = list(state.holdings)
            cargo[compartment] = minus(cargo[compartment], order.stop.delivery)
            cargo[3] = plus(cargo[3], order.stop.pickup)
            state.holdings = tuple(cargo)
            self._completed[key] = self.now_s
            self._record(i, kind, key)
            state.phase, state.committed, state.entered_s = "idle", None, self.now_s

    def _external(self, event: Event) -> str:
        if isinstance(event, Congestion):
            self._alerts = (*self._alerts, event)
            return "congestion"
        if isinstance(event, UrgentOrder):
            key = event.order.stop.id
            self._orders[key] = event.order
            self._released[key] = self.now_s
            return "urgent_order"
        key = event.order_id
        if key not in self._completed:
            self._delays[key] = self._delays.get(key, 0) + event.extra_s
            for i, state in enumerate(self._fleet):
                if state.phase == "service" and state.committed == key:
                    state.due_s += event.extra_s
                    self._push(state.due_s, "service_end", i)
        return "service_delay"

    def _apply(self) -> None:
        routes, decision = self._pending.routes, self._pending.decision
        self._pending = None
        current = tuple(state.tail for state in self._fleet)
        if routes == current:
            if decision.status == "pending":
                decision.status = "no_improvement"
            return
        checked_at = perf_counter()
        try:
            validate_routes(self._forecast(), self._views(), routes, self.now_s,
                            frozenset(self._waiting_urgent()))
        except ValueError:
            decision.status = "invalid_at_application"
            return
        finally:
            decision.planning_wall_s += perf_counter() - checked_at
        decision.changes = sum(route_changes(before, after) for before, after in zip(current, routes, strict=True))
        for state, route in zip(self._fleet, routes, strict=True):
            state.tail = route
        decision.executed, decision.status = decision.requested, "applied"
        decision.applied_s = self._last_replan = self.now_s

    def _can_plan(self, views: tuple[VehicleView, ...]) -> bool:
        release_s = self.now_s + self.settings.latency_s
        return (self._pending is None and release_s <= self._close_s and any(view.mutable for view in views)
                and all(view.phase != "staged" or release_s <= self.problem.operating_window(
                    self.problem.vehicles[view.index])[1] for view in views))

    def next_observation(self) -> Observation | None:
        """Advance to a public decision opportunity or to terminal execution."""
        if self._observation is not None:
            raise RuntimeError("Act before advancing the decision clock")
        while self._queue:
            self.now_s = self._queue[0][0]
            external, tick, release = [], False, False
            while self._queue and self._queue[0][0] == self.now_s:
                batch = []
                while self._queue and self._queue[0][0] == self.now_s:
                    batch.append(heappop(self._queue))
                for _, _, kind, payload in batch:
                    if kind in {"arrive", "service_start", "service_end"}:
                        self._transition(payload, kind)
                    elif kind == "release":
                        release = True
                for _, _, kind, payload in batch:
                    if kind == "external":
                        external.append(payload)
                    elif kind == "tick":
                        tick = True
            events = tuple(self._external(event) for event in external)
            if release:
                self._apply()
            if events or tick:
                views, forecast = self._views(), self._forecast()
                projections = [project(forecast, view, view.tail, self.now_s) for view in views if view.mutable]
                if any(projection is None for projection in projections):
                    raise RuntimeError("Physically invalid plan before decision")
                risks = [(key, delay) for projection in projections for key, delay in projection.delays if delay]
                self._observation = Observation(
                    self.now_s, events, tick, views, frozenset(self._completed), self._waiting_urgent(),
                    sum(delay for _, delay in risks), sum(p.overtime_s for p in projections), len(risks),
                    sum(self._orders[key].priority for key, _ in risks),
                    max((a.factor for a in self._alerts if self.now_s < a.end_s), default=1.),
                    None if self._last_replan is None else self.now_s - self._last_replan, self._can_plan(views),
                )
                return self._observation
            self._dispatch()
        validate_movements(self.problem, self._initial, tuple(self.movements), self._orders, self._released, self.scenario)
        return None

    def act(self, action: Action, *, planner: Callable[..., Proposal] | None = None) -> None:
        obs = self._observation
        if obs is None or not isinstance(action, Action):
            raise ValueError("Expected a valid action at a decision opportunity")
        if action == Action.LOCAL_REOPTIMIZATION and planner is None:
            raise ValueError("Local reoptimization requires a suffix optimizer")
        decision = Decision(self.now_s, obs.event_types, obs.tick, action.name,
                            predicted_delay_s=obs.predicted_delay_s, waiting_urgent=obs.waiting_urgent)
        self.decisions.append(decision)
        self._observation = None
        available_s = self.now_s + self.settings.latency_s
        if action != Action.KEEP:
            if self._pending:
                decision.status = "busy"
            elif not obs.replan_available:
                decision.status = "unavailable"
            else:
                begin = perf_counter()
                forecast = self._forecast()
                proposal = (planner or propose)(forecast, obs.vehicles, obs.waiting_urgent, action,
                                                 available_s, self.settings)
                if proposal.solver_calls or proposal.solver_status:
                    self.planning_details[len(self.decisions) - 1] = proposal.solver_calls, proposal.solver_status
                routes = proposal.routes
                try:
                    validate_routes(forecast, obs.vehicles, routes, available_s, frozenset(obs.waiting_urgent))
                except ValueError:
                    routes = tuple(view.tail for view in obs.vehicles)
                    decision.status = "invalid_candidate"
                else:
                    decision.status = "pending"
                decision.planning_wall_s = perf_counter() - begin
                decision.modeled_latency_s = self.settings.latency_s
                decision.candidates = proposal.candidates
                self._pending = Pending(routes, decision)
                self._push(available_s, "release")
        self._dispatch()

    @property
    def completed(self) -> dict[str, int]:
        return dict(self._completed)

    @property
    def orders(self) -> dict[str, Order]:
        return dict(self._orders)
