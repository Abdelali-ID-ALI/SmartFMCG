"""Paired synthetic demand, inventories and external draws independent of any plan."""

from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from hashlib import sha256
from math import ceil
import json

import numpy as np

from src.optimization.fmcg_problem import FMCGProblem, Load, Stop, ZERO
from src.planning.forecast import Forecasts
from src.planning.replay import actual_traffic
from src.simulation.experiment import beverage_load
from src.simulation.state import AMBIENT, CHILLED, Congestion, Order, Scenario, ServiceDelay, UrgentOrder, minus, plus

Reserve = tuple[tuple[Load, Load], ...]


def fingerprint(value) -> str:
    return sha256(json.dumps(value, default=sorted, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Design:
    demand: float
    pickup: float
    closing_shift_s: int
    reserve: int
    unavailable: str | None
    traffic_factor: float
    service_extra_s: int
    urgents: int
    urgent_window_s: int
    urgent_crates: tuple[int, int]


@dataclass(frozen=True)
class Draw:
    key: str
    day: date
    seed: int
    regime: str
    block: str
    variant: str
    design: Design
    penalty_cents: int
    latency_s: int

    def planning_key(self) -> tuple:
        d = self.design
        return self.day, self.seed, d.demand, d.pickup, d.closing_shift_s, d.reserve, d.unavailable


@dataclass(frozen=True)
class Case:
    draw: Draw
    problem: FMCGProblem
    reserves: Reserve
    scenario: Scenario
    traffic_fingerprint: str
    traffic_error: tuple[dict, ...]
    scenario_fingerprint: str

    def metadata(self) -> dict:
        return {"case": self.draw.key, "day": self.draw.day.isoformat(), "seed": self.draw.seed,
                "regime": self.draw.regime, "block": self.draw.block, "variant": self.draw.variant,
                "scenario_sha256": self.scenario_fingerprint, "traffic_sha256": self.traffic_fingerprint,
                "available_vehicles": len(self.problem.vehicles), "regular_orders": len(self.problem.stops),
                "delivery_crates": sum(s.delivery[2] for s in self.problem.stops),
                "pickup_crates": sum(s.pickup[2] for s in self.problem.stops),
                "urgent_total": self.draw.design.urgents, "reserve_each_product_crates": self.draw.design.reserve,
                "penalty_cents": self.draw.penalty_cents, "latency_s": self.draw.latency_s}


def build_draws(config: dict, profile: str, latency_s: int) -> tuple[Draw, ...]:
    plan = config["profiles"][profile]
    first = date.fromisoformat(plan["first_day"])
    sensitive_first = date.fromisoformat(plan["sensitivity_first_day"])
    days = tuple(first + timedelta(days=i) for i in range(plan["dates"]))
    sensitive_days = tuple(sensitive_first + timedelta(days=i) for i in range(plan["sensitivity_dates"]))
    if set(days) & set(sensitive_days) or min(*days, *sensitive_days) <= date(2026, 5, 31):
        raise ValueError("Benchmark and sensitivity dates must be disjoint and genuinely held out")

    def seed(day, slot, group):
        rng = np.random.default_rng(np.random.SeedSequence([plan["namespace"], day.toordinal(), slot, group]))
        return int(rng.integers(2**31))

    penalty = config["urgent_penalty_cents"]
    rows = []
    for day in days:
        for number, (regime, design) in enumerate(config["regimes"].items()):
            for repeat in range(plan["repeats"]):
                rows.append(Draw(f"{day}-{regime}-{repeat}", day, seed(day, 10 * number + repeat, 1),
                                 regime, "benchmark", "reference", Design(**design), penalty, latency_s))
    label = config["sensitivity"]["reference"]
    design = Design(**config["regimes"][label])
    for day in sensitive_days:
        reference = Draw(f"{day}-sensitivity-reference", day, seed(day, 0, 2), label,
                         "sensitivity", "reference", design, penalty, latency_s)
        rows.append(reference)
        for variant, adjustment in config["sensitivity"]["variants"].items():
            field, value = adjustment["field"], adjustment["value"]
            overrides = ({field: value} if field in {"urgent_penalty_cents", "latency_s"}
                         else {"design": replace(design, **{field: value})})
            if "urgent_penalty_cents" in overrides:
                overrides["penalty_cents"] = overrides.pop("urgent_penalty_cents")
            rows.append(replace(reference, key=f"{day}-sensitivity-{variant}", variant=variant, **overrides))
    seeds = [row.seed for row in rows if row.variant == "reference"]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Experimental draws unexpectedly share a seed")
    return tuple(rows)


def instance(base: FMCGProblem, draw: Draw) -> tuple[FMCGProblem, Reserve]:
    d = draw.design
    rng = np.random.default_rng(np.random.SeedSequence([draw.seed, 1001]))
    stops = []
    for stop in base.stops:
        delivered = max(1, round(stop.delivery[2] * d.demand * rng.uniform(.93, 1.07)))
        picked = max(0, round(stop.pickup[2] * d.pickup * rng.uniform(.86, 1.14)))
        pallets = ceil(delivered / 40) if stop.delivery[3] else 0
        delivery = (*beverage_load(delivered)[:3], pallets)
        pickup = (2000 * picked, 20 * picked, picked, 0)
        service = 480 + 60 * ceil(delivered / 6)
        window = stop.window[0], stop.window[1] + d.closing_shift_s
        if window[1] - window[0] < service:
            raise ValueError("Generated regular window shorter than service")
        stops.append(replace(stop, window=window, service_s=service, delivery=delivery, pickup=pickup))
    fleet = tuple(vehicle for vehicle in base.vehicles if vehicle.id != d.unavailable)
    if d.unavailable is not None and len(fleet) == len(base.vehicles):
        raise ValueError("Unknown unavailable vehicle")
    if sum(v.refrigerated for v in fleet) != 1 or not any(s.refrigerated and s.penalty_cents is None for s in stops):
        raise ValueError("Benchmark needs one refrigerated vehicle required by mandatory demand")
    reserve = beverage_load(d.reserve)
    reserved = tuple((reserve, reserve) if v.refrigerated else (ZERO, ZERO) for v in fleet)
    return replace(base, stops=tuple(stops), vehicles=fleet), reserved


def planning_capacity(problem: FMCGProblem, reserves: Reserve) -> FMCGProblem:
    """Reserve the identical physical cargo space before each alternative is planned."""
    vehicles = tuple(replace(v, capacity=minus(v.capacity, plus(*stock)))
                     for v, stock in zip(problem.vehicles, reserves, strict=True))
    return replace(problem, vehicles=vehicles)


def external_events(base: FMCGProblem, draw: Draw, ids, zones, service_s: int):
    rng = np.random.default_rng(np.random.SeedSequence([draw.seed, 1003]))
    traffic_at = int(rng.integers(9 * 3600, 11 * 3600 + 1800))
    zone = zones[ids[int(rng.integers(1, len(ids)))]]
    nodes = frozenset(i for i, item in enumerate(ids) if zones[item] == zone)
    arrival = traffic_at + int(rng.integers(1500, 3001))
    service_node = int(rng.integers(len(base.stops)))
    served = base.stops[service_node]
    service_at = served.window[0] + int(rng.integers(0, 901))
    extra = round(draw.design.service_extra_s * rng.uniform(.85, 1.15))
    events = []
    if draw.design.traffic_factor > 1:
        events.append(Congestion(traffic_at, arrival, nodes, draw.design.traffic_factor))
    if extra:
        events.append(ServiceDelay(service_at, served.id, extra))
    locations = rng.choice(len(base.stops), size=draw.design.urgents, replace=False)
    for number, index in enumerate(locations, start=1):
        site = base.stops[index]
        release = site.window[0] + int(rng.integers(-450, 1501))
        cold = bool(rng.random() < .35)
        low, high = draw.design.urgent_crates
        crates = int(rng.integers(low, high + 1))
        window = (max(site.window[0], release), release + draw.design.urgent_window_s)
        stop = Stop(f"U{number:02}", beverage_load(crates), ZERO, window, service_s, cold,
                    site.no_heavy, draw.penalty_cents)
        events.append(UrgentOrder(release, Order(stop, int(index) + 1, CHILLED if cold else AMBIENT, True)))
    return tuple(sorted(events, key=lambda event: event.at_s))


def traffic_realization(base: FMCGProblem, forecasts: Forecasts, draw: Draw, ids, zones, hour: int):
    actual, incidents = actual_traffic(forecasts.arcs, forecasts.static, draw.day, draw.seed, ids, zones)
    grid = [[list(row) for row in base.travel_s] for _ in range(24)]
    for table, values in zip(grid, np.ceil(actual * 60).astype(int), strict=True):
        for (i, j), duration in zip(forecasts.arcs, values, strict=True):
            table[i][j] = int(duration)
    mask = np.zeros(actual.shape, dtype=bool)
    mask[7:18] = True
    error = []
    for label, values in (("static", forecasts.static), ("historical_fixed", forecasts.historical[hour]),
                          ("ml_fixed", forecasts.ml[hour]), ("historical_hourly", forecasts.historical),
                          ("ml_hourly", forecasts.ml)):
        residual = np.broadcast_to(values, actual.shape) - actual
        for group, selected in (("all", mask), ("incident", mask & incidents), ("ordinary", mask & ~incidents)):
            r = residual[selected]
            error.append({"prediction": label, "traffic_group": group, "arcs_hours": len(r),
                          "mae_min": float(np.abs(r).mean()), "rmse_min": float(np.sqrt(np.square(r).mean())),
                          "bias_min": float(r.mean())})
    return tuple(tuple(map(tuple, table)) for table in grid), tuple(error)


def make_case(base: FMCGProblem, draw: Draw, config: dict, ids, zones, realized,
              traffic_hash: str, error: tuple[dict, ...]) -> Case:
    problem, reserves = instance(base, draw)
    events = external_events(base, draw, ids, zones, config["urgent_service_s"])
    signature = fingerprint({"problem": asdict(problem), "reserves": reserves,
                             "events": [asdict(event) for event in events], "traffic": traffic_hash,
                             "latency_s": draw.latency_s})
    return Case(draw, problem, reserves, Scenario(events, realized), traffic_hash, error, signature)
