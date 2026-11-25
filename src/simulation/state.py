"""Physical inputs and immutable information available at a dispatch decision."""

from dataclasses import dataclass
from math import ceil, isfinite

from src.optimization.fmcg_problem import Grid, Load, Stop, nonnegative_integer

AMBIENT = "ambient_beverage"
CHILLED = "chilled_juice"
PRODUCTS = (AMBIENT, CHILLED)


def plus(left: Load, right: Load) -> Load:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def minus(left: Load, right: Load) -> Load:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def within(load: Load, capacity: Load) -> bool:
    return all(0 <= quantity <= limit for quantity, limit in zip(load, capacity, strict=True))


def congested_seconds(duration_s: int, factor: float) -> int:
    """Round a congestion-adjusted duration up to an operational second."""
    try:
        return ceil(duration_s * factor)
    except OverflowError as exc:
        raise ValueError("Congestion produces a nonfinite travel duration") from exc


@dataclass(frozen=True)
class Order:
    stop: Stop
    node: int
    product: str
    priority: bool = False
    owner: int | None = None

    def __post_init__(self) -> None:
        if (not nonnegative_integer(self.node) or self.node == 0 or self.product not in PRODUCTS
                or self.stop.refrigerated != (self.product == CHILLED)):
            raise ValueError("Invalid order location or product")


@dataclass(frozen=True)
class Congestion:
    at_s: int
    end_s: int
    nodes: frozenset[int]
    factor: float

    def __post_init__(self) -> None:
        if (not all(map(nonnegative_integer, (self.at_s, self.end_s))) or self.end_s <= self.at_s
                or not self.nodes or not isfinite(self.factor) or self.factor < 1):
            raise ValueError("Invalid congestion")


@dataclass(frozen=True)
class ServiceDelay:
    at_s: int
    order_id: str
    extra_s: int

    def __post_init__(self) -> None:
        if not all(map(nonnegative_integer, (self.at_s, self.extra_s))) or not self.extra_s:
            raise ValueError("Invalid service delay")


@dataclass(frozen=True)
class UrgentOrder:
    at_s: int
    order: Order

    def __post_init__(self) -> None:
        if (not nonnegative_integer(self.at_s) or self.order.owner is not None
                or self.order.stop.penalty_cents is None or not any(self.order.stop.delivery)
                or not self.at_s <= self.order.stop.window[1]):
            raise ValueError("Expected a new optional order with a positive demand")


Event = Congestion | ServiceDelay | UrgentOrder


@dataclass(frozen=True)
class VehicleView:
    index: int
    phase: str
    node: int
    destination: int | None
    entered_s: int
    committed: str | None
    tail: tuple[str, ...]
    # The anchor is the forecast after completing the committed stop, if any.
    anchor_node: int
    anchor_s: int
    regular: Load
    reserves: tuple[Load, Load]
    collected: Load

    @property
    def mutable(self) -> bool:
        return self.phase not in {"returning", "returned", "unused"}

    @property
    def onboard(self) -> Load:
        return plus(plus(self.regular, self.collected), plus(*self.reserves))


@dataclass(frozen=True)
class Observation:
    now_s: int
    event_types: tuple[str, ...]
    tick: bool
    vehicles: tuple[VehicleView, ...]
    completed: frozenset[str]
    waiting_urgent: tuple[str, ...]
    predicted_delay_s: int
    predicted_overtime_s: int
    at_risk: int
    priority_at_risk: int
    congestion_factor: float
    since_replan_s: int | None
    replan_available: bool


@dataclass(frozen=True)
class Scenario:
    events: tuple[Event, ...]
    actual_s: tuple[Grid, ...]


@dataclass(frozen=True)
class Settings:
    tick_s: int
    latency_s: int
    candidate_limit: int
    delay_cents_per_min: int
    change_cents: int
    overtime_cents_per_min: int

    def __post_init__(self) -> None:
        if (not all(map(nonnegative_integer, (self.tick_s, self.latency_s, self.candidate_limit,
                                              self.delay_cents_per_min, self.change_cents,
                                              self.overtime_cents_per_min)))
                or min(self.tick_s, self.latency_s, self.candidate_limit) == 0):
            raise ValueError("Invalid simulation settings")
