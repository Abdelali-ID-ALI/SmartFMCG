"""Static heterogeneous delivery and pickup instance in integer units."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import minutes_since_midnight
from src.routing.base_provider import TravelMatrix

Load = tuple[int, int, int, int]
Window = tuple[int, int]
Grid = tuple[tuple[int | None, ...], ...]
ZERO: Load = (0, 0, 0, 0)


def nonnegative_integer(value: object) -> bool:
    """Accept only nonnegative integer scalars in the optimization model."""
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def valid_window(window: Window) -> bool:
    return (isinstance(window, tuple) and len(window) == 2
            and all(nonnegative_integer(value) for value in window)
            and window[0] <= window[1] <= 86400)


def valid_load(load: Load) -> bool:
    return isinstance(load, tuple) and len(load) == len(ZERO) and all(map(nonnegative_integer, load))


@dataclass(frozen=True)
class Stop:
    id: str
    delivery: Load
    pickup: Load
    window: Window
    service_s: int
    refrigerated: bool = False
    no_heavy: bool = False
    penalty_cents: int | None = None

    def __post_init__(self) -> None:
        if not valid_load(self.delivery) or not valid_load(self.pickup):
            raise ValueError("Loads must contain four nonnegative integers")
        if not valid_window(self.window) or not nonnegative_integer(self.service_s):
            raise ValueError("Invalid customer hours or service duration")
        if self.penalty_cents is not None and not nonnegative_integer(self.penalty_cents):
            raise ValueError("Optional penalty must be a nonnegative integer")
        if type(self.refrigerated) is not bool or type(self.no_heavy) is not bool:
            raise ValueError("Invalid customer flags")


@dataclass(frozen=True)
class Vehicle:
    id: str
    capacity: Load
    shift: Window
    fixed_cents: int
    cents_per_km: int
    cents_per_hour: int
    refrigerated: bool = False
    heavy: bool = False

    def __post_init__(self) -> None:
        if not valid_load(self.capacity) or not all(map(nonnegative_integer, (
            self.fixed_cents, self.cents_per_km, self.cents_per_hour
        ))):
            raise ValueError("Capacity and costs must be nonnegative")
        if not valid_window(self.shift):
            raise ValueError("Invalid shift")
        if type(self.refrigerated) is not bool or type(self.heavy) is not bool:
            raise ValueError("Invalid vehicle flags")

    def compatible(self, stop: Stop) -> bool:
        return (self.refrigerated or not stop.refrigerated) and (not self.heavy or not stop.no_heavy)


@dataclass(frozen=True)
class FMCGProblem:
    """Matrix index 0 denotes the depot, followed by stops in tuple order."""

    stops: tuple[Stop, ...]
    vehicles: tuple[Vehicle, ...]
    depot_window: Window
    distance_m: Grid
    travel_s: Grid

    def __post_init__(self) -> None:
        size = len(self.stops) + 1
        if not self.vehicles or len({v.id for v in self.vehicles}) != len(self.vehicles):
            raise ValueError("Expected a nonempty fleet with unique IDs")
        if len({s.id for s in self.stops}) != len(self.stops):
            raise ValueError("Expected unique customer IDs")
        if not valid_window(self.depot_window):
            raise ValueError("Invalid depot hours")
        for grid in (self.distance_m, self.travel_s):
            if len(grid) != size or any(len(row) != size for row in grid):
                raise ValueError("Incorrect matrix shape")
            if any(row[i] != 0 or any(value is not None and not nonnegative_integer(value) for value in row)
                   for i, row in enumerate(grid)):
                raise ValueError("Expected nonnegative costs and zero diagonal")
        if any((d is None) != (t is None) for d_row, t_row in zip(self.distance_m, self.travel_s, strict=True)
               for d, t in zip(d_row, t_row, strict=True)):
            raise ValueError("Distance and time must agree on unavailable arcs")
        if any(d is not None and d > 0 and t == 0
               for d_row, t_row in zip(self.distance_m, self.travel_s, strict=True)
               for d, t in zip(d_row, t_row, strict=True)):
            raise ValueError("Positive distance requires positive travel duration")

    def operating_window(self, vehicle: Vehicle) -> Window:
        return max(self.depot_window[0], vehicle.shift[0]), min(self.depot_window[1], vehicle.shift[1])


def _load_table(directory: Path, table: str, key: str, flags: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = pd.read_csv(directory / f"{table}.csv", dtype={key: str, **dict.fromkeys(flags, str)})
    for field in (key, *flags):
        if field not in frame:
            raise ValueError(f"{table}.csv missing {field}")
    if frame[key].isna().any() or not frame[key].is_unique or frame[key].str.strip().eq("").any():
        raise ValueError(f"{table}.csv requires unique nonempty {key}")
    for flag in flags:
        parsed = frame[flag].map({"True": True, "False": False})
        if parsed.isna().any():
            row = parsed.isna().idxmax() + 2
            raise ValueError(f"{table}.csv:{row}: {flag} must be True or False")
        frame[flag] = parsed
    return frame


def _scaled(value: object, scale: int, field: str, record: str, *, ceiling: bool = False) -> int:
    try:
        quantity = Decimal(str(value)) * scale
    except InvalidOperation as exc:
        raise ValueError(f"{record}: invalid {field}") from exc
    if not quantity.is_finite() or quantity < 0:
        raise ValueError(f"{record}: invalid {field}")
    integer = quantity.to_integral_value(rounding=ROUND_CEILING)
    if not ceiling and integer != quantity:
        raise ValueError(f"{record}: {field} exceeds supported unit precision")
    return int(integer)


def load_fmcg(matrix: TravelMatrix, directory: Path, optional_penalty_mad: int = 125) -> FMCGProblem:
    """Combine strictly parsed instance records with the ordered travel matrix."""
    depot = _load_table(directory, "depots", "depot_id").set_index("depot_id")
    customers = _load_table(directory, "customers", "customer_id")
    orders = _load_table(directory, "orders", "order_id", ("refrigerated_required", "mandatory"))
    fleet = _load_table(directory, "vehicles", "vehicle_id", ("refrigerated",))
    if (len(depot) != 1 or matrix.ids[0] not in depot.index
            or set(matrix.ids[1:]) != set(customers.customer_id)
            or orders.customer_id.isna().any() or not orders.customer_id.is_unique
            or set(orders.customer_id) != set(customers.customer_id)):
        raise ValueError("Matrix, customer and order IDs disagree")
    if not customers.vehicle_access_restrictions.isin(["any", "no_heavy"]).all():
        raise ValueError("Invalid customer access restriction")
    if not fleet.vehicle_type.isin(["van", "medium", "heavy", "refrigerated"]).all():
        raise ValueError("Invalid vehicle type")
    if (not orders.product_type.isin(["ambient_beverage", "chilled_juice"]).all()
            or not orders.refrigerated_required.eq(orders.product_type.eq("chilled_juice")).all()):
        raise ValueError("Invalid product or cold-chain requirement")
    visits = customers.merge(orders, on="customer_id", validate="one_to_one").set_index("customer_id")
    penalty = _scaled(optional_penalty_mad, 100, "optional_penalty_mad", "configuration")

    def clock(text: str) -> int:
        return minutes_since_midnight(text) * 60

    def load(row, prefix: str, record: str) -> Load:
        fields = [f"{prefix}_{name}" for name in ("weight", "volume", "crates", "pallets")]
        return tuple(_scaled(getattr(row, field, 0) if field == "pickup_pallets" else getattr(row, field),
                             scale, field, record)
                     for field, scale in zip(fields, (1000, 1000, 1, 1), strict=True))

    stops = tuple(
        Stop(s.Index, load(s, "delivery", s.Index), load(s, "pickup", s.Index),
             (clock(s.time_window_start), clock(s.time_window_end)),
             _scaled(s.service_duration, 60, "service_duration", s.Index, ceiling=True),
             s.refrigerated_required, s.vehicle_access_restrictions == "no_heavy",
             None if s.mandatory else penalty)
        for s in visits.loc[list(matrix.ids[1:])].itertuples()
    )
    vehicles = tuple(
        Vehicle(v.vehicle_id, load(v, "maximum", v.vehicle_id),
                (clock(v.shift_start), clock(v.shift_end)),
                _scaled(v.fixed_operating_cost, 100, "fixed_operating_cost", v.vehicle_id),
                _scaled(v.cost_per_km, 100, "cost_per_km", v.vehicle_id),
                _scaled(v.cost_per_hour, 100, "cost_per_hour", v.vehicle_id),
                v.refrigerated, v.vehicle_type == "heavy")
        for v in fleet.itertuples(index=False)
    )
    opening = depot.loc[matrix.ids[0]]
    distance, travel = integer_road_grids(matrix)
    return FMCGProblem(stops, vehicles, (clock(opening.opening_time), clock(opening.closing_time)),
                       distance, travel)


def integer_road_grids(matrix: TravelMatrix) -> tuple[Grid, Grid]:
    """Apply the planning units and directed connectivity contract to a road matrix."""
    matrix.validate()
    distance = tuple(tuple(round(value * 1000) if np.isfinite(value) else None for value in row)
                     for row in matrix.distance_km)
    # OSRM supplies seconds converted to minutes. Move one ULP toward zero
    # before ceiling to ignore floating-point noise at exact integer boundaries.
    def travel_seconds(value: float) -> int | None:
        if not np.isfinite(value):
            return None
        return 0 if value == 0 else max(1, int(np.ceil(np.nextafter(value * 60, 0))))

    travel = tuple(tuple(travel_seconds(value) for value in row) for row in matrix.duration_min)
    return distance, travel
