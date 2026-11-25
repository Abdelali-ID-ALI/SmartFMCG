"""Directed CVRP with one depot, homogeneous vehicles and weight capacity."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import csv

import numpy as np

from src.routing.base_provider import TravelMatrix

Routes = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CVRP:
    """Customer indices start at 1. Index 0 is the depot."""

    ids: tuple[str, ...]
    demand_kg: tuple[int, ...]
    distance_m: tuple[tuple[int | None, ...], ...]
    vehicle_count: int
    capacity_kg: int

    def __post_init__(self) -> None:
        size = len(self.ids)
        if size < 2 or len(set(self.ids)) != size or len(self.demand_kg) != size:
            raise ValueError("Expected one depot and uniquely identified customers")
        if self.demand_kg[0] != 0 or any(demand <= 0 for demand in self.demand_kg[1:]):
            raise ValueError("Expected zero depot demand and positive customer demands")
        if self.vehicle_count < 1 or self.capacity_kg < 1:
            raise ValueError("Fleet size and capacity must be positive")
        if len(self.distance_m) != size or any(len(row) != size for row in self.distance_m):
            raise ValueError("Distance matrix has the wrong dimensions")
        if any(row[i] != 0 or any(value is not None and (
            not isinstance(value, Integral) or isinstance(value, bool) or value < 0
        ) for value in row)
               for i, row in enumerate(self.distance_m)):
            raise ValueError("Expected nonnegative distances and zero diagonal")

    def impossible_reason(self) -> str | None:
        """Return a certificate from simple necessary capacity conditions."""
        if max(self.demand_kg) > self.capacity_kg:
            return "A customer demand exceeds vehicle capacity"
        if sum(self.demand_kg) > self.vehicle_count * self.capacity_kg:
            return "Total demand exceeds fleet capacity"
        return None


@dataclass(frozen=True)
class RoutingResult:
    status: str
    routes: Routes
    elapsed_s: float
    detail: str = ""


def load_cvrp(matrix: TravelMatrix, order_file: Path, vehicle_count: int, capacity_kg: int) -> CVRP:
    """Map one order per customer to the travel matrix order."""
    matrix.validate()
    with order_file.open(newline="") as stream:
        orders = list(csv.DictReader(stream))
    demands = {row["customer_id"]: float(row["delivery_weight"]) for row in orders}
    if len(demands) != len(orders) or demands.keys() != set(matrix.ids[1:]):
        raise ValueError("Orders must match the matrix customer IDs exactly")
    if not all(value.is_integer() for value in demands.values()):
        raise ValueError("This CVRP expects whole kilogram demands")
    distances = tuple(tuple(round(value * 1000) if np.isfinite(value) else None for value in row)
                      for row in matrix.distance_km)
    return CVRP(matrix.ids, (0, *(int(demands[key]) for key in matrix.ids[1:])),
                distances, vehicle_count, capacity_kg)
