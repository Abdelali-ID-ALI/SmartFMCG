"""Shared contract for directed travel matrices."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Location:
    """A geographic point with a stable identifier."""

    id: str
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not self.id or not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError(f"Invalid location: {self.id}")


@dataclass(frozen=True)
class TravelMatrix:
    """Directed distances in km and durations in minutes. NaN denotes unavailable arcs."""

    ids: tuple[str, ...]
    distance_km: NDArray[np.float64]
    duration_min: NDArray[np.float64]
    provenance: dict[str, str]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the directed grid, including edits to the underlying arrays."""
        n = len(self.ids)
        if n == 0 or len(set(self.ids)) != n:
            raise ValueError("Expected non-empty, unique matrix IDs")
        for array in (self.distance_km, self.duration_min):
            if (
                array.shape != (n, n)
                or np.isinf(array).any()
                or (array < 0).any()
                or not (array.diagonal() == 0).all()
            ):
                raise ValueError("Invalid matrix values, shape or diagonal")
        if not np.array_equal(np.isnan(self.distance_km), np.isnan(self.duration_min)):
            raise ValueError("Distance and duration must agree on unavailable arcs")
        if ((self.distance_km > 0) & (self.duration_min == 0)).any():
            raise ValueError("Unexplained zero arc: Positive distance requires positive travel duration")


class MatrixProvider(Protocol):
    """Build a matrix for locations in the supplied order."""

    def signature(self) -> dict[str, str]: ...
    def matrix(self, locations: tuple[Location, ...]) -> TravelMatrix: ...
