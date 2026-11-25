"""Offline straight-line distances and constant-speed proxy durations."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from src.data.preprocessing import straight_line_km
from src.routing.base_provider import Location, TravelMatrix


@dataclass(frozen=True)
class HaversineProvider:
    """Compute spherical distances using the phase-one geographical helper."""

    speed_kmh: float = 30.0

    def __post_init__(self) -> None:
        if not isfinite(self.speed_kmh) or self.speed_kmh <= 0:
            raise ValueError("Proxy speed must be finite and positive")

    def signature(self) -> dict[str, str]:
        return {"provider": "haversine", "speed_kmh": str(self.speed_kmh)}

    def matrix(self, locations: tuple[Location, ...]) -> TravelMatrix:
        points = [(point.latitude, point.longitude) for point in locations]
        distances = np.array([[straight_line_km(a, b) for b in points] for a in points])
        return TravelMatrix(tuple(point.id for point in locations), distances,
                            60 * distances / self.speed_kmh, self.signature())
