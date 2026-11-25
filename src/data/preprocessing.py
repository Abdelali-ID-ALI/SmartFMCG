"""Shared phase-one geographical and time conversions."""
import re
from math import asin, cos, radians, sin, sqrt


def straight_line_km(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    """Return spherical distance in kilometres."""
    lat1, lon1 = map(radians, origin)
    lat2, lon2 = map(radians, destination)
    angle = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)**2
    return 6371 * 2 * asin(min(1.0, sqrt(angle)))


def minutes_since_midnight(value: str) -> int:
    """Parse a strict HH:MM clock value."""
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value):
        raise ValueError(f"Invalid clock time: {value}")
    hour, minute = (int(part) for part in value.split(":"))
    return 60 * hour + minute
