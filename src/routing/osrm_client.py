"""Bounded HTTP client for OSRM table and route services."""

import json
import re
from datetime import datetime, timezone
from math import isfinite
from threading import Lock
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

_REQUEST_LOCK = Lock()
_LAST_REQUEST: dict[str, float] = {}

from src.data.preprocessing import straight_line_km
from src.routing.base_provider import Location, TravelMatrix


class RoutingError(RuntimeError):
    """The routing service could not supply a usable result."""


class OSRMProvider:
    """Query one coordinate batch at a time with explicit limits and no implicit retries."""

    def __init__(self, endpoint: str, profile: str = "driving", timeout: float = 12,
                 max_locations: int = 32, max_snap_m: float = 50) -> None:
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.query or parsed.fragment or not re.fullmatch(r"[a-zA-Z0-9_-]+", profile)):
            raise ValueError("Invalid OSRM endpoint or profile")
        if not isfinite(timeout) or timeout <= 0 or max_locations < 2 or not isfinite(max_snap_m) or max_snap_m < 0:
            raise ValueError("Invalid OSRM limits")
        self.endpoint = endpoint.rstrip("/")
        self.profile = profile
        self.timeout = timeout
        self.max_locations = max_locations
        self.max_snap_m = float(max_snap_m)
        self._host = parsed.hostname

    def signature(self) -> dict[str, str]:
        return {"provider": "osrm", "endpoint": self.endpoint, "profile": self.profile,
                "max_snap_m": str(self.max_snap_m), "snap_policy": "qualified-v1"}

    def _request(self, service: str, locations: tuple[Location, ...], query: str) -> dict:
        if not 2 <= len(locations) <= self.max_locations:
            raise RoutingError(f"OSRM request requires 2–{self.max_locations} locations")
        coordinates = ";".join(f"{p.longitude:.8f},{p.latitude:.8f}" for p in locations)
        url = f"{self.endpoint}/{service}/v1/{self.profile}/{coordinates}?{query}"
        with _REQUEST_LOCK:
            last = _LAST_REQUEST.get(self._host, float("-inf"))
            sleep(max(0, 1.05 - (monotonic() - last)))
            _LAST_REQUEST[self._host] = monotonic()
        try:
            with urlopen(Request(url, headers={"User-Agent": "SmartFMCG/0.2 (educational routing study)"}),
                         timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise RoutingError(f"OSRM {service}: HTTP {exc.code}") from exc
        except (URLError, OSError, ValueError) as exc:
            raise RoutingError(f"OSRM {service}: {exc}") from exc
        if not isinstance(result, dict) or result.get("code") != "Ok":
            code = result.get("code") if isinstance(result, dict) else "invalid payload"
            raise RoutingError(f"OSRM {service}: {code}")
        return result

    def matrix(self, locations: tuple[Location, ...]) -> TravelMatrix:
        result = self._request("table", locations, "annotations=distance,duration")
        try:
            distances = np.array(result["distances"], dtype=float) / 1000
            durations = np.array(result["durations"], dtype=float) / 60
            provenance = {**self.signature(), "data_version": str(result.get("data_version", "unknown")),
                          "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            matrix = TravelMatrix(tuple(point.id for point in locations), distances, durations, provenance)
            source_points = self._snap_points(result["sources"], locations)
            destination_points = self._snap_points(result["destinations"], locations)
            shared = []
            for i, j in zip(*np.where((distances == 0) | (durations == 0)), strict=True):
                if i == j:
                    continue
                if (distances[i, j] != 0 or durations[i, j] != 0
                        or source_points[i]["coordinates"] != destination_points[j]["coordinates"]):
                    raise ValueError(f"Unexplained zero arc {locations[i].id} → {locations[j].id}")
                shared.append([locations[i].id, locations[j].id])
            provenance.update(snap_sources=json.dumps(source_points), snap_destinations=json.dumps(destination_points),
                              shared_access_arcs=json.dumps(shared))
            return matrix
        except (KeyError, TypeError, ValueError) as exc:
            raise RoutingError(f"OSRM invalid distance/duration tables: {exc}") from exc

    def _snap_points(self, waypoints: list[dict], locations: tuple[Location, ...]) -> list[dict]:
        """Validate both ordered waypoint arrays and retain source-to-road offsets."""
        points = []
        for point, waypoint in zip(locations, waypoints, strict=True):
            longitude, latitude = map(float, waypoint["location"])
            offset = float(waypoint["distance"])
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90 and 0 <= offset <= self.max_snap_m):
                raise ValueError(f"Unacceptable snap for {point.id}: {offset:g} m (limit {self.max_snap_m:g} m)")
            geometric_offset = 1000 * straight_line_km((point.latitude, point.longitude), (latitude, longitude))
            # OSRM and Haversine can differ slightly in geodesic approximation.
            if geometric_offset > self.max_snap_m + 0.5:
                raise ValueError(f"Unacceptable snap for {point.id}: geometric offset {geometric_offset:.1f} m")
            points.append({"id": point.id, "coordinates": [longitude, latitude], "offset_m": offset})
        return points

    def route_geometry(self, locations: tuple[Location, ...]) -> tuple[tuple[float, float], ...]:
        """Return route coordinates in GeoJSON order (longitude, latitude)."""
        result = self._request("route", locations, "overview=full&geometries=geojson&steps=false")
        try:
            geometry = result["routes"][0]["geometry"]
            coordinates = tuple((float(lon), float(lat)) for lon, lat in geometry["coordinates"])
            if (geometry["type"] != "LineString" or len(coordinates) < 2
                    or not all(-180 <= lon <= 180 and -90 <= lat <= 90 for lon, lat in coordinates)):
                raise ValueError("Invalid LineString")
            return coordinates
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RoutingError(f"OSRM invalid route geometry: {exc}") from exc
