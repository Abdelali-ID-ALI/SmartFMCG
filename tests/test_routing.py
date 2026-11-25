"""Check geographic units, directed arcs, cache and OSRM failure paths."""

import io
import json
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from src.routing.base_provider import Location
from src.routing.haversine import HaversineProvider
from src.routing.matrices import cached_matrix, matrix_path
from src.routing.osrm_client import OSRMProvider, RoutingError

LOCATIONS = (Location("B", 0, 1), Location("A", 0, 0), Location("C", 0, 2))
ENDPOINT = "https://routing.example"


@pytest.fixture(autouse=True)
def reset_rate_limit(monkeypatch):
    monkeypatch.setattr("src.routing.osrm_client._LAST_REQUEST", {})


def test_haversine_equator():
    matrix = HaversineProvider(60).matrix(LOCATIONS)
    assert matrix.ids == ("B", "A", "C")
    assert matrix.distance_km[0, 1] == pytest.approx(111.195, rel=1e-5)
    np.testing.assert_array_equal(matrix.distance_km, matrix.distance_km.T)
    np.testing.assert_array_equal(matrix.distance_km, matrix.duration_min)


def test_osrm_table_preserves_direction_and_unavailable_arcs(monkeypatch):
    def response(request, timeout):
        assert "/table/v1/driving/1.00000000,0.00000000;" in request.full_url
        assert request.full_url.endswith("?annotations=distance,duration")
        assert request.get_header("User-agent").startswith("SmartFMCG/")
        assert timeout == 12
        data = {"code": "Ok", "durations": [[0, 60, None], [120, 0, 60], [120, 60, 0]],
                "distances": [[0, 1500, None], [2500, 0, 1000], [1000, 1500, 0]]}
        points = [{"location": [p.longitude, p.latitude], "distance": 0} for p in LOCATIONS]
        data.update(sources=points, destinations=points)
        return io.BytesIO(json.dumps(data).encode())

    monkeypatch.setattr("src.routing.osrm_client.urlopen", response)
    matrix = OSRMProvider(ENDPOINT).matrix(LOCATIONS)
    assert matrix.ids == ("B", "A", "C")
    assert matrix.distance_km[0, 1] == 1.5
    assert matrix.distance_km[1, 0] == 2.5
    assert matrix.duration_min[1, 0] == 2
    assert np.isnan(matrix.distance_km[0, 2])
    assert np.isfinite(matrix.distance_km[2, 0])


def test_osrm_geometry(monkeypatch):
    data = {"code": "Ok", "routes": [{"geometry": {"type": "LineString",
            "coordinates": [[1, 0], [0.5, 0.1], [0, 0]]}}]}
    monkeypatch.setattr("src.routing.osrm_client.urlopen",
                        lambda request, timeout: io.BytesIO(json.dumps(data).encode()))
    assert OSRMProvider(ENDPOINT).route_geometry(LOCATIONS[:2]) == ((1, 0), (0.5, 0.1), (0, 0))


@pytest.mark.parametrize("payload,expected", [
    (b'{"code":"NoTable"}', "NoTable"),
    (b'{"code":"TooBig"}', "TooBig"),
    (b'{"code":"Ok","durations":[[0]]}', "invalid distance/duration"),
    (b'invalid json', "Expecting value"),
])
def test_osrm_bad_payload(monkeypatch, payload, expected):
    monkeypatch.setattr("src.routing.osrm_client.urlopen",
                        lambda request, timeout: io.BytesIO(payload))
    with pytest.raises(RoutingError, match=expected):
        OSRMProvider(ENDPOINT).matrix(LOCATIONS)


@pytest.mark.parametrize("error,expected", [
    (URLError("DNS failure"), "DNS failure"),
    (TimeoutError("timed out"), "timed out"),
    (HTTPError(ENDPOINT, 429, "Too Many Requests", {}, None), "HTTP 429"),
])
def test_osrm_transport_errors(monkeypatch, tmp_path, error, expected):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr("src.routing.osrm_client.urlopen", fail)
    with pytest.raises(RoutingError, match=expected):
        cached_matrix(OSRMProvider(ENDPOINT), LOCATIONS, tmp_path)
    assert not list(tmp_path.iterdir())


def test_osrm_batch_limit_precedes_http():
    with pytest.raises(RoutingError, match="2–2"):
        OSRMProvider(ENDPOINT, max_locations=2).matrix(LOCATIONS)


def test_cache_keys_and_corruption(tmp_path):
    provider = HaversineProvider()
    created = cached_matrix(provider, LOCATIONS, tmp_path)
    path = matrix_path(provider, LOCATIONS, tmp_path)
    assert path != matrix_path(provider, LOCATIONS[::-1], tmp_path)
    moved = (Location("B", 0, 1.1), *LOCATIONS[1:])
    assert path != matrix_path(provider, moved, tmp_path)
    assert path != matrix_path(HaversineProvider(50), LOCATIONS, tmp_path)
    loaded = cached_matrix(provider, LOCATIONS, tmp_path, cache_only=True)
    np.testing.assert_array_equal(created.distance_km, loaded.distance_km)
    path.write_bytes(b"invalid")
    with pytest.raises(FileNotFoundError):
        cached_matrix(provider, LOCATIONS, tmp_path, cache_only=True)
    cached_matrix(provider, LOCATIONS, tmp_path)
    cached_matrix(provider, LOCATIONS, tmp_path, cache_only=True)


@pytest.mark.parametrize("distance,seconds,waypoint,offset,error", [
    (0, 0, (1, 0), 15, None),
    (0, 0, (1.0001, 0), 15, "Unexplained zero arc"),
    (0, 0, (0, 0), 15, "geometric offset"),
    (20, 0, (1, 0), 15, "Unexplained zero arc"),
    (0, 0, (1, 0), 51, "Unacceptable snap"),
])
def test_osrm_zero_arcs_require_plausible_shared_access(monkeypatch, distance, seconds, waypoint, offset, error):
    points = [{"location": [1, 0], "distance": 0}, {"location": waypoint, "distance": offset}]
    data = {"code": "Ok", "distances": [[0, distance], [1000, 0]],
            "durations": [[0, seconds], [60, 0]], "sources": points, "destinations": points}
    monkeypatch.setattr("src.routing.osrm_client.urlopen", lambda *_args, **_kwargs:
                        io.BytesIO(json.dumps(data).encode()))
    provider = OSRMProvider(ENDPOINT)
    locations = (LOCATIONS[0], Location("A", 0, 1.0001))
    if error:
        with pytest.raises(RoutingError, match=error):
            provider.matrix(locations)
    else:
        matrix = provider.matrix(locations)
        assert json.loads(matrix.provenance["shared_access_arcs"]) == [["B", "A"]]
        assert json.loads(matrix.provenance["snap_sources"])[1]["offset_m"] == 15


def test_rate_limit_shared_by_clients(monkeypatch):
    clock = [0.0]
    times = []
    monkeypatch.setattr("src.routing.osrm_client.monotonic", lambda: clock[0])

    def sleep(seconds):
        clock[0] += seconds

    def response(*_args, **_kwargs):
        times.append(clock[0])
        return io.BytesIO(b'{"code":"Ok"}')

    monkeypatch.setattr("src.routing.osrm_client.sleep", sleep)
    monkeypatch.setattr("src.routing.osrm_client.urlopen", response)
    for _ in range(2):
        OSRMProvider(ENDPOINT)._request("table", LOCATIONS, "")
    assert times == [0, 1.05]
