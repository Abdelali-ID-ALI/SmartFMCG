"""Load ordered locations and build or reuse a content-keyed local matrix."""

import argparse
import csv
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from src.routing.base_provider import Location, MatrixProvider, TravelMatrix
from src.routing.haversine import HaversineProvider
from src.routing.osrm_client import OSRMProvider, RoutingError


def load_locations(directory: Path) -> tuple[Location, ...]:
    """Read depots first and customers second, retaining each CSV's ordering."""
    locations = []
    for table, key in (("depots", "depot_id"), ("customers", "customer_id")):
        with (directory / f"{table}.csv").open(newline="") as stream:
            locations.extend(Location(row[key], float(row["latitude"]), float(row["longitude"]))
                             for row in csv.DictReader(stream))
    return tuple(locations)


def matrix_path(provider: MatrixProvider, locations: tuple[Location, ...], cache: Path) -> Path:
    """Identify an ordered location set and provider configuration."""
    key = {"version": 1, "provider": provider.signature(), "locations": [asdict(p) for p in locations]}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return cache / f"{provider.signature()['provider']}-{digest}.npz"


def cached_matrix(provider: MatrixProvider, locations: tuple[Location, ...], cache: Path,
                  *, cache_only: bool = False) -> TravelMatrix:
    """Atomically cache successful results, without silently substituting providers."""
    ids = tuple(point.id for point in locations)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected non-empty, unique location IDs")
    path = matrix_path(provider, locations, cache)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                provenance = json.loads(saved["provenance"].item())
                if tuple(saved["ids"]) == ids and all(
                    provenance.get(key) == value for key, value in provider.signature().items()
                ):
                    return TravelMatrix(ids, saved["distance_km"], saved["duration_min"], provenance)
        except (KeyError, OSError, ValueError, AttributeError, BadZipFile):
            pass
    if cache_only:
        raise FileNotFoundError(f"No valid cached matrix: {path}")
    matrix = provider.matrix(locations)
    matrix.validate()
    if matrix.ids != ids:
        raise ValueError("Provider returned a matrix with different location IDs or order")
    cache.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, ids=matrix.ids, distance_km=matrix.distance_km,
                                duration_min=matrix.duration_min, provenance=json.dumps(matrix.provenance))
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return matrix


def main() -> None:
    """Generate a geographic matrix for the synthetic customers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["haversine", "osrm"], default="haversine")
    parser.add_argument("--endpoint")
    parser.add_argument("--profile", default="driving")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--cache", type=Path, default=Path("data/matrices"))
    parser.add_argument("--limit", type=int, help="Keep this many locations, including the depot")
    args = parser.parse_args()
    if args.provider == "osrm" and not args.endpoint:
        parser.error("--endpoint is required with --provider osrm")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    provider = HaversineProvider() if args.provider == "haversine" else OSRMProvider(args.endpoint, args.profile)
    locations = load_locations(args.data)[:args.limit]
    try:
        matrix = cached_matrix(provider, locations, args.cache)
    except RoutingError as exc:
        parser.exit(1, f"{exc}\n")
    print(f"{len(matrix.ids)} locations, {np.isnan(matrix.distance_km).sum()} unavailable directed arcs")
    print(matrix_path(provider, locations, args.cache))


if __name__ == "__main__":
    main()
