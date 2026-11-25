"""Deterministic layouts and bounded assembly of directed OSRM tables."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from src.data.preprocessing import straight_line_km
from src.routing.base_provider import Location, TravelMatrix
from src.routing.matrices import cached_matrix, load_locations, matrix_path
from src.routing.osrm_client import OSRMProvider

SOURCE = Path("data/synthetic")
CORRIDOR = Path("data/matrices/osrm-D01-C30.geojson")


def locations_for(customers: int, *, geographic: bool) -> tuple[Location, ...]:
    """Use established points and sample distinct vertices on the saved OSM route."""
    if customers < 1:
        raise ValueError("Expected at least one customer")
    anchors = load_locations(SOURCE)
    if customers < len(anchors):
        return anchors[:customers + 1]
    needed = customers + 1 - len(anchors)
    if geographic:
        path = json.loads(CORRIDOR.read_text())["geometry"]["coordinates"]
        chosen = [(p.latitude, p.longitude) for p in anchors]
        available = [(lat, lon) for lon, lat in path]
        gaps = [min(straight_line_km(point, anchor) for anchor in chosen) for point in available]
        for _ in range(needed):
            gap, point = max(zip(gaps, available, strict=True))
            if gap < .045:
                raise ValueError("Archived OSM corridor lacks enough distinct road points for this scale")
            chosen.append(point)
            gaps = [min(gap, straight_line_km(candidate, point))
                    for gap, candidate in zip(gaps, available, strict=True)]
        points = chosen[len(anchors):]
    else:
        rng = np.random.default_rng(104001)
        indices = rng.integers(1, len(anchors), size=needed)
        offsets = rng.uniform(-.003, .003, (needed, 2))
        points = [(anchors[i].latitude + dy, anchors[i].longitude + dx)
                  for i, (dy, dx) in zip(indices, offsets, strict=True)]
    return (*anchors, *(Location(f"S{i:04d}", lat, lon) for i, (lat, lon) in enumerate(points, 31)))


class BlockTables:
    """Join bounded square OSRM queries, preserving directions and checking overlaps."""

    def __init__(self, provider: OSRMProvider, block_size: int, graph: str, cache: Path):
        if not 2 <= block_size <= provider.max_locations // 2 or not graph.strip():
            raise ValueError("Expected a graph label and blocks whose paired size fits the client")
        self.provider = provider
        self.block_size = block_size
        self.graph = graph
        self.cache = cache

    def signature(self) -> dict[str, str]:
        return {**self.provider.signature(), "provider": "osrm-blocks",
                "configured_graph": self.graph, "block_size": str(self.block_size)}

    def matrix(self, locations: tuple[Location, ...]) -> TravelMatrix:
        n = len(locations)
        groups = [np.arange(i, min(i + self.block_size, n)) for i in range(0, n, self.block_size)]
        batches = [groups[0]] if len(groups) == 1 else [
            np.concatenate((a, b)) for i, a in enumerate(groups) for b in groups[i + 1:]
        ]
        distances = np.full((n, n), np.nan)
        durations = distances.copy()
        filled = np.zeros((n, n), dtype=bool)
        points, versions, retrievals = {}, set(), set()
        for indices in batches:
            batch = tuple(locations[i] for i in indices)
            table = cached_matrix(self._block_provider(), batch, self.cache)
            versions.add(table.provenance["data_version"])
            retrievals.add(table.provenance["retrieved_utc"])
            for field in ("snap_sources", "snap_destinations"):
                for point in json.loads(table.provenance[field]):
                    key = point["id"]
                    if key in points and points[key] != point["coordinates"]:
                        raise ValueError(f"OSRM block snap changed for {key}")
                    points[key] = point["coordinates"]
            ix = np.ix_(indices, indices)
            overlap = filled[ix]
            for merged, current in ((distances, table.distance_km), (durations, table.duration_min)):
                if not np.array_equal(merged[ix][overlap], current[overlap], equal_nan=True):
                    raise ValueError("OSRM block overlap disagrees")
                merged[ix] = current
            filled[ix] = True
        if not filled.all() or len(versions) != 1:
            raise ValueError("Incomplete table or inconsistent OSRM graph version")
        return TravelMatrix(tuple(p.id for p in locations), distances, durations, {
            **self.signature(), "data_version": versions.pop(), "block_count": str(len(batches)),
            "retrieved_utc": json.dumps(sorted(retrievals)), "snap_coordinates": json.dumps(points),
        })

    def _block_provider(self):
        """Namespace individual cache entries by the same known graph as the merged result."""
        provider = self.provider

        class GraphTable(OSRMProvider):
            def signature(inner_self):
                return {**super().signature(), "configured_graph": self.graph}

        return GraphTable(provider.endpoint, provider.profile, provider.timeout,
                          provider.max_locations, provider.max_snap_m)


def diagnostics(matrix: TravelMatrix) -> dict:
    n = len(matrix.ids)
    off_diagonal = ~np.eye(n, dtype=bool)
    duration = matrix.duration_min
    distance = matrix.distance_km
    comparable = np.isfinite(distance) & np.isfinite(distance.T) & off_diagonal
    if not np.array_equal(distance == 0, duration == 0):
        raise ValueError("OSRM zero distance and duration disagree")
    if "snap_coordinates" in matrix.provenance:
        snapped = list(json.loads(matrix.provenance["snap_coordinates"]).values())
    else:
        snapped = [row["coordinates"] for row in json.loads(matrix.provenance.get("snap_sources", "[]"))]
    return {"matrix_cells": n * n, "matrix_array_bytes": distance.nbytes + duration.nbytes,
            "unreachable_arcs": int(np.isnan(distance).sum()),
            "zero_off_diagonal": int(((distance == 0) & off_diagonal).sum()),
            "duplicate_snaps": len(snapped) - len({tuple(point) for point in snapped}) if snapped else None,
            "asymmetric_distance_share": float(np.mean(distance[comparable] != distance.T[comparable])),
            "asymmetric_duration_share": float(np.mean(duration[comparable] != duration.T[comparable]))}


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--max-locations", type=int, default=51)
    parser.add_argument("--graph", help="Serving graph identifier or single collection label for block caches")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.block_size and not args.graph:
        parser.error("--graph is required for block tables")
    begin = perf_counter()
    locations = locations_for(args.customers, geographic=True)
    locations_s = perf_counter() - begin
    provider = OSRMProvider(args.endpoint, max_locations=args.max_locations)
    if args.block_size:
        provider = BlockTables(provider, args.block_size, args.graph, args.cache)
    path = matrix_path(provider, locations, args.cache)
    begin = perf_counter()
    hit_before = path.exists()
    matrix = cached_matrix(provider, locations, args.cache, cache_only=args.cache_only)
    elapsed = perf_counter() - begin
    begin = perf_counter()
    warm = cached_matrix(provider, locations, args.cache, cache_only=True)
    warm_s = perf_counter() - begin
    if not all(np.array_equal(getattr(matrix, x), getattr(warm, x), equal_nan=True)
               for x in ("distance_km", "duration_min")):
        raise ValueError("Cache replay changed the OSRM values")
    data = {"customers": args.customers, "cache_file": path.name, "locations_s": locations_s,
            "matrix_build_or_load_s": elapsed, "cache_present_before": hit_before, "cache_reload_s": warm_s,
            "block_count": int(matrix.provenance.get("block_count", "1")), **diagnostics(matrix)}
    print(json.dumps(data))
    if args.record:
        args.record.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
