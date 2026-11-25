"""Generate a seeded synthetic beverage distribution instance."""
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocessing import straight_line_km
from src.data.validation import feasibility_witness, validate_tables, validate_traffic_reference
from src.routing.base_provider import Location, TravelMatrix
from src.routing.matrices import cached_matrix
from src.routing.osrm_client import OSRMProvider

ZONES = {
    "ain_sebaa": (33.608, -7.553),
    "sidi_bernoussi": (33.610, -7.505),
    "roches_noires": (33.600, -7.590),
    "maarif": (33.579, -7.641),
    "bourgogne": (33.598, -7.646),
    "ain_diab": (33.589, -7.689),
}
# Synthetic Aïn Diab sites anchored to distinct vertices of the archived
# D01→C30 road geometry after the geographic audit.
AIN_DIAB_ROAD_LOCATIONS = (
    (33.587283, -7.684843),
    (33.587716, -7.684036),
    (33.588279, -7.682978),
    (33.588886, -7.681652),
    (33.587333, -7.686370),
)
# Category, expected crates, access constraint.
PROFILES = [
    ("grocery", 16, "no_heavy"), ("supermarket", 46, "any"),
    ("restaurant", 22, "no_heavy"), ("cafe", 10, "no_heavy"),
    ("wholesaler", 65, "any"),
]


def generate(seed: int, service_date: date, observation_count: int, *,
             routing_matrix: TravelMatrix | None = None,
             matrix_cache: Path = Path("data/matrices"),
             osrm_endpoint: str | None = None) -> dict[str, pd.DataFrame]:
    """Build all six linked tables from independent seeded random streams."""
    if observation_count < 1:
        raise ValueError("traffic_observations must be positive")
    demand_rng, traffic_rng, event_rng = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(3))
    depots = [dict(depot_id="D01", latitude=33.599, longitude=-7.547,
                   opening_time="07:00", closing_time="18:00")]
    vehicles = pd.DataFrame([
        ("V01", "van", 2300, 10, 195, 4, 160, 2.2, 45, 10, False),
        ("V02", "van", 2400, 11, 200, 4, 175, 2.3, 45, 11, False),
        ("V03", "medium", 2900, 16, 250, 6, 245, 3.5, 60, 17, False),
        ("V04", "medium", 3200, 18, 270, 6, 260, 3.7, 60, 18, False),
        ("V05", "heavy", 6000, 30, 500, 12, 390, 5.0, 75, 25, False),
        ("V06", "refrigerated", 3000, 17, 260, 6, 320, 4.3, 65, 20, True),
    ], columns="""vehicle_id vehicle_type maximum_weight maximum_volume maximum_crates
        maximum_pallets fixed_operating_cost cost_per_km cost_per_hour fuel_consumption refrigerated""".split())
    vehicles = vehicles.assign(shift_start="07:30", shift_end="17:30")
    customers, orders = [], []
    for zone_index, (zone, (latitude, longitude)) in enumerate(ZONES.items()):
        for slot in range(5):
            index = zone_index * 5 + slot + 1
            category, typical, access = PROFILES[(slot + zone_index) % len(PROFILES)]
            customer_id = f"C{index:02}"
            priority = "high" if category == "supermarket" else "standard"
            crates = int(np.clip(round(demand_rng.normal(typical, typical * 0.14)), 5, int(typical * 1.18)))
            pickup = int(demand_rng.binomial(crates, 0.56 if category in {"cafe", "restaurant"} else 0.28))
            cold = zone_index == 5 and category in {"restaurant", "supermarket"}
            # Consume the same draws to preserve the historical demand stream.
            point = (round(latitude + demand_rng.uniform(-.003, .003), 6),
                     round(longitude + demand_rng.uniform(-.003, .003), 6))
            if zone == "ain_diab":
                point = AIN_DIAB_ROAD_LOCATIONS[slot]
            customers.append(dict(customer_id=customer_id, customer_type=category,
                latitude=point[0], longitude=point[1],
                geographical_zone=zone, time_window_start=f"{8 + slot:02}:00",
                time_window_end=f"{11 + slot:02}:00", service_duration=8 + (crates + 5) // 6,
                priority=priority, vehicle_access_restrictions="any" if zone_index == 4 else access))
            orders.append(dict(order_id=f"O{index:02}", customer_id=customer_id,
                product_type="chilled_juice" if cold else "ambient_beverage",
                quantity=crates * 12, delivery_weight=crates * 12,
                delivery_volume=round(crates * .032, 3), delivery_crates=crates,
                delivery_pallets=(crates + 39) // 40 if typical >= 46 else 0,
                pickup_weight=pickup * 2, pickup_volume=round(pickup * .020, 3),
                pickup_crates=pickup, refrigerated_required=cold, priority=priority,
                mandatory=category != "cafe", order_date=service_date.isoformat()))
    locations = depots + customers
    ordered_locations = tuple(Location(point.get("customer_id", "D01"), point["latitude"], point["longitude"])
                              for point in locations)
    if osrm_endpoint is not None:
        if routing_matrix is not None:
            raise ValueError("Provide a matrix or an OSRM endpoint, not both")
        routing_matrix = cached_matrix(OSRMProvider(osrm_endpoint), ordered_locations, matrix_cache, cache_only=True)
    if routing_matrix is not None:
        if routing_matrix.ids != tuple(point.id for point in ordered_locations):
            raise ValueError("Traffic reference must match ordered generated locations")
        pairs = np.argwhere((routing_matrix.distance_km > 0) & (routing_matrix.duration_min > 0))
        if not len(pairs):
            raise ValueError("Traffic reference has no positive routed arcs")
    rows = []
    for _ in range(observation_count):
        origin_index, target_index = (
            traffic_rng.choice(len(locations), size=2, replace=False) if routing_matrix is None
            else pairs[traffic_rng.integers(len(pairs))]
        )
        origin, target = locations[origin_index], locations[target_index]
        timestamp = datetime.combine(service_date, datetime.min.time()) + timedelta(
            days=int(traffic_rng.integers(-56, 0)), hours=int(traffic_rng.integers(6, 22)),
            minutes=int(traffic_rng.integers(0, 60)))
        same_zone = origin.get("geographical_zone") == target.get("geographical_zone")
        road = "local" if same_zone else "arterial"
        if routing_matrix is None:
            distance = 1.28 * straight_line_km((origin["latitude"], origin["longitude"]),
                                               (target["latitude"], target["longitude"]))
            static = 60 * distance / (22 if same_zone else 35)
        else:
            distance = routing_matrix.distance_km[origin_index, target_index]
            static = routing_matrix.duration_min[origin_index, target_index]
        rush = timestamp.hour in {7, 8, 9, 16, 17, 18}
        weekday = timestamp.weekday() < 5
        central = any(point.get("geographical_zone") in {"maarif", "bourgogne"} for point in (origin, target))
        incident = traffic_rng.random() < .035
        multiplier = 1.04 + (.46 if rush and weekday else .15 if rush else 0)
        multiplier += .12 * central + .08 * same_zone + .65 * incident
        multiplier += traffic_rng.normal(0, .12)
        rows.append(dict(timestamp=timestamp.isoformat(timespec="minutes"),
            origin_id=origin.get("customer_id", "D01"), destination_id=target.get("customer_id", "D01"),
            road_distance=round(distance, 4), static_travel_time=round(static, 4),
            departure_hour=timestamp.hour, day_of_week=timestamp.weekday(), road_characteristics=road,
            congestion_indicators="incident" if incident else "normal",
            observed_travel_time=round(static * max(.8, multiplier), 4)))
    events = []
    for index, (kind, time, unit, low, high) in enumerate([
        ("traffic_congestion", "09:15", "travel_time_multiplier", 1.3, 2.0),
        ("service_delay", "10:30", "minutes", 8, 25),
        ("urgent_order", "11:45", "crates", 5, 15),
        ("cancellation", "07:45", "orders", 1, 1),
        ("extra_pickup", "12:30", "crates", 2, 12),
    ], start=1):
        customer = customers[int(event_rng.integers(len(customers)))]
        severity = (round(event_rng.uniform(low, high), 2) if kind == "traffic_congestion"
                    else int(event_rng.integers(low, high + 1)))
        events.append(dict(event_id=f"E{index:02}", timestamp=f"{service_date}T{time}",
            event_type=kind, target_id=customer["geographical_zone"] if kind == "traffic_congestion"
            else customer["customer_id"], severity=severity, unit=unit))
    traffic = pd.DataFrame(rows).sort_values("timestamp", kind="stable")
    validate_traffic_reference(traffic, routing_matrix, (*depots, *customers))
    return {
        "depots": pd.DataFrame(depots), "customers": pd.DataFrame(customers),
        "vehicles": vehicles, "orders": pd.DataFrame(orders),
        "traffic_observations": traffic,
        "events": pd.DataFrame(events).sort_values("timestamp", kind="stable"),
    }


def main() -> None:
    """Generate, validate and export the configured instance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/development.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--traffic-reference", choices=("osrm", "proxy"),
                        help="Override the configured traffic reference, e.g. proxy to bootstrap locations")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    mode = args.traffic_reference or config["traffic_reference"]
    if mode not in {"osrm", "proxy"}:
        raise ValueError("traffic_reference must be osrm or proxy")
    tables = generate(config["seed"], date.fromisoformat(config["service_date"]), config["traffic_observations"],
                      matrix_cache=Path(config.get("matrix_cache", "data/matrices")),
                      osrm_endpoint=config["osrm_endpoint"] if mode == "osrm" else None)
    validate_tables(tables)
    feasibility_witness(tables)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.output / f"{name}.csv", index=False, lineterminator="\n")
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
