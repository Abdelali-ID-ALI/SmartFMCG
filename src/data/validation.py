"""Validate the phase-one data contract and a constructive feasibility witness."""
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.data.preprocessing import minutes_since_midnight, straight_line_km
from src.routing.base_provider import TravelMatrix

SCHEMAS = {
    "depots": "depot_id latitude longitude opening_time closing_time",
    "customers": """customer_id customer_type latitude longitude geographical_zone time_window_start
        time_window_end service_duration priority vehicle_access_restrictions""",
    "vehicles": """vehicle_id vehicle_type maximum_weight maximum_volume maximum_crates maximum_pallets
        fixed_operating_cost cost_per_km cost_per_hour fuel_consumption refrigerated shift_start shift_end""",
    "orders": """order_id customer_id product_type quantity delivery_weight delivery_volume
        delivery_crates delivery_pallets pickup_weight pickup_volume pickup_crates refrigerated_required
        priority mandatory order_date""",
    "traffic_observations": """timestamp origin_id destination_id road_distance static_travel_time
        departure_hour day_of_week road_characteristics congestion_indicators observed_travel_time""",
    "events": "event_id timestamp event_type target_id severity unit",
}
ZONE_VEHICLE = dict(zip(
    ("ain_sebaa", "sidi_bernoussi", "roches_noires", "maarif", "bourgogne", "ain_diab"),
    ("V01", "V02", "V03", "V04", "V05", "V06"), strict=True))
DIMENSIONS = ["weight", "volume", "crates", "pallets"]
EVENT_UNITS = dict(traffic_congestion="travel_time_multiplier", service_delay="minutes",
                   urgent_order="crates", cancellation="orders", extra_pickup="crates")


def require(condition: bool, message: str) -> None:
    """Raise a domain validation error."""
    if not condition:
        raise ValueError(message)


def capacity_trace(deliveries: np.ndarray, pickups: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Replay unload-before-pickup and return total onboard loads."""
    remaining = deliveries.sum(axis=0)
    returns = np.zeros_like(remaining)
    trace = [remaining.copy()]
    require(bool((remaining <= capacity + 1e-9).all()), "Initial load exceeds capacity")
    for delivery, pickup in zip(deliveries, pickups, strict=True):
        remaining -= delivery
        require(bool((remaining >= -1e-9).all()), "Negative delivery inventory")
        returns += pickup
        trace.append(remaining + returns)
        require(bool((trace[-1] <= capacity + 1e-9).all()), "Pickup makes load exceed capacity")
    return np.array(trace)


def feasibility_witness(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Replay a fixed zone allocation and time-window order under nominal proxy travel times."""
    depot = tables["depots"].iloc[0]
    visits = tables["orders"].merge(tables["customers"], on="customer_id", validate="one_to_one")
    vehicles = tables["vehicles"].set_index("vehicle_id")
    require(set(visits["geographical_zone"]) == ZONE_VEHICLE.keys(), "Unexpected service zones")
    require(set(vehicles.index) == set(ZONE_VEHICLE.values()), "Unexpected fleet identifiers")
    summary = []
    for zone, route in visits.groupby("geographical_zone"):
        route = route.sort_values("time_window_start")
        vehicle = vehicles.loc[ZONE_VEHICLE[zone]]
        require(vehicle.refrigerated or not route["refrigerated_required"].any(), f"Cold chain: {zone}")
        require(vehicle.vehicle_type != "heavy" or not route["vehicle_access_restrictions"].eq("no_heavy").any(),
                f"Vehicle access: {zone}")
        deliveries = route[[f"delivery_{dim}" for dim in DIMENSIONS]].to_numpy(dtype=float)
        pickups = route[["pickup_weight", "pickup_volume", "pickup_crates"]].assign(pickup_pallets=0).to_numpy()
        capacities = vehicle[[f"maximum_{dim}" for dim in DIMENSIONS]].to_numpy(dtype=float)
        trace = capacity_trace(deliveries, pickups, capacities)
        start = max(map(minutes_since_midnight, [depot.opening_time, vehicle.shift_start]))
        end = min(map(minutes_since_midnight, [depot.closing_time, vehicle.shift_end]))
        depot_point = (depot.latitude, depot.longitude)
        previous, clock = depot_point, start
        for stop in route.itertuples():
            point = (stop.latitude, stop.longitude)
            travel_time = 60 * 1.28 * straight_line_km(previous, point) / 22
            clock = max(clock + travel_time, minutes_since_midnight(stop.time_window_start))
            clock += stop.service_duration
            require(clock <= minutes_since_midnight(stop.time_window_end), f"Service window: {stop.customer_id}")
            previous = point
        clock += 60 * 1.28 * straight_line_km(previous, depot_point) / 22
        require(clock <= end, f"Shift/depot closing: {zone}")
        summary.append({"vehicle_id": vehicle.name, "zone": zone, "stops": len(route),
                        "return_min": clock, **{f"peak_{dim}_pct": value for dim, value in
                            zip(DIMENSIONS, 100 * trace.max(axis=0) / capacities, strict=True)}})
    return pd.DataFrame(summary).sort_values("vehicle_id", ignore_index=True)


def validate_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    """Validate schemas, physical units, references and nominal feasibility."""
    require(tables.keys() == SCHEMAS.keys(), "Missing or unexpected table")
    for name, schema in SCHEMAS.items():
        frame = tables[name]
        columns = schema.split()
        require(list(frame.columns) == columns, f"Invalid schema: {name}")
        require(not frame.empty and not frame.isna().any().any(), f"Missing values: {name}")
        if name != "traffic_observations":
            require(frame[columns[0]].is_unique, f"Duplicate primary key: {name}")

    depots, customers, vehicles, orders, traffic, events = (tables[name] for name in SCHEMAS)
    require(len(depots) == 1, "Expected one depot")
    require(set(orders.customer_id) == set(customers.customer_id) and orders.customer_id.is_unique,
            "Expected one order per customer")
    require(set(depots.depot_id).isdisjoint(customers.customer_id), "Location ID collision")
    for frame in (depots, customers):
        require(frame.latitude.between(33.45, 33.75).all() and frame.longitude.between(-7.8, -7.35).all(),
                "Coordinates outside the Casablanca study envelope")
    for frame, start, end in [(depots, "opening_time", "closing_time"),
                              (customers, "time_window_start", "time_window_end"),
                              (vehicles, "shift_start", "shift_end")]:
        require((frame[start].map(minutes_since_midnight) < frame[end].map(minutes_since_midnight)).all(),
                f"Inverted interval: {start}")
    for frame, columns, floor in [
        (vehicles, "maximum_weight maximum_volume maximum_crates maximum_pallets fixed_operating_cost cost_per_km cost_per_hour fuel_consumption", 0),
        (customers, "service_duration", 0),
        (orders, "quantity delivery_weight delivery_volume delivery_crates", 0),
        (orders, "delivery_pallets pickup_weight pickup_volume pickup_crates", -1),
        (traffic, "road_distance static_travel_time observed_travel_time", 0),
    ]:
        values = frame[columns.split()].to_numpy(dtype=float)
        valid = values > 0 if floor == 0 else values >= 0
        require(bool((np.isfinite(values) & valid).all()), f"Invalid physical values: {columns}")
    for frame, columns in [(vehicles, "maximum_crates maximum_pallets"),
                           (orders, "quantity delivery_crates delivery_pallets pickup_crates")]:
        require(bool((frame[columns.split()].to_numpy() % 1 == 0).all()), f"Non-integer counts: {columns}")
    for series in (vehicles.refrigerated, orders.refrigerated_required, orders.mandatory):
        require(pd.api.types.is_bool_dtype(series), f"Expected booleans: {series.name}")
    require(customers.vehicle_access_restrictions.isin(["any", "no_heavy"]).all(), "Invalid access code")
    require(orders.priority.eq(orders.customer_id.map(customers.set_index("customer_id").priority)).all(),
            "Order and customer priority mismatch")
    require(orders.quantity.eq(12 * orders.delivery_crates).all()
            and np.allclose(orders.delivery_weight, 12 * orders.delivery_crates)
            and np.allclose(orders.delivery_volume, .032 * orders.delivery_crates)
            and np.allclose(orders.pickup_weight, 2 * orders.pickup_crates)
            and np.allclose(orders.pickup_volume, .020 * orders.pickup_crates)
            and orders.refrigerated_required.eq(orders.product_type.eq("chilled_juice")).all(),
            "Inconsistent beverage packaging or cold-chain requirements")
    ids = set(depots.depot_id) | set(customers.customer_id)
    require(traffic.origin_id.isin(ids).all() and traffic.destination_id.isin(ids).all()
            and traffic.origin_id.ne(traffic.destination_id).all(), "Invalid traffic endpoints")
    require(traffic.road_characteristics.isin(["local", "arterial"]).all()
            and traffic.congestion_indicators.isin(["normal", "incident"]).all(), "Unknown traffic category")
    service_dates = pd.to_datetime(orders.order_date, format="%Y-%m-%d")
    timestamps = pd.to_datetime(traffic.timestamp, format="%Y-%m-%dT%H:%M")
    require(service_dates.nunique() == 1 and timestamps.lt(service_dates.iloc[0]).all()
            and timestamps.is_monotonic_increasing
            and timestamps.dt.hour.eq(traffic.departure_hour).all()
            and timestamps.dt.day_of_week.eq(traffic.day_of_week).all(), "Inconsistent traffic chronology")
    event_times = pd.to_datetime(events.timestamp, format="%Y-%m-%dT%H:%M")
    require(event_times.dt.normalize().eq(service_dates.iloc[0]).all()
            and event_times.is_monotonic_increasing, "Invalid event chronology")
    require(set(events.event_type) == EVENT_UNITS.keys() and events.event_type.is_unique, "Invalid event types")
    for event in events.itertuples():
        targets = customers.geographical_zone if event.event_type == "traffic_congestion" else customers.customer_id
        valid_severity = (np.isfinite(event.severity) and
                          (event.severity > 1 if event.event_type == "traffic_congestion"
                           else event.severity > 0 and event.severity % 1 == 0))
        require(event.target_id in set(targets) and event.unit == EVENT_UNITS[event.event_type]
                and valid_severity, f"Invalid event: {event.event_id}")



def validate_traffic_reference(traffic: pd.DataFrame, matrix: TravelMatrix | None,
                               locations: tuple[dict, ...]) -> None:
    """Check agreement with the declared reference at the precision of the CSV."""
    identifiers = [point.get("customer_id", point.get("depot_id")) for point in locations]
    lookup = {key: index for index, key in enumerate(identifiers)}
    require(traffic.origin_id.isin(lookup).all() and traffic.destination_id.isin(lookup).all(),
            "Unknown traffic reference endpoint")
    if matrix is not None:
        require(tuple(identifiers) == matrix.ids, "Traffic matrix ordering mismatch")
        origin = traffic.origin_id.map(lookup).to_numpy()
        target = traffic.destination_id.map(lookup).to_numpy()
        distance = matrix.distance_km[origin, target]
        duration = matrix.duration_min[origin, target]
    else:
        coords = [(point["latitude"], point["longitude"]) for point in locations]
        pairs = [(lookup[i], lookup[j]) for i, j in zip(traffic.origin_id, traffic.destination_id, strict=True)]
        distance = np.array([1.28 * straight_line_km(coords[i], coords[j]) for i, j in pairs])
        local = np.array([locations[i].get("geographical_zone") == locations[j].get("geographical_zone")
                          for i, j in pairs])
        duration = 60 * distance / np.where(local, 22, 35)
    require((distance > 0).all() and (duration > 0).all()
            and np.allclose(traffic.road_distance, distance, atol=.00005 + 1e-12, rtol=0)
            and np.allclose(traffic.static_travel_time, duration, atol=.00005 + 1e-12, rtol=0),
            "Traffic distances or durations disagree with the declared reference")
