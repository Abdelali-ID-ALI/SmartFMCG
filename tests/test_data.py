"""Verify reproducibility, table integrity and physical edge cases."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate
from src.data.validation import capacity_trace, feasibility_witness, validate_tables


@pytest.fixture
def tables():
    return generate(42, date(2026, 6, 1), 100)


def test_reproducibility_and_feasibility(tables):
    validate_tables(tables)
    same = generate(42, date(2026, 6, 1), 100)
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, same[name])
    changed = generate(43, date(2026, 6, 1), 100)
    assert not changed["orders"].equals(tables["orders"])
    validate_tables(changed)
    assert feasibility_witness(tables).stops.sum() == 30


@pytest.mark.parametrize("table,column,value,message", [
    ("customers", "latitude", 0.0, "Coordinates"),
    ("customers", "time_window_end", "06:00", "Inverted interval"),
    ("orders", "customer_id", "UNKNOWN", "one order per customer"),
    ("orders", "delivery_weight", 1, "Inconsistent beverage"),
    ("traffic_observations", "departure_hour", 99, "chronology"),
    ("traffic_observations", "destination_id", "UNKNOWN", "endpoints"),
    ("vehicles", "maximum_volume", 0, "physical values"),
    ("vehicles", "maximum_weight", 50, "Initial load"),
    ("events", "target_id", "UNKNOWN", "Invalid event"),
])
def test_rejects_invalid_data(tables, table, column, value, message):
    frame = tables[table]
    frame.loc[frame.index[0], column] = value
    check = feasibility_witness if message == "Initial load" else validate_tables
    with pytest.raises(ValueError, match=message):
        check(tables)


def test_rejects_incompatible_cold_chain(tables):
    tables["vehicles"].loc[5, "refrigerated"] = False
    with pytest.raises(ValueError, match="Cold chain"):
        feasibility_witness(tables)


def test_rejects_incompatible_vehicle_access(tables):
    tables["customers"].loc[20, "vehicle_access_restrictions"] = "no_heavy"
    with pytest.raises(ValueError, match="Vehicle access"):
        feasibility_witness(tables)


def test_unload_before_pickup_and_intermediate_capacity():
    deliveries = np.array([[3.0], [6.0]])
    pickups = np.array([[5.0], [0.0]])
    with pytest.raises(ValueError, match="Pickup"):
        capacity_trace(deliveries, pickups, np.array([10.0]))
    np.testing.assert_array_equal(
        capacity_trace(deliveries, pickups, np.array([11.0])).ravel(), [9.0, 11.0, 5.0])
