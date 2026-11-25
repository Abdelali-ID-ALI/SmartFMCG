"""Chronological partitions and departure-time traffic features."""

import numpy as np
import pandas as pd

TARGET = "observed_travel_time"
STATIC = "static_travel_time"
INPUTS = ("timestamp", "origin_id", "destination_id", "road_distance", STATIC, "road_characteristics")
NUMERIC = ("road_distance", STATIC, "hour", "hour_sin", "hour_cos", "day_sin", "day_cos",
           "weekend", "rush", "weekday_rush", "central_zone")
CATEGORICAL = ("origin_zone", "destination_zone", "road_characteristics")


def chronological_split(frame: pd.DataFrame, train_days: int, validation_days: int) -> dict[str, pd.DataFrame]:
    """Split on local calendar dates, never on individual rows."""
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamps.isna().any():
        raise ValueError("Missing timestamp")
    day = timestamps.dt.normalize()
    days = sorted(day.unique())
    if train_days < 1 or validation_days < 1 or train_days + validation_days >= len(days):
        raise ValueError("All three partitions require complete days")
    partitions = (days[:train_days], days[train_days:train_days + validation_days],
                  days[train_days + validation_days:])
    return {label: frame.loc[day.isin(selected)].sort_values("timestamp", kind="stable").copy()
            for label, selected in zip(("train", "validation", "test"), partitions, strict=True)}


def make_features(frame: pd.DataFrame, zones: dict[str, str]) -> pd.DataFrame:
    """Read only the explicit departure-time input contract."""
    features = frame.loc[:, list(INPUTS)]
    clock = pd.to_datetime(features["timestamp"], errors="raise")
    origin = features.origin_id.map(zones)
    destination = features.destination_id.map(zones)
    road = features.road_characteristics
    static = features[[STATIC, "road_distance"]].to_numpy(dtype=float)
    if (clock.isna().any() or origin.isna().any() or destination.isna().any()
            or not road.isin(["local", "arterial"]).all()
            or not (np.isfinite(static) & (static > 0)).all()):
        raise ValueError("Invalid departure-time input")
    day, hour = clock.dt.day_of_week, clock.dt.hour
    fraction = hour + clock.dt.minute / 60
    weekend = day.ge(5)
    rush = hour.isin([7, 8, 9, 16, 17, 18])
    central = origin.isin(["maarif", "bourgogne"]) | destination.isin(["maarif", "bourgogne"])
    return pd.DataFrame({
        "road_distance": static[:, 1], STATIC: static[:, 0], "hour": hour,
        "hour_sin": np.sin(2*np.pi*fraction/24), "hour_cos": np.cos(2*np.pi*fraction/24),
        "day_sin": np.sin(2*np.pi*day/7), "day_cos": np.cos(2*np.pi*day/7),
        "weekend": weekend.astype(int), "rush": rush.astype(int),
        "weekday_rush": (~weekend & rush).astype(int), "central_zone": central.astype(int),
        "origin_zone": origin, "destination_zone": destination, "road_characteristics": road,
    }, index=frame.index)
