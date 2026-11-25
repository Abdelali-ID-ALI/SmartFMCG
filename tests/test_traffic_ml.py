"""Independent checks of chronological and departure-time model contracts."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.ml.features import INPUTS, STATIC, TARGET, chronological_split, make_features
from src.ml.model import HistoricalAverage, fit_model

ZONES = {"D": "depot", "A": "maarif"}


def trips():
    return pd.DataFrame({
        "timestamp": ["2026-04-06T08:00", "2026-04-06T12:00", "2026-04-07T08:00",
                      "2026-04-08T08:00", "2026-04-11T08:00"],
        "origin_id": ["D"] * 5, "destination_id": ["A"] * 5,
        "road_distance": [2.] * 5, STATIC: [10.] * 5, "road_characteristics": ["arterial"] * 5,
        TARGET: [15., 10., 99., 88., 77.], "congestion_indicators": ["normal"] * 5,
    })


def test_complete_days_and_chronological_order():
    split = chronological_split(trips().iloc[::-1], 1, 1)
    assert [len(split[key]) for key in ("train", "validation", "test")] == [2, 1, 2]
    assert split["train"].timestamp.max() < split["validation"].timestamp.min()
    assert split["validation"].timestamp.max() < split["test"].timestamp.min()
    with pytest.raises(ValueError, match="complete days"):
        chronological_split(trips(), 3, 1)


def test_features_use_calendar_and_ignore_same_trip_outcomes():
    source = trips()
    features = make_features(source, ZONES)
    assert features.weekday_rush.tolist() == [1, 0, 1, 1, 0]
    assert features.central_zone.tolist() == [1] * 5
    assert features.hour_sin.iloc[0] == pytest.approx(np.sin(2 * np.pi * 8 / 24))
    source[TARGET] = -999
    source["congestion_indicators"] = "incident"
    source["departure_hour"] = -1
    source["day_of_week"] = -1
    pd.testing.assert_frame_equal(features, make_features(source, ZONES))
    pd.testing.assert_frame_equal(features, make_features(source.loc[:, list(INPUTS)], ZONES))


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_positive_finite_reference_is_required(bad):
    source = trips()
    source.loc[0, STATIC] = bad
    with pytest.raises(ValueError, match="Invalid"):
        make_features(source, ZONES)


def test_train_only_statistics_and_reload(tmp_path):
    frames = chronological_split(trips(), 1, 1)
    train = frames["train"]
    feat = make_features(train, ZONES)
    historical = HistoricalAverage.fit(feat, train[TARGET])
    validation = frames["validation"]
    test = frames["test"]
    assert historical.predict(make_features(validation, ZONES))[0] == 15
    assert historical.predict(make_features(test, ZONES))[-1] == 12.5
    model = fit_model(feat, train[TARGET], LinearRegression(), ZONES, "ratio", "linear")
    preprocess = model.pipeline["preprocess"]
    assert preprocess.named_transformers_["numeric"].mean_[2] == 10
    expected = model.predict(test.loc[:, list(INPUTS)])
    assert np.all(np.isfinite(expected) & (expected > 0))
    changed = test.assign(observed_travel_time=10000, congestion_indicators="incident")
    np.testing.assert_array_equal(expected, model.predict(changed))
    saved = tmp_path / "model.joblib"
    joblib.dump(model, saved)
    np.testing.assert_array_equal(expected, joblib.load(saved).predict(test))
