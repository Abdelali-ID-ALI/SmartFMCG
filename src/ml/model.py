"""Trainable traffic predictors and a versioned, reusable inference contract."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.features import CATEGORICAL, NUMERIC, STATIC, TARGET, make_features


@dataclass
class HistoricalAverage:
    """Mean train-period OSRM ratio by local hour and weekend status."""

    ratios: dict[tuple[int, int], float]
    fallback: float

    @classmethod
    def fit(cls, engineered: pd.DataFrame, observed: pd.Series) -> "HistoricalAverage":
        frame = engineered[["weekend", "hour"]].assign(ratio=observed / engineered[STATIC])
        return cls(frame.groupby(["weekend", "hour"]).ratio.mean().to_dict(), float(frame.ratio.mean()))

    def predict(self, engineered: pd.DataFrame) -> np.ndarray:
        ratios = [self.ratios.get((weekend, hour), self.fallback)
                  for weekend, hour in zip(engineered.weekend, engineered.hour, strict=True)]
        return engineered[STATIC].to_numpy() * ratios


@dataclass
class TrafficModel:
    """Persist preprocessing, fitted estimator, feature contract and target mode."""

    pipeline: Pipeline
    zones: dict[str, str]
    target_mode: str
    family: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_features(make_features(frame, self.zones))

    def predict_features(self, features: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.pipeline.predict(features), dtype=float)
        if self.target_mode == "ratio":
            values *= features[STATIC].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite ML prediction")
        return np.maximum(values, 0.01)


def fit_model(features: pd.DataFrame, observed: pd.Series, estimator: RegressorMixin,
              zones: dict[str, str], target_mode: str, family: str,
              columns: tuple[str, ...] = NUMERIC + CATEGORICAL) -> TrafficModel:
    if target_mode not in {"absolute", "ratio"}:
        raise ValueError("Unrecognized prediction target")
    transformers = [("numeric", StandardScaler() if family == "linear" else "passthrough",
                     [name for name in NUMERIC if name in columns])]
    categories = [name for name in CATEGORICAL if name in columns]
    if categories:
        transformers.append(("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories))
    preprocess = ColumnTransformer(transformers).set_output(transform="pandas")
    target = observed if target_mode == "absolute" else observed / features[STATIC]
    pipeline = Pipeline([("preprocess", preprocess), ("regressor", estimator)]).fit(features, target)
    return TrafficModel(pipeline, dict(zones), target_mode, family)
