"""Lightweight feature-based distance correction interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import RangingFeatures


@dataclass(slots=True)
class LinearDistanceCorrector:
    """A serializable baseline that can be calibrated from measured data."""

    bias_m: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)

    def predict_delta_m(self, features: RangingFeatures) -> float:
        values = features.as_dict()
        return float(
            self.bias_m
            + sum(weight * values.get(name, 0.0) for name, weight in self.weights.items())
        )

    def correct(self, raw_distance_m: float, features: RangingFeatures) -> float:
        return float(raw_distance_m + self.predict_delta_m(features))
