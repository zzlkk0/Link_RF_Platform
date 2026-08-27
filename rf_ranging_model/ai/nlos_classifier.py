"""Deterministic baseline NLOS classifier for pipeline integration."""

from __future__ import annotations

import math

from .features import RangingFeatures


class HeuristicNLOSClassifier:
    """Transparent baseline; replace with a trained model via the same API."""

    def predict_probability(self, features: RangingFeatures) -> float:
        score = (
            2.2 * features.phase_residual_rms_rad
            + 0.16 * features.amplitude_std_db
            + 0.08 * (features.delay_spread_s / 1.0e-9)
            - 0.045 * (features.snr_db - 15.0)
            - 1.0
        )
        return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score)))))
