"""Small event-driven leaky integrate-and-fire NLOS baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(slots=True)
class EventDrivenNLOSNeuron:
    threshold: float = 1.0
    leak: float = 0.92
    event_scale: float = 0.25

    def run(self, events: ArrayLike) -> tuple[int, float]:
        """Return spike count and final membrane potential."""

        if not 0.0 <= self.leak <= 1.0:
            raise ValueError("leak must lie in [0, 1]")
        if self.threshold <= 0.0:
            raise ValueError("threshold must be greater than zero")
        membrane = 0.0
        spikes = 0
        for event in np.asarray(events, dtype=float).ravel():
            membrane = self.leak * membrane + self.event_scale * max(0.0, float(event))
            if membrane >= self.threshold:
                spikes += 1
                membrane -= self.threshold
        return spikes, float(membrane)

    def predict_probability(self, events: ArrayLike) -> float:
        spikes, membrane = self.run(events)
        return float(1.0 - np.exp(-(spikes + membrane / self.threshold)))
