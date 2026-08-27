"""Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PropagationPath:
    """One specular propagation path in the complex baseband model.

    The complex gain excludes the frequency-dependent delay phase. That phase
    is applied by the channel engine as exp(-j*2*pi*f*delay_s).
    """

    delay_s: float
    complex_gain: complex = 1.0 + 0.0j
    label: str = "path"

    def __post_init__(self) -> None:
        if self.delay_s < 0.0:
            raise ValueError("delay_s must be non-negative")

    @property
    def power_gain(self) -> float:
        return float(abs(self.complex_gain) ** 2)
