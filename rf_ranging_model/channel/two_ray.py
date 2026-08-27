"""Level-1 direct-plus-reflected two-ray model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import C0
from ..types import PropagationPath
from .free_space import free_space_response


@dataclass(frozen=True, slots=True)
class TwoRayGeometry:
    horizontal_distance_m: float
    tx_height_m: float = 1.2
    rx_height_m: float = 1.2

    def __post_init__(self) -> None:
        values = (self.horizontal_distance_m, self.tx_height_m, self.rx_height_m)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("geometry values must be finite")
        if self.horizontal_distance_m <= 0.0:
            raise ValueError("horizontal_distance_m must be greater than zero")
        if self.tx_height_m < 0.0 or self.rx_height_m < 0.0:
            raise ValueError("antenna heights must be non-negative")

    @property
    def direct_distance_m(self) -> float:
        return float(np.hypot(self.horizontal_distance_m, self.tx_height_m - self.rx_height_m))

    @property
    def reflected_distance_m(self) -> float:
        return float(np.hypot(self.horizontal_distance_m, self.tx_height_m + self.rx_height_m))


def two_ray_response(
    frequencies_hz: ArrayLike,
    geometry: TwoRayGeometry,
    *,
    reflection_coefficient: complex = -0.65 + 0.0j,
) -> NDArray[np.complex128]:
    """Return the physical direct and ground-reflected response."""

    direct = free_space_response(frequencies_hz, geometry.direct_distance_m)
    reflected = reflection_coefficient * free_space_response(
        frequencies_hz, geometry.reflected_distance_m
    )
    return np.asarray(direct + reflected, dtype=np.complex128)


def two_ray_paths(
    geometry: TwoRayGeometry,
    reference_frequency_hz: float,
    *,
    reflection_coefficient: complex = -0.65 + 0.0j,
) -> list[PropagationPath]:
    """Return the two rays in the generic path representation."""

    if reference_frequency_hz <= 0.0:
        raise ValueError("reference_frequency_hz must be greater than zero")

    direct_distance = geometry.direct_distance_m
    reflected_distance = geometry.reflected_distance_m
    direct_gain = C0 / (4.0 * np.pi * reference_frequency_hz * direct_distance)
    reflected_gain = (
        reflection_coefficient
        * C0
        / (4.0 * np.pi * reference_frequency_hz * reflected_distance)
    )
    return [
        PropagationPath(direct_distance / C0, direct_gain, "direct"),
        PropagationPath(reflected_distance / C0, reflected_gain, "reflection"),
    ]
