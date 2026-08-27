"""Level-0 Friis free-space propagation model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import C0
from ..types import PropagationPath


def _validate_distance(distance_m: float) -> None:
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("distance_m must be finite and greater than zero")


def free_space_response(
    frequencies_hz: ArrayLike,
    distance_m: float,
    *,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
    phase_offset_rad: float = 0.0,
) -> NDArray[np.complex128]:
    """Return the one-way complex free-space transfer function.

    Amplitude follows the Friis voltage ratio lambda/(4*pi*d). Antenna gains
    are power gains, so their product enters as a square-root voltage factor.
    """

    _validate_distance(distance_m)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("all frequencies must be finite and greater than zero")

    antenna_voltage_gain = 10.0 ** ((tx_gain_db + rx_gain_db) / 20.0)
    amplitude = antenna_voltage_gain * C0 / (4.0 * np.pi * frequencies * distance_m)
    delay_s = distance_m / C0
    phase = phase_offset_rad - 2.0 * np.pi * frequencies * delay_s
    return np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex128)


def free_space_path(
    distance_m: float,
    reference_frequency_hz: float,
    *,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
    phase_offset_rad: float = 0.0,
    label: str = "LOS",
) -> PropagationPath:
    """Represent free-space propagation as a reusable narrowband path."""

    _validate_distance(distance_m)
    if reference_frequency_hz <= 0.0:
        raise ValueError("reference_frequency_hz must be greater than zero")
    voltage_gain = (
        10.0 ** ((tx_gain_db + rx_gain_db) / 20.0)
        * C0
        / (4.0 * np.pi * reference_frequency_hz * distance_m)
    )
    return PropagationPath(
        delay_s=distance_m / C0,
        complex_gain=voltage_gain * np.exp(1j * phase_offset_rad),
        label=label,
    )
