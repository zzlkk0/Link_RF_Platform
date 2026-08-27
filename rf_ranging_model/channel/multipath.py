"""Deterministic multipath CFR and CIR generation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..types import PropagationPath


def _require_paths(paths: Sequence[PropagationPath]) -> None:
    if not paths:
        raise ValueError("at least one propagation path is required")


def frequency_response(
    frequencies_hz: ArrayLike,
    paths: Sequence[PropagationPath],
    *,
    normalize: bool = False,
) -> NDArray[np.complex128]:
    """Sum all path phasors to produce H(f)."""

    _require_paths(paths)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(frequencies)):
        raise ValueError("frequencies_hz contains a non-finite value")

    delays = np.asarray([path.delay_s for path in paths], dtype=float)
    gains = np.asarray([path.complex_gain for path in paths], dtype=np.complex128)
    response = np.exp(-1j * 2.0 * np.pi * frequencies[:, None] * delays[None, :]) @ gains
    if normalize:
        peak = float(np.max(np.abs(response)))
        if peak > 0.0:
            response = response / peak
    return np.asarray(response, dtype=np.complex128)


def sampled_cir(
    paths: Sequence[PropagationPath],
    sample_rate_hz: float,
    duration_s: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Deposit path impulses on a sampled time grid with linear interpolation."""

    _require_paths(paths)
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be greater than zero")
    max_delay = max(path.delay_s for path in paths)
    if duration_s is None:
        duration_s = max_delay + 8.0 / sample_rate_hz
    if duration_s <= max_delay:
        raise ValueError("duration_s must extend beyond the last path")

    sample_count = int(np.ceil(duration_s * sample_rate_hz)) + 1
    times = np.arange(sample_count, dtype=float) / sample_rate_hz
    cir = np.zeros(sample_count, dtype=np.complex128)
    for path in paths:
        fractional_index = path.delay_s * sample_rate_hz
        lower = int(np.floor(fractional_index))
        fraction = fractional_index - lower
        if lower < sample_count:
            cir[lower] += path.complex_gain * (1.0 - fraction)
        if fraction > 0.0 and lower + 1 < sample_count:
            cir[lower + 1] += path.complex_gain * fraction
    return times, cir


def delay_spread_rms(paths: Sequence[PropagationPath]) -> float:
    """Return the power-weighted RMS delay spread in seconds."""

    _require_paths(paths)
    powers = np.asarray([path.power_gain for path in paths], dtype=float)
    delays = np.asarray([path.delay_s for path in paths], dtype=float)
    total_power = float(np.sum(powers))
    if total_power <= 0.0:
        return 0.0
    mean_delay = float(np.sum(powers * delays) / total_power)
    return float(np.sqrt(np.sum(powers * (delays - mean_delay) ** 2) / total_power))
