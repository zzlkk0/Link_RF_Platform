"""Composable RFIC phase, group-delay, IQ, oscillator, and noise errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..signal.noise import add_awgn


@dataclass(frozen=True, slots=True)
class RFICImpairmentConfig:
    phase_offset_rad: float = 0.0
    base_group_delay_s: float = 0.0
    group_delay_ripple_s: float = 0.0
    group_delay_ripple_period_hz: float = 20.0e6
    amplitude_ripple_db: float = 0.0
    iq_gain_mismatch_db: float = 0.0
    iq_phase_mismatch_deg: float = 0.0
    phase_noise_std_deg: float = 0.0
    cfo_hz: float = 0.0
    tone_interval_s: float = 0.0
    snr_db: float | None = None

    def __post_init__(self) -> None:
        if self.group_delay_ripple_period_hz <= 0.0:
            raise ValueError("group_delay_ripple_period_hz must be positive")
        if self.phase_noise_std_deg < 0.0:
            raise ValueError("phase_noise_std_deg must be non-negative")
        if self.tone_interval_s < 0.0:
            raise ValueError("tone_interval_s must be non-negative")


def group_delay_phase_response(
    frequencies_hz: ArrayLike,
    *,
    base_group_delay_s: float = 0.0,
    ripple_s: float = 0.0,
    ripple_period_hz: float = 20.0e6,
    phase_offset_rad: float = 0.0,
) -> NDArray[np.complex128]:
    """Synthesize phase whose derivative has the requested group delay."""

    frequencies = np.asarray(frequencies_hz, dtype=float)
    center = float(np.mean(frequencies))
    delta_frequency = frequencies - center
    phase = phase_offset_rad - 2.0 * np.pi * base_group_delay_s * delta_frequency
    if ripple_s != 0.0:
        phase -= ripple_s * ripple_period_hz * np.sin(
            2.0 * np.pi * delta_frequency / ripple_period_hz
        )
    return np.asarray(np.exp(1j * phase), dtype=np.complex128)


def _apply_iq_imbalance(
    samples: NDArray[np.complex128],
    gain_mismatch_db: float,
    phase_mismatch_deg: float,
) -> NDArray[np.complex128]:
    gain_ratio = 10.0 ** (gain_mismatch_db / 20.0)
    phase_error = np.deg2rad(phase_mismatch_deg)
    alpha = 0.5 * (1.0 + gain_ratio * np.exp(-1j * phase_error))
    beta = 0.5 * (1.0 - gain_ratio * np.exp(1j * phase_error))
    return np.asarray(alpha * samples + beta * np.conj(samples), dtype=np.complex128)


def apply_rfic_impairments(
    frequencies_hz: ArrayLike,
    channel_response: ArrayLike,
    config: RFICImpairmentConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Apply deterministic impairments first, then stochastic receiver errors."""

    cfg = config or RFICImpairmentConfig()
    frequencies = np.asarray(frequencies_hz, dtype=float)
    response = np.asarray(channel_response, dtype=np.complex128)
    if frequencies.ndim != 1 or response.ndim != 1 or frequencies.shape != response.shape:
        raise ValueError("frequencies_hz and channel_response must be matching 1-D arrays")
    generator = rng or np.random.default_rng()
    center = float(np.mean(frequencies))
    delta_frequency = frequencies - center

    amplitude = 10.0 ** (
        cfg.amplitude_ripple_db
        * np.cos(2.0 * np.pi * delta_frequency / cfg.group_delay_ripple_period_hz)
        / 20.0
    )
    phase_response = group_delay_phase_response(
        frequencies,
        base_group_delay_s=cfg.base_group_delay_s,
        ripple_s=cfg.group_delay_ripple_s,
        ripple_period_hz=cfg.group_delay_ripple_period_hz,
        phase_offset_rad=cfg.phase_offset_rad,
    )
    measured = response * amplitude * phase_response

    if cfg.cfo_hz != 0.0 and cfg.tone_interval_s != 0.0:
        tone_time = np.arange(frequencies.size, dtype=float) * cfg.tone_interval_s
        measured *= np.exp(1j * 2.0 * np.pi * cfg.cfo_hz * tone_time)
    if cfg.phase_noise_std_deg > 0.0:
        phase_noise = generator.normal(
            0.0, np.deg2rad(cfg.phase_noise_std_deg), size=frequencies.size
        )
        measured *= np.exp(1j * phase_noise)
    if cfg.iq_gain_mismatch_db != 0.0 or cfg.iq_phase_mismatch_deg != 0.0:
        measured = _apply_iq_imbalance(
            measured, cfg.iq_gain_mismatch_db, cfg.iq_phase_mismatch_deg
        )
    if cfg.snr_db is not None:
        measured = add_awgn(measured, cfg.snr_db, rng=generator)
    return np.asarray(measured, dtype=np.complex128)
