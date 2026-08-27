"""Complex additive white Gaussian noise."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def add_awgn(
    samples: ArrayLike,
    snr_db: float,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Add circular complex AWGN at the requested average signal SNR."""

    signal = np.asarray(samples, dtype=np.complex128)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    if signal_power == 0.0:
        return signal.copy()
    generator = rng or np.random.default_rng()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        generator.standard_normal(signal.shape)
        + 1j * generator.standard_normal(signal.shape)
    )
    return np.asarray(signal + noise, dtype=np.complex128)
