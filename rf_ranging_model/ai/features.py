"""Stable feature contract for later ML and SNN replacement."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class RangingFeatures:
    phase_residual_rms_rad: float
    amplitude_std_db: float
    delay_spread_s: float
    snr_db: float
    cfo_hz: float = 0.0
    iq_gain_mismatch_db: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def extract_features(
    response: ArrayLike,
    *,
    phase_residual_rms_rad: float,
    delay_spread_s: float,
    snr_db: float,
    cfo_hz: float = 0.0,
    iq_gain_mismatch_db: float = 0.0,
) -> RangingFeatures:
    channel = np.asarray(response, dtype=np.complex128)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(channel), 1.0e-15))
    return RangingFeatures(
        phase_residual_rms_rad=float(phase_residual_rms_rad),
        amplitude_std_db=float(np.std(magnitude_db)),
        delay_spread_s=float(delay_spread_s),
        snr_db=float(snr_db),
        cfo_hz=float(cfo_hz),
        iq_gain_mismatch_db=float(iq_gain_mismatch_db),
    )
