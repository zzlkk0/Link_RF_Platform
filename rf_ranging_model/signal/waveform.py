"""Reference UWB pulse generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian_monocycle(
    time_s: ArrayLike,
    *,
    center_s: float = 0.0,
    width_s: float = 0.5e-9,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Return the first derivative of a Gaussian pulse."""

    if width_s <= 0.0:
        raise ValueError("width_s must be greater than zero")
    time = np.asarray(time_s, dtype=float)
    normalized_time = (time - center_s) / width_s
    pulse = -normalized_time * np.exp(-0.5 * normalized_time**2)
    if normalize:
        peak = float(np.max(np.abs(pulse))) if pulse.size else 0.0
        if peak > 0.0:
            pulse = pulse / peak
    return np.asarray(pulse, dtype=float)
