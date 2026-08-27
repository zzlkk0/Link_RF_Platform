"""Bluetooth Channel Sounding RF channel definitions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


CS_FIRST_FREQUENCY_HZ = 2_402_000_000.0
CS_CHANNEL_SPACING_HZ = 1_000_000.0
CS_CHANNEL_COUNT = 79
CS_DISALLOWED_CHANNEL_INDICES = frozenset({0, 1, 23, 24, 25, 77, 78})


def cs_channel_frequency_hz(channel_index: int) -> float:
    """Return the center frequency for a Bluetooth CS channel index."""

    if isinstance(channel_index, bool) or not isinstance(channel_index, (int, np.integer)):
        raise TypeError("channel_index must be an integer")
    if not 0 <= int(channel_index) < CS_CHANNEL_COUNT:
        raise ValueError("channel_index must lie in [0, 78]")
    return CS_FIRST_FREQUENCY_HZ + int(channel_index) * CS_CHANNEL_SPACING_HZ


def allowed_cs_channel_indices() -> NDArray[np.int64]:
    """Return the 72 channel indices allowed for CS communication."""

    return np.asarray(
        [
            index
            for index in range(CS_CHANNEL_COUNT)
            if index not in CS_DISALLOWED_CHANNEL_INDICES
        ],
        dtype=np.int64,
    )


def allowed_cs_frequencies_hz() -> NDArray[np.float64]:
    """Return the 72 allowed CS tone-center frequencies."""

    indices = allowed_cs_channel_indices()
    return np.asarray(
        CS_FIRST_FREQUENCY_HZ + indices * CS_CHANNEL_SPACING_HZ,
        dtype=float,
    )
