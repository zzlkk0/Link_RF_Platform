"""Minimal Touchstone S2P S21 importer with linear complex interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


_UNIT_SCALE = {"hz": 1.0, "khz": 1.0e3, "mhz": 1.0e6, "ghz": 1.0e9}


def _pair_to_complex(first: float, second: float, representation: str) -> complex:
    if representation == "ri":
        return complex(first, second)
    magnitude = first if representation == "ma" else 10.0 ** (first / 20.0)
    return complex(magnitude * np.exp(1j * np.deg2rad(second)))


def load_s2p_s21(
    path: str | Path,
    target_frequencies_hz: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Load S21 from a simple 2-port Touchstone v1 file.

    Supported data encodings are RI, MA, and DB. Wrapped multi-line records are
    intentionally rejected so malformed RF data does not silently enter a run.
    """

    unit = "ghz"
    representation = "ma"
    frequencies: list[float] = []
    s21_values: list[complex] = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            tokens = line[1:].lower().split()
            if tokens:
                unit = tokens[0]
            for candidate in ("ri", "ma", "db"):
                if candidate in tokens:
                    representation = candidate
                    break
            continue
        values = line.split("!", maxsplit=1)[0].split()
        if len(values) < 9:
            raise ValueError("only one-line 9-column Touchstone v1 records are supported")
        numbers = [float(value) for value in values[:9]]
        frequencies.append(numbers[0] * _UNIT_SCALE[unit])
        s21_values.append(_pair_to_complex(numbers[3], numbers[4], representation))

    if not frequencies:
        raise ValueError(f"no S-parameter records found in {path}")
    source_frequencies = np.asarray(frequencies, dtype=float)
    source_s21 = np.asarray(s21_values, dtype=np.complex128)
    if target_frequencies_hz is None:
        return source_frequencies, source_s21

    target = np.asarray(target_frequencies_hz, dtype=float)
    real = np.interp(target, source_frequencies, source_s21.real)
    imag = np.interp(target, source_frequencies, source_s21.imag)
    return target, np.asarray(real + 1j * imag, dtype=np.complex128)
