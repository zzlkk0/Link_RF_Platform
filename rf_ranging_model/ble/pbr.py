"""Phase-slope ranging across stepped BLE tones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import C0


@dataclass(frozen=True, slots=True)
class PBRResult:
    distance_m: float
    delay_s: float
    phase_slope_rad_per_hz: float
    intercept_rad: float
    residual_rms_rad: float
    unwrapped_phase_rad: NDArray[np.float64]
    fitted_phase_rad: NDArray[np.float64]
    coarse_delay_s: float | None = None
    ambiguity_peak_ratio: float | None = None
    estimator: str = "linear_unwrap"
    # Appended after the legacy positional fields to preserve callers that
    # constructed PBRResult positionally before ambiguity metadata existed.
    unambiguous_distance_m: float | None = None


def _frequency_grid_spacing_hz(
    frequencies_hz: NDArray[np.float64],
) -> float | None:
    """Return the fundamental integer-Hz tone grid when it is identifiable."""

    differences_hz = np.diff(frequencies_hz)
    rounded_hz = np.rint(differences_hz)
    if not np.allclose(differences_hz, rounded_hz, rtol=0.0, atol=1.0e-3):
        return None
    integer_differences = np.abs(rounded_hz.astype(np.int64))
    spacing_hz = int(np.gcd.reduce(integer_differences))
    return float(spacing_hz) if spacing_hz > 0 else None


def _linear_phase_fit(
    frequencies_hz: NDArray[np.float64],
    phase_rad: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, NDArray[np.float64]]:
    centered_frequency = frequencies_hz - np.average(frequencies_hz, weights=weights)
    design = np.column_stack((centered_frequency, np.ones_like(centered_frequency)))
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_phase = phase_rad * np.sqrt(weights)
    slope, centered_intercept = np.linalg.lstsq(
        weighted_design, weighted_phase, rcond=None
    )[0]
    fitted = slope * centered_frequency + centered_intercept
    intercept = centered_intercept - slope * np.average(frequencies_hz, weights=weights)
    return float(slope), float(intercept), fitted


def _coarse_delay_search(
    frequencies_hz: NDArray[np.float64],
    response: NDArray[np.complex128],
    weights: NDArray[np.float64],
    *,
    maximum_delay_s: float,
    oversampling: int,
) -> tuple[float, float]:
    """Find the global delay peak without assuming adjacent phase continuity."""

    bandwidth_hz = float(np.ptp(frequencies_hz))
    if bandwidth_hz <= 0.0:
        raise ValueError("frequency span must be greater than zero")
    if maximum_delay_s <= 0.0:
        raise ValueError("maximum_delay_s must be greater than zero")
    if oversampling < 4:
        raise ValueError("coarse_oversampling must be at least four")

    delay_step_s = 1.0 / (bandwidth_hz * oversampling)
    delay_count = max(3, int(np.ceil(maximum_delay_s / delay_step_s)) + 1)
    candidate_delays = np.linspace(0.0, maximum_delay_s, delay_count)
    center_frequency_hz = float(np.average(frequencies_hz, weights=weights))
    centered_frequencies = frequencies_hz - center_frequency_hz
    # The fine WLS fit uses phase-variance weights.  In the coarse search,
    # normalize away channel magnitude first so multiplying by those weights
    # does not accidentally produce an |H|^3 contribution in deep fading.
    unit_phase_response = response / np.maximum(
        np.abs(response), np.finfo(float).tiny
    )
    weighted_response = weights * unit_phase_response

    scores = np.empty(delay_count, dtype=float)
    chunk_size = 2048
    for start in range(0, delay_count, chunk_size):
        stop = min(start + chunk_size, delay_count)
        steering = np.exp(
            1j
            * 2.0
            * np.pi
            * candidate_delays[start:stop, None]
            * centered_frequencies[None, :]
        )
        scores[start:stop] = np.abs(steering @ weighted_response)

    peak_index = int(np.argmax(scores))
    coarse_delay_s = float(candidate_delays[peak_index])
    exclusion_radius = max(1, oversampling)
    second_peak_scores = scores.copy()
    second_peak_scores[
        max(0, peak_index - exclusion_radius) : min(
            delay_count, peak_index + exclusion_radius + 1
        )
    ] = 0.0
    second_peak = float(np.max(second_peak_scores))
    peak_ratio = (
        float(scores[peak_index] / second_peak)
        if second_peak > 0.0
        else float("inf")
    )
    return coarse_delay_s, peak_ratio


def _ambiguity_aware_phase_fit(
    frequencies_hz: NDArray[np.float64],
    response: NDArray[np.complex128],
    weights: NDArray[np.float64],
    *,
    maximum_delay_s: float,
    coarse_oversampling: int,
) -> tuple[
    float,
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
]:
    coarse_delay_s, peak_ratio = _coarse_delay_search(
        frequencies_hz,
        response,
        weights,
        maximum_delay_s=maximum_delay_s,
        oversampling=coarse_oversampling,
    )
    center_frequency_hz = float(np.average(frequencies_hz, weights=weights))
    centered_frequencies = frequencies_hz - center_frequency_hz
    derotated = response * np.exp(
        1j * 2.0 * np.pi * centered_frequencies * coarse_delay_s
    )
    residual_phase = np.unwrap(np.angle(derotated))
    unwrapped_phase = (
        residual_phase
        - 2.0 * np.pi * centered_frequencies * coarse_delay_s
    )
    slope, intercept, fitted_phase = _linear_phase_fit(
        frequencies_hz, unwrapped_phase, weights
    )
    return (
        slope,
        intercept,
        np.asarray(unwrapped_phase, dtype=float),
        np.asarray(fitted_phase, dtype=float),
        coarse_delay_s,
        peak_ratio,
    )


def estimate_distance(
    frequencies_hz: ArrayLike,
    measured_response: ArrayLike,
    *,
    path_factor: float = 1.0,
    magnitude_weighting: bool = True,
    estimator: Literal["ambiguity_aware", "linear_unwrap"] = "ambiguity_aware",
    max_distance_m: float = 149.0,
    coarse_oversampling: int = 16,
) -> PBRResult:
    """Estimate propagation distance from d(phi)/df.

    path_factor=1 matches the one-way H(f) model in the project specification.
    Use path_factor=2 when the supplied phase is a round-trip measurement.
    The default ambiguity-aware estimator performs a global delay search before
    residual phase fitting, so gaps in a Bluetooth CS channel map do not cause
    adjacent-sample phase-unwrapping failures.
    """

    frequencies = np.asarray(frequencies_hz, dtype=float)
    response = np.asarray(measured_response, dtype=np.complex128)
    if frequencies.ndim != 1 or response.ndim != 1 or frequencies.shape != response.shape:
        raise ValueError("frequencies_hz and measured_response must be matching 1-D arrays")
    if frequencies.size < 3:
        raise ValueError("at least three frequency tones are required")
    if not np.all(np.isfinite(frequencies)) or not np.all(
        np.isfinite(response.real) & np.isfinite(response.imag)
    ):
        raise ValueError(
            "frequencies_hz and measured_response must contain only finite values"
        )
    if path_factor <= 0.0:
        raise ValueError("path_factor must be greater than zero")
    if max_distance_m <= 0.0:
        raise ValueError("max_distance_m must be greater than zero")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("frequencies_hz must be strictly increasing")

    frequency_grid_spacing_hz = _frequency_grid_spacing_hz(frequencies)
    unambiguous_distance_m = (
        C0 / (path_factor * frequency_grid_spacing_hz)
        if frequency_grid_spacing_hz is not None
        else None
    )
    if (
        estimator == "ambiguity_aware"
        and unambiguous_distance_m is not None
        and max_distance_m >= unambiguous_distance_m
    ):
        raise ValueError(
            "max_distance_m must be below the tone-grid unambiguous range "
            f"({unambiguous_distance_m:.9g} m for path_factor={path_factor:g})"
        )

    magnitudes = np.abs(response)
    if float(np.max(magnitudes)) <= 0.0:
        raise ValueError("measured_response must contain a non-zero value")
    if magnitude_weighting and float(np.max(magnitudes)) > 0.0:
        weights = np.maximum(magnitudes / np.max(magnitudes), 1.0e-6) ** 2
    else:
        weights = np.ones_like(frequencies)

    coarse_delay_s: float | None = None
    ambiguity_peak_ratio: float | None = None
    if estimator == "linear_unwrap":
        phase = np.unwrap(np.angle(response))
        slope, intercept, fitted = _linear_phase_fit(frequencies, phase, weights)
    elif estimator == "ambiguity_aware":
        maximum_delay_s = path_factor * max_distance_m / C0
        (
            slope,
            intercept,
            phase,
            fitted,
            coarse_delay_s,
            ambiguity_peak_ratio,
        ) = _ambiguity_aware_phase_fit(
            frequencies,
            response,
            weights,
            maximum_delay_s=maximum_delay_s,
            coarse_oversampling=coarse_oversampling,
        )
    else:
        raise ValueError("estimator must be 'ambiguity_aware' or 'linear_unwrap'")

    residual_rms = float(
        np.sqrt(np.average((phase - fitted) ** 2, weights=weights))
    )
    delay_s = -slope / (2.0 * np.pi)
    distance_m = C0 * delay_s / path_factor
    if estimator == "ambiguity_aware":
        distance_tolerance_m = max(1.0e-9, max_distance_m * 1.0e-12)
        if not -distance_tolerance_m <= distance_m <= (
            max_distance_m + distance_tolerance_m
        ):
            raise ValueError(
                "estimated distance lies outside the configured search prior"
            )
    return PBRResult(
        distance_m=float(distance_m),
        delay_s=float(delay_s),
        phase_slope_rad_per_hz=slope,
        intercept_rad=intercept,
        residual_rms_rad=residual_rms,
        unwrapped_phase_rad=np.asarray(phase, dtype=float),
        fitted_phase_rad=np.asarray(fitted, dtype=float),
        coarse_delay_s=coarse_delay_s,
        ambiguity_peak_ratio=ambiguity_peak_ratio,
        unambiguous_distance_m=unambiguous_distance_m,
        estimator=estimator,
    )


def phase_linearity_metrics(
    frequencies_hz: ArrayLike,
    response: ArrayLike,
) -> dict[str, float]:
    result = estimate_distance(
        frequencies_hz, response, magnitude_weighting=False
    )
    return {
        "phase_residual_rms_rad": result.residual_rms_rad,
        "phase_residual_rms_deg": float(np.rad2deg(result.residual_rms_rad)),
        "estimated_delay_s": result.delay_s,
    }
