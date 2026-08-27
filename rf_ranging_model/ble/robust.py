"""Quality-gated BLE phase-based ranging.

The legacy :func:`estimate_distance` function deliberately remains a numeric
estimator.  This module adds the policy layer needed by applications: it
validates the actual tone map, measures independent delay peaks, and returns an
explicit availability decision instead of silently trusting every finite
distance estimate.  A matrix-pencil model is also fitted on the longest
uniform tone run.  When BIC supports more than one path, the earliest
significant component is reported instead of the effective phase slope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import C0
from .pbr import PBRResult, _frequency_grid_spacing_hz, estimate_distance


RobustPBRStatus = Literal["available", "rejected", "invalid_input"]


@dataclass(frozen=True, slots=True)
class RobustPBRConfig:
    """Configurable quality gates for ambiguity-aware PBR.

    ``max_distance_m`` is a prior, not merely a plotting limit.  It must lie
    strictly below the ambiguity range implied by the *actual* tone map.
    Missing tones may therefore reduce the legal prior when their channel-index
    differences have a common divisor greater than one.
    """

    path_factor: float = 1.0
    max_distance_m: float = 149.0
    coarse_oversampling: int = 16
    magnitude_weighting: bool = True
    minimum_tone_count: int = 8
    minimum_peak_ratio: float = 1.25
    maximum_residual_rms_rad: float = 0.60
    minimum_confidence: float = 0.30
    independent_peak_separation_widths: float = 1.0
    sort_frequencies: bool = True
    require_identifiable_frequency_grid: bool = True
    multipath_model_enabled: bool = True
    multipath_max_model_order: int = 6
    multipath_minimum_contiguous_tones: int = 12
    multipath_candidate_amplitude_floor_ratio: float = 0.20
    multipath_minimum_early_path_amplitude_ratio: float = 0.40
    multipath_maximum_reconstruction_residual: float = 0.30
    multipath_strong_maximum_reconstruction_residual: float = 0.25
    multipath_maximum_path_competition_ratio: float = 0.98
    multipath_unresolved_separation_widths: float = 1.0
    multipath_unresolved_minimum_legacy_disagreement_m: float = 1.0
    multipath_challenger_maximum_noise_subspace_ratio: float = 0.10
    multipath_challenger_minimum_secondary_amplitude_ratio: float = 0.10
    multipath_challenger_minimum_legacy_disagreement_m: float = 0.10

    def __post_init__(self) -> None:
        if not np.isfinite(self.path_factor) or self.path_factor <= 0.0:
            raise ValueError("path_factor must be finite and greater than zero")
        if not np.isfinite(self.max_distance_m) or self.max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be finite and greater than zero")
        if self.coarse_oversampling < 4:
            raise ValueError("coarse_oversampling must be at least four")
        if self.minimum_tone_count < 3:
            raise ValueError("minimum_tone_count must be at least three")
        if (
            not np.isfinite(self.minimum_peak_ratio)
            or self.minimum_peak_ratio < 1.0
        ):
            raise ValueError("minimum_peak_ratio must be finite and at least one")
        if (
            not np.isfinite(self.maximum_residual_rms_rad)
            or self.maximum_residual_rms_rad <= 0.0
        ):
            raise ValueError(
                "maximum_residual_rms_rad must be finite and greater than zero"
            )
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must lie in [0, 1]")
        if (
            not np.isfinite(self.independent_peak_separation_widths)
            or self.independent_peak_separation_widths <= 0.0
        ):
            raise ValueError(
                "independent_peak_separation_widths must be finite and positive"
            )
        if self.multipath_max_model_order < 2:
            raise ValueError("multipath_max_model_order must be at least two")
        if self.multipath_minimum_contiguous_tones < 8:
            raise ValueError(
                "multipath_minimum_contiguous_tones must be at least eight"
            )
        if not 0.0 < self.multipath_candidate_amplitude_floor_ratio <= 1.0:
            raise ValueError(
                "multipath_candidate_amplitude_floor_ratio must lie in (0, 1]"
            )
        if not 0.0 < self.multipath_minimum_early_path_amplitude_ratio <= 1.0:
            raise ValueError(
                "multipath_minimum_early_path_amplitude_ratio must lie in (0, 1]"
            )
        if (
            self.multipath_minimum_early_path_amplitude_ratio
            < self.multipath_candidate_amplitude_floor_ratio
        ):
            raise ValueError(
                "multipath early-path gate cannot be below the candidate floor"
            )
        if (
            not np.isfinite(self.multipath_maximum_reconstruction_residual)
            or self.multipath_maximum_reconstruction_residual <= 0.0
        ):
            raise ValueError(
                "multipath_maximum_reconstruction_residual must be finite and "
                "positive"
            )
        if (
            not np.isfinite(
                self.multipath_strong_maximum_reconstruction_residual
            )
            or self.multipath_strong_maximum_reconstruction_residual <= 0.0
            or self.multipath_strong_maximum_reconstruction_residual
            > self.multipath_maximum_reconstruction_residual
        ):
            raise ValueError(
                "multipath_strong_maximum_reconstruction_residual must be "
                "positive and no greater than the general reconstruction gate"
            )
        if not 0.0 < self.multipath_maximum_path_competition_ratio <= 1.0:
            raise ValueError(
                "multipath_maximum_path_competition_ratio must lie in (0, 1]"
            )
        for name, value in (
            (
                "multipath_unresolved_separation_widths",
                self.multipath_unresolved_separation_widths,
            ),
            (
                "multipath_unresolved_minimum_legacy_disagreement_m",
                self.multipath_unresolved_minimum_legacy_disagreement_m,
            ),
            (
                "multipath_challenger_minimum_legacy_disagreement_m",
                self.multipath_challenger_minimum_legacy_disagreement_m,
            ),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            (
                "multipath_challenger_maximum_noise_subspace_ratio",
                self.multipath_challenger_maximum_noise_subspace_ratio,
            ),
            (
                "multipath_challenger_minimum_secondary_amplitude_ratio",
                self.multipath_challenger_minimum_secondary_amplitude_ratio,
            ),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class RobustPBRResult:
    """PBR estimate plus an explicit usability decision and its evidence."""

    availability: bool
    status: RobustPBRStatus
    rejection_reason: str | None
    confidence: float
    distance_m: float | None
    delay_s: float | None
    residual_rms_rad: float | None
    peak_ratio: float | None
    frequency_grid_spacing_hz: float | None
    unambiguous_distance_m: float | None
    primary_peak_delay_s: float | None
    secondary_peak_delay_s: float | None
    primary_peak_score: float | None
    secondary_peak_score: float | None
    independent_peak_count: int
    used_tone_count: int
    input_was_sorted: bool
    quality_flags: tuple[str, ...]
    estimate: PBRResult | None = None
    estimator_mode: str = "single_path_phase_fit"
    model_order: int = 0
    path_delays_s: tuple[float, ...] = ()
    path_amplitudes: tuple[float, ...] = ()
    reconstruction_residual: float | None = None
    selected_path_amplitude_ratio: float | None = None
    legacy_disagreement_m: float | None = None
    minimum_path_separation_s: float | None = None
    subspace_noise_ratio: float | None = None
    challenger_delay_s: float | None = None
    challenger_secondary_amplitude_ratio: float | None = None
    challenger_disagreement_m: float | None = None
    model_bic: float | None = None
    path_competition_ratio: float | None = None
    strong_multipath_evidence: bool = False
    model_rejection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _DelayPeakProfile:
    primary_delay_s: float
    secondary_delay_s: float | None
    primary_score: float
    secondary_score: float
    peak_ratio: float
    independent_peak_count: int


@dataclass(frozen=True, slots=True)
class _MultipathModel:
    order: int
    delays_s: tuple[float, ...]
    amplitudes: tuple[float, ...]
    reconstruction_residual: float
    selected_delay_s: float
    selected_amplitude_ratio: float
    bic: float
    minimum_significant_separation_s: float | None
    subspace_noise_ratio: float
    challenger_delay_s: float | None
    challenger_secondary_amplitude_ratio: float | None


def _empty_result(
    *,
    status: RobustPBRStatus,
    reason: str,
    tone_count: int = 0,
    input_was_sorted: bool = False,
    grid_spacing_hz: float | None = None,
    unambiguous_distance_m: float | None = None,
    profile: _DelayPeakProfile | None = None,
) -> RobustPBRResult:
    return RobustPBRResult(
        availability=False,
        status=status,
        rejection_reason=reason,
        confidence=0.0,
        distance_m=None,
        delay_s=None,
        residual_rms_rad=None,
        peak_ratio=profile.peak_ratio if profile is not None else None,
        frequency_grid_spacing_hz=grid_spacing_hz,
        unambiguous_distance_m=unambiguous_distance_m,
        primary_peak_delay_s=(
            profile.primary_delay_s if profile is not None else None
        ),
        secondary_peak_delay_s=(
            profile.secondary_delay_s if profile is not None else None
        ),
        primary_peak_score=(
            profile.primary_score if profile is not None else None
        ),
        secondary_peak_score=(
            profile.secondary_score if profile is not None else None
        ),
        independent_peak_count=(
            profile.independent_peak_count if profile is not None else 0
        ),
        used_tone_count=tone_count,
        input_was_sorted=input_was_sorted,
        quality_flags=(reason,),
        estimate=None,
    )


def _phase_variance_weights(
    response: NDArray[np.complex128], magnitude_weighting: bool
) -> NDArray[np.float64]:
    if not magnitude_weighting:
        return np.ones(response.size, dtype=float)
    magnitudes = np.abs(response)
    normalized = magnitudes / float(np.max(magnitudes))
    return np.asarray(np.maximum(normalized, 1.0e-6) ** 2, dtype=float)


def _longest_uniform_tone_run(
    frequencies_hz: NDArray[np.float64],
    response: NDArray[np.complex128],
    grid_spacing_hz: float,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return the longest run whose adjacent tones occupy consecutive bins."""

    consecutive = np.isclose(
        np.diff(frequencies_hz),
        grid_spacing_hz,
        rtol=0.0,
        atol=1.0e-3,
    )
    split_points = np.flatnonzero(~consecutive) + 1
    starts = np.concatenate((np.asarray([0]), split_points))
    stops = np.concatenate((split_points, np.asarray([frequencies_hz.size])))
    lengths = stops - starts
    longest = int(np.argmax(lengths))
    selected = slice(int(starts[longest]), int(stops[longest]))
    return frequencies_hz[selected], response[selected]


def _fit_multipath_model(
    frequencies_hz: NDArray[np.float64],
    response: NDArray[np.complex128],
    *,
    grid_spacing_hz: float,
    maximum_delay_s: float,
    max_model_order: int,
    minimum_contiguous_tones: int,
    candidate_amplitude_floor_ratio: float,
) -> _MultipathModel | None:
    """Fit a BIC-selected exponential model without using range truth.

    A shifted Hankel matrix pencil is built only from the longest truly uniform
    tone run.  Its delay roots initialize a complex least-squares reconstruction
    over *all* supplied tones, including tones outside that run.  Amplitudes are normalized by the
    largest input magnitude; their absolute scale is therefore diagnostic only.
    """

    scale = float(np.max(np.abs(response)))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    normalized_response = np.asarray(response / scale, dtype=np.complex128)
    _, block_response = _longest_uniform_tone_run(
        frequencies_hz,
        normalized_response,
        grid_spacing_hz,
    )
    block_count = int(block_response.size)
    if block_count < minimum_contiguous_tones:
        return None

    window_length = block_count // 2
    if window_length < 3:
        return None
    pencil_left = np.lib.stride_tricks.sliding_window_view(
        block_response[:-1], window_length
    ).T
    pencil_right = np.lib.stride_tricks.sliding_window_view(
        block_response[1:], window_length
    ).T
    try:
        left_vectors, singular_values, right_vectors_h = np.linalg.svd(
            pencil_left, full_matrices=False, compute_uv=True
        )
    except np.linalg.LinAlgError:
        return None

    usable_order = min(
        int(max_model_order),
        int(singular_values.size),
    )
    if usable_order < 1:
        return None
    centered_frequencies = frequencies_hz - float(np.mean(frequencies_hz))
    response_energy = float(np.vdot(normalized_response, normalized_response).real)
    if not np.isfinite(response_energy) or response_energy <= 0.0:
        return None
    ambiguity_period_s = 1.0 / grid_spacing_hz
    delay_tolerance_s = max(1.0e-15, maximum_delay_s * 1.0e-12)
    real_observation_count = 2 * block_count
    candidates: list[
        tuple[
            float,
            int,
            int,
            NDArray[np.float64],
            NDArray[np.float64],
            float,
        ]
    ] = []
    singular_value_floor = (
        np.finfo(float).eps
        * max(pencil_left.shape)
        * float(singular_values[0])
    )

    for trial_order in range(1, usable_order + 1):
        if singular_values[trial_order - 1] <= singular_value_floor:
            break
        try:
            inverse_singular_values = np.diag(
                1.0 / singular_values[:trial_order]
            )
            right_subspace = right_vectors_h.conj().T[:, :trial_order]
            shift_operator = (
                inverse_singular_values
                @ left_vectors[:, :trial_order].conj().T
                @ pencil_right
                @ right_subspace
            )
            roots = np.linalg.eigvals(shift_operator)
        except np.linalg.LinAlgError:
            continue
        delays_s = np.mod(
            -np.angle(roots) / (2.0 * np.pi * grid_spacing_hz),
            ambiguity_period_s,
        )
        valid = np.isfinite(delays_s) & (
            delays_s <= maximum_delay_s + delay_tolerance_s
        )
        delays_s = np.asarray(delays_s[valid], dtype=float)
        if delays_s.size == 0:
            continue
        steering = np.exp(
            -1j
            * 2.0
            * np.pi
            * centered_frequencies[:, None]
            * delays_s[None, :]
        )
        try:
            complex_amplitudes = np.linalg.lstsq(
                steering, normalized_response, rcond=None
            )[0]
        except np.linalg.LinAlgError:
            continue
        residual = normalized_response - steering @ complex_amplitudes
        sum_squared_error = float(np.vdot(residual, residual).real)
        if not np.isfinite(sum_squared_error):
            continue
        effective_order = int(delays_s.size)
        bic = float(
            real_observation_count
            * np.log(
                max(
                    sum_squared_error / real_observation_count,
                    1.0e-15,
                )
            )
            + 3.0
            * effective_order
            * np.log(real_observation_count)
        )
        candidates.append(
            (
                bic,
                trial_order,
                effective_order,
                delays_s,
                np.asarray(np.abs(complex_amplitudes), dtype=float),
                sum_squared_error,
            )
        )

    if not candidates:
        return None
    bic, _, order, delays_s, amplitudes, sum_squared_error = min(
        candidates, key=lambda item: (item[0], item[2])
    )
    maximum_amplitude = float(np.max(amplitudes))
    if not np.isfinite(maximum_amplitude) or maximum_amplitude <= 0.0:
        return None
    significant = amplitudes >= (
        candidate_amplitude_floor_ratio * maximum_amplitude
    )
    if not np.any(significant):
        return None
    significant_indices = np.flatnonzero(significant)
    selected_index = int(
        significant_indices[np.argmin(delays_s[significant_indices])]
    )
    selected_delay_s = float(delays_s[selected_index])
    selected_amplitude_ratio = float(
        amplitudes[selected_index] / maximum_amplitude
    )
    sorting = np.argsort(delays_s, kind="stable")
    significant_delays_s = np.sort(delays_s[significant])
    minimum_significant_separation_s = (
        float(np.min(np.diff(significant_delays_s)))
        if significant_delays_s.size >= 2
        else None
    )
    challenger = next(
        (
            item
            for item in candidates
            if item[1] == 2 and item[2] == 2
        ),
        None,
    )
    challenger_delay_s: float | None = None
    challenger_secondary_amplitude_ratio: float | None = None
    if challenger is not None:
        challenger_delays_s = challenger[3]
        challenger_amplitudes = challenger[4]
        challenger_maximum_amplitude = float(np.max(challenger_amplitudes))
        challenger_significant = challenger_amplitudes >= (
            candidate_amplitude_floor_ratio * challenger_maximum_amplitude
        )
        challenger_indices = np.flatnonzero(challenger_significant)
        if challenger_indices.size:
            challenger_index = int(
                challenger_indices[
                    np.argmin(challenger_delays_s[challenger_indices])
                ]
            )
            challenger_delay_s = float(challenger_delays_s[challenger_index])
            challenger_secondary_amplitude_ratio = float(
                np.min(challenger_amplitudes)
                / challenger_maximum_amplitude
            )
    noise_index = min(5, singular_values.size - 1)
    subspace_noise_ratio = float(
        singular_values[noise_index] / singular_values[0]
    )
    reconstruction_residual = float(
        np.sqrt(max(sum_squared_error, 0.0) / response_energy)
    )
    return _MultipathModel(
        order=order,
        delays_s=tuple(float(value) for value in delays_s[sorting]),
        amplitudes=tuple(float(value) for value in amplitudes[sorting]),
        reconstruction_residual=reconstruction_residual,
        selected_delay_s=selected_delay_s,
        selected_amplitude_ratio=selected_amplitude_ratio,
        bic=float(bic),
        minimum_significant_separation_s=minimum_significant_separation_s,
        subspace_noise_ratio=subspace_noise_ratio,
        challenger_delay_s=challenger_delay_s,
        challenger_secondary_amplitude_ratio=(
            challenger_secondary_amplitude_ratio
        ),
    )


def _local_peak_indices(scores: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return strict-or-shouldered local maxima, including search endpoints."""

    if scores.size == 1:
        return np.asarray([0], dtype=np.int64)
    peaks: list[int] = []
    if scores[0] > scores[1]:
        peaks.append(0)
    if scores.size > 2:
        middle = np.flatnonzero(
            (scores[1:-1] >= scores[:-2])
            & (scores[1:-1] >= scores[2:])
            & (
                (scores[1:-1] > scores[:-2])
                | (scores[1:-1] > scores[2:])
            )
        )
        peaks.extend((middle + 1).tolist())
    if scores[-1] > scores[-2]:
        peaks.append(scores.size - 1)
    return np.asarray(peaks, dtype=np.int64)


def _delay_peak_profile(
    frequencies_hz: NDArray[np.float64],
    response: NDArray[np.complex128],
    weights: NDArray[np.float64],
    *,
    maximum_delay_s: float,
    oversampling: int,
    independent_peak_separation_widths: float,
) -> _DelayPeakProfile:
    """Evaluate the coherent delay spectrum and compare distinct local peaks."""

    bandwidth_hz = float(np.ptp(frequencies_hz))
    delay_step_s = 1.0 / (bandwidth_hz * oversampling)
    delay_count = max(3, int(np.ceil(maximum_delay_s / delay_step_s)) + 1)
    candidate_delays = np.linspace(0.0, maximum_delay_s, delay_count)
    center_frequency_hz = float(np.average(frequencies_hz, weights=weights))
    centered_frequencies = frequencies_hz - center_frequency_hz
    unit_phase = response / np.maximum(
        np.abs(response), np.finfo(float).tiny
    )
    weighted_phase = weights * unit_phase
    normalization = float(np.sum(weights))

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
        scores[start:stop] = np.abs(steering @ weighted_phase) / normalization

    primary_index = int(np.argmax(scores))
    primary_delay_s = float(candidate_delays[primary_index])
    local_peaks = _local_peak_indices(scores)
    minimum_separation_s = (
        independent_peak_separation_widths / bandwidth_hz
    )
    independent = local_peaks[
        np.abs(candidate_delays[local_peaks] - primary_delay_s)
        >= minimum_separation_s
    ]
    if independent.size:
        second_index = int(independent[np.argmax(scores[independent])])
        secondary_score = float(scores[second_index])
        secondary_delay_s: float | None = float(candidate_delays[second_index])
    else:
        secondary_score = 0.0
        secondary_delay_s = None
    primary_score = float(scores[primary_index])
    peak_ratio = (
        primary_score / secondary_score
        if secondary_score > 0.0
        else float("inf")
    )
    return _DelayPeakProfile(
        primary_delay_s=primary_delay_s,
        secondary_delay_s=secondary_delay_s,
        primary_score=primary_score,
        secondary_score=secondary_score,
        peak_ratio=float(peak_ratio),
        independent_peak_count=int(independent.size),
    )


def _quality_confidence(
    peak_ratio: float,
    residual_rms_rad: float,
    maximum_residual_rms_rad: float,
) -> float:
    peak_quality = (
        1.0
        if np.isinf(peak_ratio)
        else float(np.clip(1.0 - 1.0 / max(peak_ratio, 1.0), 0.0, 1.0))
    )
    residual_quality = float(
        np.exp(-((residual_rms_rad / maximum_residual_rms_rad) ** 2))
    )
    return float(np.sqrt(peak_quality * residual_quality))


def estimate_distance_robust(
    frequencies_hz: ArrayLike,
    measured_response: ArrayLike,
    *,
    config: RobustPBRConfig | None = None,
) -> RobustPBRResult:
    """Estimate BLE PBR distance and reject measurements lacking evidence.

    Unsorted tone/response pairs are stably sorted by nominal frequency by
    default.  Any CFO/FFO correction that depends on acquisition timestamps
    must be applied *before* calling this function.
    """

    cfg = config or RobustPBRConfig()
    try:
        frequencies = np.asarray(frequencies_hz, dtype=float)
        response = np.asarray(measured_response, dtype=np.complex128)
    except (TypeError, ValueError):
        return _empty_result(status="invalid_input", reason="array_conversion_failed")

    if frequencies.ndim != 1 or response.ndim != 1:
        return _empty_result(status="invalid_input", reason="arrays_must_be_1d")
    if frequencies.shape != response.shape:
        return _empty_result(
            status="invalid_input",
            reason="frequency_response_shape_mismatch",
            tone_count=int(min(frequencies.size, response.size)),
        )
    tone_count = int(frequencies.size)
    if tone_count < cfg.minimum_tone_count:
        return _empty_result(
            status="invalid_input",
            reason="insufficient_tone_count",
            tone_count=tone_count,
        )
    if not np.all(np.isfinite(frequencies)) or not np.all(
        np.isfinite(response.real) & np.isfinite(response.imag)
    ):
        return _empty_result(
            status="invalid_input",
            reason="nonfinite_input",
            tone_count=tone_count,
        )
    if np.any(frequencies <= 0.0):
        return _empty_result(
            status="invalid_input",
            reason="nonpositive_frequency",
            tone_count=tone_count,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        response_magnitudes = np.abs(response)
    if not np.all(np.isfinite(response_magnitudes)):
        return _empty_result(
            status="invalid_input",
            reason="response_magnitude_overflow",
            tone_count=tone_count,
        )
    if float(np.max(response_magnitudes)) <= 0.0:
        return _empty_result(
            status="invalid_input",
            reason="all_zero_response",
            tone_count=tone_count,
        )

    input_was_sorted = bool(np.all(np.diff(frequencies) > 0.0))
    if not input_was_sorted:
        if not cfg.sort_frequencies:
            return _empty_result(
                status="invalid_input",
                reason="frequencies_not_strictly_increasing",
                tone_count=tone_count,
                input_was_sorted=False,
            )
        order = np.argsort(frequencies, kind="stable")
        frequencies = frequencies[order]
        response = response[order]
    if np.any(np.diff(frequencies) == 0.0):
        return _empty_result(
            status="invalid_input",
            reason="duplicate_frequencies",
            tone_count=tone_count,
            input_was_sorted=input_was_sorted,
        )

    grid_spacing_hz = _frequency_grid_spacing_hz(frequencies)
    if grid_spacing_hz is None and cfg.require_identifiable_frequency_grid:
        return _empty_result(
            status="rejected",
            reason="frequency_grid_not_identifiable",
            tone_count=tone_count,
            input_was_sorted=input_was_sorted,
        )
    unambiguous_distance_m = (
        C0 / (cfg.path_factor * grid_spacing_hz)
        if grid_spacing_hz is not None
        else None
    )
    if (
        unambiguous_distance_m is not None
        and cfg.max_distance_m >= unambiguous_distance_m
    ):
        return _empty_result(
            status="rejected",
            reason="search_prior_exceeds_unambiguous_range",
            tone_count=tone_count,
            input_was_sorted=input_was_sorted,
            grid_spacing_hz=grid_spacing_hz,
            unambiguous_distance_m=unambiguous_distance_m,
        )

    weights = _phase_variance_weights(response, cfg.magnitude_weighting)
    maximum_delay_s = cfg.path_factor * cfg.max_distance_m / C0
    profile = _delay_peak_profile(
        frequencies,
        response,
        weights,
        maximum_delay_s=maximum_delay_s,
        oversampling=cfg.coarse_oversampling,
        independent_peak_separation_widths=(
            cfg.independent_peak_separation_widths
        ),
    )
    try:
        estimate = estimate_distance(
            frequencies,
            response,
            path_factor=cfg.path_factor,
            magnitude_weighting=cfg.magnitude_weighting,
            estimator="ambiguity_aware",
            max_distance_m=cfg.max_distance_m,
            coarse_oversampling=cfg.coarse_oversampling,
        )
    except ValueError as exc:
        reason = (
            "estimated_distance_outside_search_prior"
            if "outside the configured search prior" in str(exc)
            else "numeric_estimation_failed"
        )
        return _empty_result(
            status="rejected",
            reason=reason,
            tone_count=tone_count,
            input_was_sorted=input_was_sorted,
            grid_spacing_hz=grid_spacing_hz,
            unambiguous_distance_m=unambiguous_distance_m,
            profile=profile,
        )

    multipath_model = (
        _fit_multipath_model(
            frequencies,
            response,
            grid_spacing_hz=grid_spacing_hz,
            maximum_delay_s=maximum_delay_s,
            max_model_order=cfg.multipath_max_model_order,
            minimum_contiguous_tones=(
                cfg.multipath_minimum_contiguous_tones
            ),
            candidate_amplitude_floor_ratio=(
                cfg.multipath_candidate_amplitude_floor_ratio
            ),
        )
        if cfg.multipath_model_enabled and grid_spacing_hz is not None
        else None
    )
    selected_delay_s = float(estimate.delay_s)
    selected_distance_m = float(estimate.distance_m)
    estimator_mode = "single_path_phase_fit"
    if multipath_model is None:
        estimator_mode = "single_path_phase_fit_fallback"
    elif multipath_model.order >= 2:
        selected_delay_s = multipath_model.selected_delay_s
        selected_distance_m = C0 * selected_delay_s / cfg.path_factor
        estimator_mode = "multipath_matrix_pencil"
    challenger_disagreement_m = (
        abs(
            C0 * multipath_model.challenger_delay_s / cfg.path_factor
            - estimate.distance_m
        )
        if multipath_model is not None
        and multipath_model.challenger_delay_s is not None
        else None
    )

    path_competition_ratio: float | None = None
    if multipath_model is not None and multipath_model.order >= 2:
        descending_amplitudes = np.sort(
            np.asarray(multipath_model.amplitudes, dtype=float)
        )[::-1]
        if descending_amplitudes.size >= 2 and descending_amplitudes[0] > 0.0:
            path_competition_ratio = float(
                descending_amplitudes[1] / descending_amplitudes[0]
            )
    resolution_width_s = 1.0 / float(np.ptp(frequencies))
    strong_multipath_evidence = bool(
        multipath_model is not None
        and multipath_model.order >= 2
        and multipath_model.reconstruction_residual
        <= cfg.multipath_strong_maximum_reconstruction_residual
        and multipath_model.selected_amplitude_ratio
        >= cfg.multipath_minimum_early_path_amplitude_ratio
        and multipath_model.minimum_significant_separation_s is not None
        and multipath_model.minimum_significant_separation_s
        >= cfg.multipath_unresolved_separation_widths * resolution_width_s
        and path_competition_ratio is not None
    )

    confidence = _quality_confidence(
        profile.peak_ratio,
        estimate.residual_rms_rad,
        cfg.maximum_residual_rms_rad,
    )
    if strong_multipath_evidence and multipath_model is not None:
        residual_margin = float(
            np.clip(
                1.0
                - multipath_model.reconstruction_residual
                / cfg.multipath_strong_maximum_reconstruction_residual,
                0.0,
                1.0,
            )
        )
        confidence = float(
            np.clip(
                0.4 * (0.5 + 0.5 * residual_margin)
                + 0.3 * multipath_model.selected_amplitude_ratio
                + 0.3 * (1.0 - float(path_competition_ratio)),
                0.0,
                1.0,
            )
        )
    quality_flags: list[str] = []
    model_rejection_reason: str | None = None
    if not strong_multipath_evidence:
        if profile.peak_ratio < cfg.minimum_peak_ratio:
            quality_flags.append("ambiguous_delay_peaks")
        if estimate.residual_rms_rad > cfg.maximum_residual_rms_rad:
            quality_flags.append("phase_residual_too_high")
    distance_tolerance_m = max(1.0e-9, cfg.max_distance_m * 1.0e-12)
    if not -distance_tolerance_m <= selected_distance_m <= (
        cfg.max_distance_m + distance_tolerance_m
    ):
        quality_flags.append("estimated_distance_outside_search_prior")
    if multipath_model is not None and multipath_model.order >= 2:
        if (
            multipath_model.reconstruction_residual
            > cfg.multipath_maximum_reconstruction_residual
        ):
            quality_flags.append("multipath_reconstruction_residual_too_high")
            model_rejection_reason = "multipath_reconstruction_residual_too_high"
        if (
            multipath_model.selected_amplitude_ratio
            < cfg.multipath_minimum_early_path_amplitude_ratio
        ):
            quality_flags.append("multipath_early_path_too_weak")
            if model_rejection_reason is None:
                model_rejection_reason = "multipath_early_path_too_weak"
        if (
            multipath_model.minimum_significant_separation_s is not None
            and multipath_model.minimum_significant_separation_s
            < cfg.multipath_unresolved_separation_widths * resolution_width_s
            and abs(selected_distance_m - estimate.distance_m)
            >= cfg.multipath_unresolved_minimum_legacy_disagreement_m
        ):
            quality_flags.append("unresolved_multipath_components")
            if model_rejection_reason is None:
                model_rejection_reason = "unresolved_multipath_components"
        if (
            strong_multipath_evidence
            and path_competition_ratio is not None
            and path_competition_ratio
            >= cfg.multipath_maximum_path_competition_ratio
        ):
            # Keep the historical primary reason stable for pipeline/UI
            # consumers while exposing the model-specific diagnosis as well.
            quality_flags.append("ambiguous_delay_peaks")
            quality_flags.append("multipath_path_competition_too_high")
            if model_rejection_reason is None:
                model_rejection_reason = "multipath_path_competition_too_high"
    elif (
        multipath_model is not None
        and multipath_model.order == 1
        and challenger_disagreement_m is not None
        and multipath_model.challenger_secondary_amplitude_ratio is not None
        and multipath_model.subspace_noise_ratio
        <= cfg.multipath_challenger_maximum_noise_subspace_ratio
        and multipath_model.challenger_secondary_amplitude_ratio
        >= cfg.multipath_challenger_minimum_secondary_amplitude_ratio
        and challenger_disagreement_m
        >= cfg.multipath_challenger_minimum_legacy_disagreement_m
    ):
        quality_flags.append("sparse_multipath_challenger_disagrees")
        model_rejection_reason = "sparse_multipath_challenger_disagrees"
    # The public confidence floor is authoritative for every estimator mode.
    # Strong multipath evidence selects the multipath-aware candidate; it must
    # never silently bypass a caller's requested rejection threshold.
    if confidence < cfg.minimum_confidence:
        quality_flags.append("confidence_below_threshold")

    availability = not quality_flags
    status: RobustPBRStatus = "available" if availability else "rejected"
    rejection_reason = quality_flags[0] if quality_flags else None
    return RobustPBRResult(
        availability=availability,
        status=status,
        rejection_reason=rejection_reason,
        confidence=confidence,
        distance_m=selected_distance_m,
        delay_s=selected_delay_s,
        residual_rms_rad=float(estimate.residual_rms_rad),
        peak_ratio=float(profile.peak_ratio),
        frequency_grid_spacing_hz=grid_spacing_hz,
        unambiguous_distance_m=unambiguous_distance_m,
        primary_peak_delay_s=profile.primary_delay_s,
        secondary_peak_delay_s=profile.secondary_delay_s,
        primary_peak_score=profile.primary_score,
        secondary_peak_score=profile.secondary_score,
        independent_peak_count=profile.independent_peak_count,
        used_tone_count=tone_count,
        input_was_sorted=input_was_sorted,
        quality_flags=tuple(quality_flags),
        estimate=estimate,
        estimator_mode=estimator_mode,
        model_order=(multipath_model.order if multipath_model is not None else 0),
        path_delays_s=(
            multipath_model.delays_s if multipath_model is not None else ()
        ),
        path_amplitudes=(
            multipath_model.amplitudes if multipath_model is not None else ()
        ),
        reconstruction_residual=(
            multipath_model.reconstruction_residual
            if multipath_model is not None
            else None
        ),
        selected_path_amplitude_ratio=(
            multipath_model.selected_amplitude_ratio
            if multipath_model is not None
            else None
        ),
        legacy_disagreement_m=float(
            abs(selected_distance_m - estimate.distance_m)
        ),
        minimum_path_separation_s=(
            multipath_model.minimum_significant_separation_s
            if multipath_model is not None
            else None
        ),
        subspace_noise_ratio=(
            multipath_model.subspace_noise_ratio
            if multipath_model is not None
            else None
        ),
        challenger_delay_s=(
            multipath_model.challenger_delay_s
            if multipath_model is not None
            else None
        ),
        challenger_secondary_amplitude_ratio=(
            multipath_model.challenger_secondary_amplitude_ratio
            if multipath_model is not None
            else None
        ),
        challenger_disagreement_m=challenger_disagreement_m,
        model_bic=(multipath_model.bic if multipath_model is not None else None),
        path_competition_ratio=path_competition_ratio,
        strong_multipath_evidence=strong_multipath_evidence,
        model_rejection_reason=model_rejection_reason,
    )


__all__ = [
    "RobustPBRConfig",
    "RobustPBRResult",
    "RobustPBRStatus",
    "estimate_distance_robust",
]
