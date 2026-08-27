"""UWB time-of-flight distance estimation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

from numpy.typing import ArrayLike

from ..constants import C0
from .first_path import FirstPathResult, FirstPathStatus, detect_first_path


def estimate_distance_from_cir(
    delay_s: ArrayLike,
    cir: ArrayLike,
    *,
    path_factor: float = 1.0,
    threshold_ratio: float = 0.18,
    search_start_s: float = 0.0,
    search_end_s: float | None = None,
    detector: Literal[
        "first_lobe_peak", "threshold_crossing"
    ] = "first_lobe_peak",
    calibration_delay_s: float = 0.0,
    noise_threshold_sigma: float = 6.0,
    false_alarm_probability: float = 1.0e-6,
    min_consecutive_samples: int = 3,
    candidate_threshold_ratio: float | None = None,
    minimum_pass_snr_db: float = 12.0,
    ambiguity_peak_ratio: float = 0.80,
    early_path_enhancement: bool = False,
    clean_frequency_sample_count: int | None = None,
    clean_window: Literal["hann", "hamming"] | None = "hann",
    minimum_clean_dominant_snr_db: float = 28.0,
    early_path_suggestive_snr_db: float = 8.0,
    early_path_reliable_snr_db: float = 14.0,
    require_available: bool = False,
) -> tuple[float, FirstPathResult]:
    """Return calibrated distance and raw first-path detection evidence.

    A normal non-detection returns ``(nan, NO_PATH_result)`` so that noise is
    not converted into a forced range. Set ``require_available`` to raise when
    a caller requires a usable PASS or LOW_CONFIDENCE estimate.
    """

    if path_factor <= 0.0:
        raise ValueError("path_factor must be greater than zero")
    if not math.isfinite(calibration_delay_s):
        raise ValueError("calibration_delay_s must be finite")
    result = detect_first_path(
        delay_s,
        cir,
        threshold_ratio=threshold_ratio,
        search_start_s=search_start_s,
        search_end_s=search_end_s,
        method=detector,
        noise_threshold_sigma=noise_threshold_sigma,
        false_alarm_probability=false_alarm_probability,
        min_consecutive_samples=min_consecutive_samples,
        candidate_threshold_ratio=candidate_threshold_ratio,
        minimum_pass_snr_db=minimum_pass_snr_db,
        ambiguity_peak_ratio=ambiguity_peak_ratio,
        early_path_enhancement=early_path_enhancement,
        clean_frequency_sample_count=clean_frequency_sample_count,
        clean_window=clean_window,
        minimum_clean_dominant_snr_db=minimum_clean_dominant_snr_db,
        early_path_suggestive_snr_db=early_path_suggestive_snr_db,
        early_path_reliable_snr_db=early_path_reliable_snr_db,
    )
    if not result.available:
        if require_available:
            raise ValueError(
                "no available first path: "
                f"{result.rejection_reason or 'NO_PATH'}"
            )
        return math.nan, result
    propagation_delay_s = result.delay_s - calibration_delay_s
    if propagation_delay_s < 0.0:
        rejected = replace(
            result,
            status=FirstPathStatus.NO_PATH,
            rejection_reason="CALIBRATION_EXCEEDS_DETECTED_DELAY",
            confidence=0.0,
        )
        if require_available:
            raise ValueError(
                "no available first path: CALIBRATION_EXCEEDS_DETECTED_DELAY"
            )
        return math.nan, rejected
    return C0 * propagation_delay_s / path_factor, result
