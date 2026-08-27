"""CFAR-assisted UWB first-path detection with explicit availability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


class FirstPathStatus(str, Enum):
    """Quality state of a first-path estimate."""

    PASS = "PASS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_PATH = "NO_PATH"


@dataclass(frozen=True, slots=True)
class FirstPathResult:
    delay_s: float
    sample_index: int
    threshold: float
    peak_magnitude: float
    confidence: float
    crossing_delay_s: float
    crossing_index: int
    fractional_sample_index: float
    method: str
    status: FirstPathStatus = FirstPathStatus.PASS
    rejection_reason: str | None = None
    noise_floor: float = 0.0
    noise_rms: float = 0.0
    cfar_threshold: float = 0.0
    peak_snr_db: float = math.inf
    lobe_sample_count: int = 0
    peak_to_global_ratio: float = 1.0
    ambiguity_ratio: float = 0.0
    early_path_enhanced: bool = False
    early_path_evidence_snr_db: float = -300.0
    clean_residual_ratio: float = 1.0
    clean_frequency_sample_count: int = 0

    @property
    def available(self) -> bool:
        """Whether this result contains a usable (possibly low-quality) delay."""

        return self.status is not FirstPathStatus.NO_PATH and math.isfinite(
            self.delay_s
        )


def _parabolic_peak_offset(magnitude: np.ndarray, index: int) -> float:
    """Return a bounded sub-sample offset from three magnitude samples."""

    if index <= 0 or index >= magnitude.size - 1:
        return 0.0
    left = float(magnitude[index - 1])
    center = float(magnitude[index])
    right = float(magnitude[index + 1])
    denominator = left - 2.0 * center + right
    if denominator >= -np.finfo(float).eps * max(1.0, center):
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(np.clip(offset, -0.5, 0.5))


def _noise_statistics(magnitude: np.ndarray) -> tuple[float, float, float]:
    """Estimate a complex-noise RMS level and a robust amplitude spread.

    Squared magnitudes of circular complex Gaussian noise are exponential. A
    lower-quartile estimate is less easily contaminated by sparse CIR lobes
    than a mean, while retaining the corresponding exponential correction.
    """

    power = np.square(magnitude)
    power_quartile = float(np.quantile(power, 0.25))
    exponential_quartile = -math.log(0.75)
    noise_power = max(0.0, power_quartile / exponential_quartile)
    noise_rms = math.sqrt(noise_power)

    background_limit = float(np.quantile(magnitude, 0.60))
    background = magnitude[magnitude <= background_limit]
    noise_floor = float(np.median(background))
    noise_mad = float(np.median(np.abs(background - noise_floor)))
    return noise_floor, noise_rms, noise_mad


def _global_cfar_threshold(
    noise_rms: float,
    cell_count: int,
    false_alarm_probability: float,
) -> float:
    """Return a Rayleigh-amplitude threshold for a whole search interval."""

    if noise_rms <= 0.0:
        return 0.0
    cell_probability = -math.expm1(
        math.log1p(-false_alarm_probability) / cell_count
    )
    return noise_rms * math.sqrt(-math.log(cell_probability))


def _contiguous_runs(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    return [run for run in np.split(indices, split_points) if run.size]


@dataclass(frozen=True, slots=True)
class _EarlyPathEvidence:
    sample_index: int
    fractional_sample_index: float
    crossing_index: int
    lobe_sample_count: int
    peak_magnitude: float
    local_noise_rms: float
    peak_snr_db: float
    peak_to_dominant_ratio: float
    clean_residual_ratio: float
    frequency_sample_count: int
    suggestive: bool
    reliable: bool


def _frequency_window(sample_count: int, window: str | None) -> np.ndarray:
    if window is None:
        return np.ones(sample_count, dtype=float)
    if window == "hann":
        return np.hanning(sample_count)
    if window == "hamming":
        return np.hamming(sample_count)
    raise ValueError("clean_window must be None, 'hann', or 'hamming'")


def _infer_frequency_sample_count(
    cir: np.ndarray,
    window: str | None,
) -> int | None:
    """Infer a zero-padded CFR support without assuming an arbitrary CIR."""

    spectrum = np.fft.fft(cir)
    scale = float(np.max(np.abs(spectrum)))
    if scale <= 0.0:
        return None
    tolerance = max(np.finfo(float).eps * cir.size * scale, 1.0e-10 * scale)
    active = np.flatnonzero(np.abs(spectrum) > tolerance)
    if active.size == 0:
        return None
    # A Hann window has an exact zero at its last sample, whereas a Hamming
    # or rectangular window does not.  The inferred support is accepted only
    # when a material zero-padding tail remains, so arbitrary time-domain CIR
    # inputs cannot accidentally enter the CLEAN path.
    sample_count = int(active[-1] + (2 if window == "hann" else 1))
    if sample_count < 4 or sample_count > cir.size:
        return None
    minimum_tail = max(8, cir.size // 16)
    if cir.size - sample_count < minimum_tail:
        return None
    total_energy = float(np.vdot(spectrum, spectrum).real)
    tail_energy = float(
        np.vdot(spectrum[sample_count:], spectrum[sample_count:]).real
    )
    if total_energy <= 0.0 or tail_energy > 1.0e-12 * total_energy:
        return None
    return sample_count


def _robust_side_noise_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return math.nan
    power_quantile = float(np.quantile(np.square(samples), 0.75))
    return math.sqrt(max(0.0, power_quantile / math.log(4.0)))


def _dominant_path_clean_recheck(
    impulse_response: np.ndarray,
    magnitudes: np.ndarray,
    search_indices: np.ndarray,
    dominant_index: int,
    *,
    global_noise_rms: float,
    min_consecutive_samples: int,
    frequency_sample_count: int | None,
    window: str | None,
    suggestive_snr_db: float,
    reliable_snr_db: float,
) -> _EarlyPathEvidence | None:
    """Subtract a dominant late lobe and assess earlier residual evidence.

    This is a one-component frequency-domain CLEAN step.  It is enabled only
    for CIRs whose FFT contains a verifiable zero-padded CFR support.  The
    local threshold combines ordered-statistic and smaller-of-cell-averages
    noise estimates, and the candidate must have samples on the rising side
    of its peak.  Suggestive evidence is used only to lower confidence; it is
    not promoted to a range unless it clears the stricter reliability gate.
    """

    sample_count = frequency_sample_count
    if sample_count is None:
        sample_count = _infer_frequency_sample_count(impulse_response, window)
    if sample_count is None:
        return None
    if sample_count < 4 or sample_count > impulse_response.size:
        return None

    weights = _frequency_window(sample_count, window)
    coherent_gain = float(np.mean(weights))
    if coherent_gain <= 0.0:
        return None
    dominant_fractional_index = float(
        dominant_index + _parabolic_peak_offset(magnitudes, dominant_index)
    )
    frequency_indices = np.arange(sample_count, dtype=float)
    template = (
        weights
        / coherent_gain
        * np.exp(
            -2j
            * np.pi
            * frequency_indices
            * dominant_fractional_index
            / impulse_response.size
        )
    )
    denominator = float(np.vdot(template, template).real)
    if denominator <= 0.0:
        return None
    spectrum = np.fft.fft(impulse_response)
    complex_gain = np.vdot(template, spectrum[:sample_count]) / denominator
    residual_spectrum = spectrum.copy()
    residual_spectrum[:sample_count] -= complex_gain * template
    residual = np.fft.ifft(residual_spectrum)
    residual_magnitude = np.abs(residual)

    zero_pad_factor = impulse_response.size / sample_count
    main_lobe_guard = max(
        min_consecutive_samples,
        int(math.ceil(2.0 * zero_pad_factor)),
    )
    prefix = search_indices[
        search_indices <= dominant_index - main_lobe_guard
    ]
    if prefix.size < 2 * min_consecutive_samples + 1:
        return None
    interior = prefix[(prefix > search_indices[0]) & (prefix < prefix[-1])]
    if interior.size == 0:
        return None
    candidate_index = int(interior[np.argmax(residual_magnitude[interior])])
    candidate_peak = float(residual_magnitude[candidate_index])

    training_count = max(24, 2 * main_lobe_guard)
    left_start = max(int(search_indices[0]), candidate_index - main_lobe_guard - training_count)
    left_stop = max(left_start, candidate_index - main_lobe_guard)
    right_start = min(
        int(search_indices[-1]) + 1,
        candidate_index + main_lobe_guard + 1,
    )
    right_stop = min(
        int(search_indices[-1]) + 1,
        right_start + training_count,
    )
    left_noise = _robust_side_noise_rms(
        residual_magnitude[left_start:left_stop]
    )
    right_noise = _robust_side_noise_rms(
        residual_magnitude[right_start:right_stop]
    )
    side_estimates = np.asarray(
        [value for value in (left_noise, right_noise) if math.isfinite(value)],
        dtype=float,
    )
    training = np.concatenate(
        (
            residual_magnitude[left_start:left_stop],
            residual_magnitude[right_start:right_stop],
        )
    )
    os_noise = _robust_side_noise_rms(training)
    soca_noise = (
        float(np.min(side_estimates))
        if side_estimates.size
        else global_noise_rms
    )
    local_noise_rms = max(
        0.90 * global_noise_rms,
        min(os_noise, soca_noise) if math.isfinite(os_noise) else soca_noise,
    )
    if local_noise_rms <= 0.0:
        peak_snr_db = math.inf if candidate_peak > 0.0 else -math.inf
    else:
        peak_snr_db = 20.0 * math.log10(
            max(candidate_peak, np.finfo(float).tiny) / local_noise_rms
        )

    shape_threshold = max(2.0 * local_noise_rms, 0.45 * candidate_peak)
    crossing_index = candidate_index
    while (
        crossing_index > int(prefix[0])
        and residual_magnitude[crossing_index - 1] >= shape_threshold
    ):
        crossing_index -= 1
    run_end = candidate_index
    while (
        run_end < int(prefix[-1])
        and residual_magnitude[run_end + 1] >= shape_threshold
    ):
        run_end += 1
    lobe_sample_count = run_end - crossing_index + 1
    has_peak_front = (
        candidate_index - crossing_index >= 1
        and lobe_sample_count >= min_consecutive_samples
    )

    dominant_peak = max(
        float(magnitudes[dominant_index]), np.finfo(float).tiny
    )
    peak_to_dominant_ratio = candidate_peak / dominant_peak
    dominant_slice = slice(
        max(0, dominant_index - 2),
        min(residual_magnitude.size, dominant_index + 3),
    )
    clean_residual_ratio = (
        float(np.max(residual_magnitude[dominant_slice])) / dominant_peak
    )
    plausible_relative_level = 0.03 <= peak_to_dominant_ratio <= 0.35
    clean_is_stable = clean_residual_ratio <= 0.35
    suggestive = (
        has_peak_front
        and plausible_relative_level
        and clean_is_stable
        and peak_snr_db >= suggestive_snr_db
    )
    reliable = suggestive and peak_snr_db >= reliable_snr_db
    fractional_index = float(
        candidate_index
        + _parabolic_peak_offset(residual_magnitude, candidate_index)
    )
    return _EarlyPathEvidence(
        sample_index=candidate_index,
        fractional_sample_index=fractional_index,
        crossing_index=crossing_index,
        lobe_sample_count=lobe_sample_count,
        peak_magnitude=candidate_peak,
        local_noise_rms=local_noise_rms,
        peak_snr_db=peak_snr_db,
        peak_to_dominant_ratio=peak_to_dominant_ratio,
        clean_residual_ratio=clean_residual_ratio,
        frequency_sample_count=sample_count,
        suggestive=suggestive,
        reliable=reliable,
    )


def _no_path_result(
    *,
    method: str,
    reason: str,
    threshold: float,
    peak: float,
    noise_floor: float,
    noise_rms: float,
    cfar_threshold: float,
) -> FirstPathResult:
    return FirstPathResult(
        delay_s=math.nan,
        sample_index=-1,
        threshold=threshold,
        peak_magnitude=peak,
        confidence=0.0,
        crossing_delay_s=math.nan,
        crossing_index=-1,
        fractional_sample_index=math.nan,
        method=method,
        status=FirstPathStatus.NO_PATH,
        rejection_reason=reason,
        noise_floor=noise_floor,
        noise_rms=noise_rms,
        cfar_threshold=cfar_threshold,
        peak_snr_db=-math.inf,
        lobe_sample_count=0,
        peak_to_global_ratio=0.0,
        ambiguity_ratio=0.0,
    )


def detect_first_path(
    delay_s: ArrayLike,
    cir: ArrayLike,
    *,
    threshold_ratio: float = 0.18,
    search_start_s: float = 0.0,
    search_end_s: float | None = None,
    method: Literal["first_lobe_peak", "threshold_crossing"] = "first_lobe_peak",
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
) -> FirstPathResult:
    """Detect the first significant CIR lobe and estimate its arrival time.

    The default method combines a robust global CFAR threshold with a minimum
    contiguous-run requirement. It accepts a first lobe down to five percent
    of a later global peak, but marks candidates below ``threshold_ratio`` as
    low confidence. ``threshold_crossing`` retains the legacy 18-percent
    candidate ratio unless ``candidate_threshold_ratio`` is supplied.

    When a high-SNR dominant lobe is selected, the detector can perform one
    conservative frequency-domain CLEAN recheck for earlier evidence.  A
    merely suggestive residual never replaces the selected delay: it only
    downgrades the status.  Repeated CFR measurements should normally be
    phase-aligned and coherently combined before this detector is called.

    A non-detection returns a ``NO_PATH`` result containing NaN delay evidence;
    malformed arrays and invalid configuration still raise ``ValueError``.
    """

    delays = np.asarray(delay_s, dtype=float)
    impulse_response = np.asarray(cir, dtype=np.complex128)
    if delays.ndim != 1 or impulse_response.ndim != 1 or delays.shape != impulse_response.shape:
        raise ValueError("delay_s and cir must be matching 1-D arrays")
    if delays.size < 3:
        raise ValueError("CIR must contain at least three samples")
    if not np.all(np.isfinite(delays)) or not np.all(
        np.isfinite(impulse_response.real)
        & np.isfinite(impulse_response.imag)
    ):
        raise ValueError("delay_s and cir must contain only finite values")
    if np.any(np.diff(delays) <= 0.0):
        raise ValueError("delay_s must be strictly increasing")
    delay_steps = np.diff(delays)
    sample_spacing_s = float(np.median(delay_steps))
    if not np.allclose(
        delay_steps,
        sample_spacing_s,
        rtol=1.0e-6,
        atol=max(np.finfo(float).eps, abs(sample_spacing_s) * 1.0e-9),
    ):
        raise ValueError("delay_s must be uniformly sampled")
    if not 0.0 < threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must lie in (0, 1]")
    if noise_threshold_sigma < 0.0:
        raise ValueError("noise_threshold_sigma must be non-negative")
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must lie in (0, 1)")
    if isinstance(min_consecutive_samples, bool) or min_consecutive_samples < 1:
        raise ValueError("min_consecutive_samples must be a positive integer")
    if int(min_consecutive_samples) != min_consecutive_samples:
        raise ValueError("min_consecutive_samples must be a positive integer")
    min_consecutive_samples = int(min_consecutive_samples)
    if candidate_threshold_ratio is not None and not (
        0.0 < candidate_threshold_ratio <= 1.0
    ):
        raise ValueError("candidate_threshold_ratio must lie in (0, 1]")
    if not math.isfinite(minimum_pass_snr_db):
        raise ValueError("minimum_pass_snr_db must be finite")
    if not 0.0 < ambiguity_peak_ratio <= 1.0:
        raise ValueError("ambiguity_peak_ratio must lie in (0, 1]")
    if not isinstance(early_path_enhancement, bool):
        raise ValueError("early_path_enhancement must be a boolean")
    if clean_frequency_sample_count is not None:
        if (
            isinstance(clean_frequency_sample_count, bool)
            or clean_frequency_sample_count < 4
            or int(clean_frequency_sample_count) != clean_frequency_sample_count
        ):
            raise ValueError(
                "clean_frequency_sample_count must be an integer of at least four"
            )
        clean_frequency_sample_count = int(clean_frequency_sample_count)
    if clean_window not in {None, "hann", "hamming"}:
        raise ValueError("clean_window must be None, 'hann', or 'hamming'")
    if not math.isfinite(minimum_clean_dominant_snr_db):
        raise ValueError("minimum_clean_dominant_snr_db must be finite")
    if not math.isfinite(early_path_suggestive_snr_db):
        raise ValueError("early_path_suggestive_snr_db must be finite")
    if not math.isfinite(early_path_reliable_snr_db):
        raise ValueError("early_path_reliable_snr_db must be finite")
    if early_path_reliable_snr_db < early_path_suggestive_snr_db:
        raise ValueError(
            "early_path_reliable_snr_db must not be below the suggestive gate"
        )
    if search_end_s is not None and search_end_s <= search_start_s:
        raise ValueError("search_end_s must be greater than search_start_s")
    if method not in {"first_lobe_peak", "threshold_crossing"}:
        raise ValueError(
            "method must be 'first_lobe_peak' or 'threshold_crossing'"
        )

    mask = delays >= search_start_s
    if search_end_s is not None:
        mask &= delays <= search_end_s
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("the requested search interval contains no CIR samples")

    magnitudes = np.abs(impulse_response)
    search_magnitudes = magnitudes[indices]
    peak = float(np.max(search_magnitudes))
    noise_floor, noise_rms, noise_mad = _noise_statistics(search_magnitudes)
    cfar_threshold = _global_cfar_threshold(
        noise_rms, indices.size, false_alarm_probability
    )
    mad_threshold = (
        noise_floor + noise_threshold_sigma * 1.4826 * noise_mad
    )

    if peak <= 0.0:
        return _no_path_result(
            method=method,
            reason="ZERO_ENERGY",
            threshold=max(cfar_threshold, mad_threshold),
            peak=peak,
            noise_floor=noise_floor,
            noise_rms=noise_rms,
            cfar_threshold=cfar_threshold,
        )

    if candidate_threshold_ratio is None:
        candidate_ratio = (
            threshold_ratio
            if method == "threshold_crossing"
            else min(0.05, threshold_ratio)
        )
    else:
        candidate_ratio = min(candidate_threshold_ratio, threshold_ratio)
    threshold = max(
        candidate_ratio * peak,
        cfar_threshold,
        mad_threshold,
    )
    crossings = indices[magnitudes[indices] >= threshold]
    if crossings.size == 0:
        return _no_path_result(
            method=method,
            reason="PEAK_BELOW_CFAR_THRESHOLD",
            threshold=threshold,
            peak=peak,
            noise_floor=noise_floor,
            noise_rms=noise_rms,
            cfar_threshold=cfar_threshold,
        )

    all_runs = _contiguous_runs(crossings)
    runs = [
        run for run in all_runs if run.size >= min_consecutive_samples
    ]
    if not runs:
        return _no_path_result(
            method=method,
            reason="NO_RUN_MEETS_MIN_CONSECUTIVE_SAMPLES",
            threshold=threshold,
            peak=peak,
            noise_floor=noise_floor,
            noise_rms=noise_rms,
            cfar_threshold=cfar_threshold,
        )

    first_run = runs[0]
    first_index = int(first_run[0])
    lobe_peak_index = int(first_run[np.argmax(magnitudes[first_run])])
    selected_index = first_index
    fractional_offset = 0.0
    if method == "first_lobe_peak":
        selected_index = lobe_peak_index
        if (
            selected_index - 1 >= indices[0]
            and selected_index + 1 <= indices[-1]
        ):
            fractional_offset = _parabolic_peak_offset(
                magnitudes, selected_index
            )

    estimated_delay_s = float(
        delays[selected_index] + fractional_offset * sample_spacing_s
    )
    lobe_peak = float(magnitudes[lobe_peak_index])
    peak_to_global_ratio = lobe_peak / peak
    if noise_rms > 0.0:
        peak_snr_db = 20.0 * math.log10(max(lobe_peak, np.finfo(float).tiny) / noise_rms)
    else:
        peak_snr_db = math.inf

    local_peak_values: list[float] = []
    for index in first_run:
        integer_index = int(index)
        if integer_index <= 0 or integer_index >= magnitudes.size - 1:
            continue
        value = float(magnitudes[integer_index])
        if (
            value >= ambiguity_peak_ratio * lobe_peak
            and value >= magnitudes[integer_index - 1]
            and value > magnitudes[integer_index + 1]
        ):
            local_peak_values.append(value)
    local_peak_values.sort(reverse=True)
    ambiguity_ratio = (
        local_peak_values[1] / lobe_peak
        if len(local_peak_values) > 1
        else 0.0
    )
    for other_run in runs[1:]:
        other_peak = float(np.max(magnitudes[other_run]))
        ambiguity_ratio = max(ambiguity_ratio, other_peak / lobe_peak)

    reported_crossing_index = first_index
    reported_run_start = int(first_run[0])
    reported_run_end = int(first_run[-1])
    selected_lobe_sample_count = int(first_run.size)
    early_path_enhanced = False
    early_path_suggestive = False
    early_path_evidence_snr_db = -300.0
    clean_residual_ratio = 1.0
    clean_sample_count = 0
    dominant_index = int(indices[np.argmax(magnitudes[indices])])
    dominant_snr_db = (
        20.0 * math.log10(max(peak, np.finfo(float).tiny) / noise_rms)
        if noise_rms > 0.0
        else math.inf
    )
    if clean_frequency_sample_count is not None and (
        clean_frequency_sample_count > impulse_response.size
    ):
        raise ValueError(
            "clean_frequency_sample_count exceeds the CIR sample count"
        )
    if (
        early_path_enhancement
        and method == "first_lobe_peak"
        and abs(lobe_peak_index - dominant_index) <= 1
        and dominant_snr_db >= minimum_clean_dominant_snr_db
    ):
        early_evidence = _dominant_path_clean_recheck(
            impulse_response,
            magnitudes,
            indices,
            dominant_index,
            global_noise_rms=noise_rms,
            min_consecutive_samples=min_consecutive_samples,
            frequency_sample_count=clean_frequency_sample_count,
            window=clean_window,
            suggestive_snr_db=early_path_suggestive_snr_db,
            reliable_snr_db=early_path_reliable_snr_db,
        )
        if early_evidence is not None:
            early_path_evidence_snr_db = early_evidence.peak_snr_db
            clean_residual_ratio = early_evidence.clean_residual_ratio
            clean_sample_count = early_evidence.frequency_sample_count
            early_path_suggestive = early_evidence.suggestive
            if (
                early_evidence.reliable
                and early_evidence.sample_index < selected_index
            ):
                early_path_enhanced = True
                selected_index = early_evidence.sample_index
                fractional_offset = (
                    early_evidence.fractional_sample_index - selected_index
                )
                estimated_delay_s = float(
                    delays[selected_index]
                    + fractional_offset * sample_spacing_s
                )
                lobe_peak_index = early_evidence.sample_index
                lobe_peak = early_evidence.peak_magnitude
                peak_to_global_ratio = early_evidence.peak_to_dominant_ratio
                peak_snr_db = early_evidence.peak_snr_db
                reported_crossing_index = early_evidence.crossing_index
                reported_run_start = early_evidence.crossing_index
                reported_run_end = (
                    early_evidence.crossing_index
                    + early_evidence.lobe_sample_count
                    - 1
                )
                selected_lobe_sample_count = early_evidence.lobe_sample_count
                ambiguity_ratio = max(
                    ambiguity_ratio,
                    peak / max(lobe_peak, np.finfo(float).tiny),
                )

    low_confidence_reasons: list[str] = []
    if early_path_enhanced:
        low_confidence_reasons.append(
            "EARLY_PATH_RECOVERED_AFTER_DOMINANT_CLEAN"
        )
    elif early_path_suggestive:
        low_confidence_reasons.append(
            "POSSIBLE_EARLY_PATH_BELOW_RELIABLE_THRESHOLD"
        )
    elif early_path_enhancement:
        # This opt-in mode is deliberately a conservative single-frame guard.
        # Failure to find reliable earlier evidence cannot prove that the
        # dominant lobe is the physical first path, so it must never retain a
        # high-confidence PASS merely because a weak cue fell just below a
        # tuning threshold.
        low_confidence_reasons.append(
            "EARLY_PATH_NOT_CONFIRMED_IN_CONSERVATIVE_MODE"
        )
    if lobe_peak < threshold_ratio * peak:
        low_confidence_reasons.append("FIRST_LOBE_BELOW_PRIMARY_THRESHOLD")
    if peak_snr_db < minimum_pass_snr_db:
        low_confidence_reasons.append("PEAK_SNR_BELOW_PASS_THRESHOLD")
    if reported_run_start == indices[0] or reported_run_end == indices[-1]:
        low_confidence_reasons.append("LOBE_CLIPPED_BY_SEARCH_BOUNDARY")
    if ambiguity_ratio >= ambiguity_peak_ratio:
        low_confidence_reasons.append("COMPETING_OR_OVERLAPPING_LOBE")

    status = (
        FirstPathStatus.LOW_CONFIDENCE
        if low_confidence_reasons
        else FirstPathStatus.PASS
    )
    relative_score = min(
        1.0, peak_to_global_ratio / max(threshold_ratio, np.finfo(float).eps)
    )
    snr_score = (
        1.0
        if math.isinf(peak_snr_db)
        else float(np.clip(peak_snr_db / max(minimum_pass_snr_db, 1.0), 0.0, 1.0))
    )
    confidence = min(relative_score, snr_score)
    if "COMPETING_OR_OVERLAPPING_LOBE" in low_confidence_reasons:
        # A strong competing lobe is direct evidence that the selected peak is
        # not unique.  Keep the numerical candidate available for audit, but
        # prevent the quality score from contradicting LOW_CONFIDENCE.
        confidence *= max(0.05, 1.0 - min(ambiguity_ratio, 1.0))
    if (
        "EARLY_PATH_RECOVERED_AFTER_DOMINANT_CLEAN"
        in low_confidence_reasons
    ):
        confidence *= 0.50
    if (
        "POSSIBLE_EARLY_PATH_BELOW_RELIABLE_THRESHOLD"
        in low_confidence_reasons
    ):
        confidence *= 0.40
    if (
        "EARLY_PATH_NOT_CONFIRMED_IN_CONSERVATIVE_MODE"
        in low_confidence_reasons
    ):
        confidence *= 0.40
    if "LOBE_CLIPPED_BY_SEARCH_BOUNDARY" in low_confidence_reasons:
        confidence *= 0.75
    if low_confidence_reasons:
        confidence = min(confidence, 0.95)

    return FirstPathResult(
        delay_s=estimated_delay_s,
        sample_index=selected_index,
        threshold=threshold,
        peak_magnitude=peak,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        crossing_delay_s=float(delays[reported_crossing_index]),
        crossing_index=reported_crossing_index,
        fractional_sample_index=float(selected_index + fractional_offset),
        method=method,
        status=status,
        rejection_reason=(
            ";".join(low_confidence_reasons)
            if low_confidence_reasons
            else None
        ),
        noise_floor=noise_floor,
        noise_rms=noise_rms,
        cfar_threshold=cfar_threshold,
        peak_snr_db=peak_snr_db,
        lobe_sample_count=selected_lobe_sample_count,
        peak_to_global_ratio=peak_to_global_ratio,
        ambiguity_ratio=ambiguity_ratio,
        early_path_enhanced=early_path_enhanced,
        early_path_evidence_snr_db=early_path_evidence_snr_db,
        clean_residual_ratio=clean_residual_ratio,
        clean_frequency_sample_count=clean_sample_count,
    )
