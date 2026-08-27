"""Frequency-dependent RF front-end calibration profiles.

The calibration layer is deliberately separate from the impairment generator:
an injected impairment is not a measured calibration.  A profile must be built
from an independently supplied complex transfer response (for example S21),
then explicitly applied to a raw response.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sparameter import load_s2p_s21


MetadataValue: TypeAlias = str | int | float | bool | None
PhaseModel: TypeAlias = Literal["direct", "polynomial"]


def _readonly_float(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_complex(values: ArrayLike) -> NDArray[np.complex128]:
    result = np.array(values, dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


def _validate_frequency_axis(frequencies_hz: NDArray[np.float64]) -> None:
    if frequencies_hz.ndim != 1 or frequencies_hz.size < 3:
        raise ValueError("at least three 1-D frequency samples are required")
    if not np.all(np.isfinite(frequencies_hz)):
        raise ValueError("frequencies_hz must contain only finite values")
    if np.any(np.diff(frequencies_hz) <= 0.0):
        raise ValueError("frequencies_hz must be strictly increasing")


def _validate_complex_vector(
    values: NDArray[np.complex128],
    frequencies_hz: NDArray[np.float64],
    *,
    name: str,
) -> None:
    if values.ndim != 1 or values.shape != frequencies_hz.shape:
        raise ValueError(f"{name} must match the 1-D frequency axis")
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError(f"{name} must contain only finite complex values")


def _normalise_metadata(
    metadata: Mapping[str, MetadataValue] | None,
) -> Mapping[str, MetadataValue]:
    copied: dict[str, MetadataValue] = {}
    for key, value in (metadata or {}).items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata keys must be non-empty strings")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError("metadata values must be JSON scalar values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("floating-point metadata values must be finite")
        copied[key] = value
    return MappingProxyType(copied)


def _profile_digest(
    frequencies_hz: NDArray[np.float64],
    complex_correction: NDArray[np.complex128],
    reference_plane: str,
    source: str,
    metadata: Mapping[str, MetadataValue],
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(frequencies_hz, dtype="<f8").tobytes())
    digest.update(np.asarray(complex_correction, dtype="<c16").tobytes())
    digest.update(reference_plane.encode("utf-8"))
    digest.update(source.encode("utf-8"))
    digest.update(
        json.dumps(
            dict(metadata), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    )
    return digest.hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Immutable complex correction curve and its calibration provenance."""

    frequencies_hz: NDArray[np.float64]
    complex_correction: NDArray[np.complex128]
    measured_transfer: NDArray[np.complex128]
    unwrapped_measured_phase_rad: NDArray[np.float64]
    fitted_phase_rad: NDArray[np.float64]
    group_delay_s: NDArray[np.float64]
    constant_group_delay_s: float
    reference_plane: str
    source: str
    phase_model: PhaseModel
    phase_fit_rms_rad: float
    maximum_interpolation_gap_hz: float
    maximum_correction_magnitude: float
    maximum_expected_group_delay_s: float | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    profile_id: str = field(init=False)
    _correction_phase_rad: NDArray[np.float64] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        frequencies = _readonly_float(self.frequencies_hz)
        correction = _readonly_complex(self.complex_correction)
        measured = _readonly_complex(self.measured_transfer)
        unwrapped_phase = _readonly_float(self.unwrapped_measured_phase_rad)
        fitted_phase = _readonly_float(self.fitted_phase_rad)
        group_delay = _readonly_float(self.group_delay_s)
        _validate_frequency_axis(frequencies)
        _validate_complex_vector(correction, frequencies, name="complex_correction")
        _validate_complex_vector(measured, frequencies, name="measured_transfer")
        for name, values in (
            ("unwrapped_measured_phase_rad", unwrapped_phase),
            ("fitted_phase_rad", fitted_phase),
            ("group_delay_s", group_delay),
        ):
            if values.ndim != 1 or values.shape != frequencies.shape:
                raise ValueError(f"{name} must match the frequency axis")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        if np.any(np.abs(correction) <= 0.0):
            raise ValueError("complex_correction must never be zero")
        if not math.isfinite(self.constant_group_delay_s):
            raise ValueError("constant_group_delay_s must be finite")
        if not math.isfinite(self.phase_fit_rms_rad) or self.phase_fit_rms_rad < 0.0:
            raise ValueError("phase_fit_rms_rad must be finite and non-negative")
        if (
            not math.isfinite(self.maximum_interpolation_gap_hz)
            or self.maximum_interpolation_gap_hz <= 0.0
        ):
            raise ValueError(
                "maximum_interpolation_gap_hz must be finite and positive"
            )
        if (
            not math.isfinite(self.maximum_correction_magnitude)
            or self.maximum_correction_magnitude < 1.0
        ):
            raise ValueError(
                "maximum_correction_magnitude must be finite and at least one"
            )
        if self.maximum_expected_group_delay_s is not None and (
            not math.isfinite(self.maximum_expected_group_delay_s)
            or self.maximum_expected_group_delay_s <= 0.0
        ):
            raise ValueError(
                "maximum_expected_group_delay_s must be finite and positive"
            )
        if float(np.max(np.diff(frequencies))) > (
            self.maximum_interpolation_gap_hz
            * (1.0 + 16.0 * np.finfo(float).eps)
        ):
            raise ValueError(
                "frequency axis contains a gap larger than the profile limit"
            )
        if float(np.max(np.abs(correction))) > (
            self.maximum_correction_magnitude
            * (1.0 + 16.0 * np.finfo(float).eps)
        ):
            raise ValueError(
                "complex correction exceeds the configured stable gain limit"
            )
        if self.phase_model not in {"direct", "polynomial"}:
            raise ValueError("phase_model must be 'direct' or 'polynomial'")
        reference_plane = self.reference_plane.strip()
        source = self.source.strip()
        if not reference_plane:
            raise ValueError("reference_plane must be recorded")
        if not source:
            raise ValueError("source must be recorded")
        metadata = _normalise_metadata(self.metadata)
        correction_phase = _readonly_float(np.unwrap(np.angle(correction)))

        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "complex_correction", correction)
        object.__setattr__(self, "measured_transfer", measured)
        object.__setattr__(self, "unwrapped_measured_phase_rad", unwrapped_phase)
        object.__setattr__(self, "fitted_phase_rad", fitted_phase)
        object.__setattr__(self, "group_delay_s", group_delay)
        object.__setattr__(self, "reference_plane", reference_plane)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_correction_phase_rad", correction_phase)
        object.__setattr__(
            self,
            "profile_id",
            _profile_digest(
                frequencies, correction, reference_plane, source, metadata
            ),
        )

    @property
    def frequency_dependent_group_delay_s(self) -> NDArray[np.float64]:
        """Group-delay residual after removing the best constant component."""

        residual = self.group_delay_s - self.constant_group_delay_s
        return _readonly_float(residual)

    @property
    def frequency_min_hz(self) -> float:
        return float(self.frequencies_hz[0])

    @property
    def frequency_max_hz(self) -> float:
        return float(self.frequencies_hz[-1])

    def correction_at(self, frequencies_hz: ArrayLike) -> NDArray[np.complex128]:
        """Interpolate correction in log-magnitude and unwrapped phase.

        Extrapolation is intentionally forbidden because an RF calibration is
        valid only inside its measured frequency span.
        """

        target = np.asarray(frequencies_hz, dtype=float)
        _validate_frequency_axis(target)
        if target[0] < self.frequencies_hz[0] or target[-1] > self.frequencies_hz[-1]:
            raise ValueError(
                "requested frequencies fall outside the calibration profile span"
            )
        insertion = np.searchsorted(self.frequencies_hz, target, side="left")
        insertion = np.clip(insertion, 1, self.frequencies_hz.size - 1)
        bracket_gaps = (
            self.frequencies_hz[insertion]
            - self.frequencies_hz[insertion - 1]
        )
        if np.any(
            bracket_gaps
            > self.maximum_interpolation_gap_hz
            * (1.0 + 16.0 * np.finfo(float).eps)
        ):
            raise ValueError(
                "requested frequency lies across an unmeasured calibration gap"
            )
        log_magnitude = np.log(np.abs(self.complex_correction))
        interpolated_log_magnitude = np.interp(
            target, self.frequencies_hz, log_magnitude
        )
        interpolated_phase = np.interp(
            target, self.frequencies_hz, self._correction_phase_rad
        )
        correction = np.exp(interpolated_log_magnitude + 1j * interpolated_phase)
        if not np.all(np.isfinite(correction.real)) or not np.all(
            np.isfinite(correction.imag)
        ):
            raise ValueError("interpolated calibration contains non-finite values")
        return _readonly_complex(correction)

    def summary(self) -> dict[str, MetadataValue]:
        return {
            "profile_id": self.profile_id,
            "source": self.source,
            "reference_plane": self.reference_plane,
            "phase_model": self.phase_model,
            "sample_count": int(self.frequencies_hz.size),
            "frequency_min_hz": self.frequency_min_hz,
            "frequency_max_hz": self.frequency_max_hz,
            "constant_group_delay_s": float(self.constant_group_delay_s),
            "group_delay_ripple_peak_s": float(
                np.max(np.abs(self.frequency_dependent_group_delay_s))
            ),
            "phase_fit_rms_rad": float(self.phase_fit_rms_rad),
            "maximum_interpolation_gap_hz": float(
                self.maximum_interpolation_gap_hz
            ),
            "maximum_correction_magnitude": float(
                self.maximum_correction_magnitude
            ),
            "maximum_expected_group_delay_s": (
                float(self.maximum_expected_group_delay_s)
                if self.maximum_expected_group_delay_s is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Raw and calibrated response evidence from one explicit application."""

    frequencies_hz: NDArray[np.float64]
    raw_response: NDArray[np.complex128]
    calibrated_response: NDArray[np.complex128]
    applied_correction: NDArray[np.complex128]
    profile_id: str
    reference_plane: str
    source: str

    def __post_init__(self) -> None:
        frequencies = _readonly_float(self.frequencies_hz)
        raw = _readonly_complex(self.raw_response)
        calibrated = _readonly_complex(self.calibrated_response)
        correction = _readonly_complex(self.applied_correction)
        _validate_frequency_axis(frequencies)
        _validate_complex_vector(raw, frequencies, name="raw_response")
        _validate_complex_vector(
            calibrated, frequencies, name="calibrated_response"
        )
        _validate_complex_vector(
            correction, frequencies, name="applied_correction"
        )
        if not self.profile_id:
            raise ValueError("profile_id must be recorded")
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "raw_response", raw)
        object.__setattr__(self, "calibrated_response", calibrated)
        object.__setattr__(self, "applied_correction", correction)

    def evidence(self) -> dict[str, MetadataValue]:
        raw_phase = np.unwrap(np.angle(self.raw_response))
        calibrated_phase = np.unwrap(np.angle(self.calibrated_response))
        return {
            "profile_id": self.profile_id,
            "source": self.source,
            "reference_plane": self.reference_plane,
            "sample_count": int(self.frequencies_hz.size),
            "frequency_min_hz": float(self.frequencies_hz[0]),
            "frequency_max_hz": float(self.frequencies_hz[-1]),
            "raw_magnitude_min": float(np.min(np.abs(self.raw_response))),
            "raw_magnitude_max": float(np.max(np.abs(self.raw_response))),
            "calibrated_magnitude_min": float(
                np.min(np.abs(self.calibrated_response))
            ),
            "calibrated_magnitude_max": float(
                np.max(np.abs(self.calibrated_response))
            ),
            "raw_phase_span_rad": float(np.ptp(raw_phase)),
            "calibrated_phase_span_rad": float(np.ptp(calibrated_phase)),
        }


def build_calibration_profile(
    frequencies_hz: ArrayLike,
    measured_transfer: ArrayLike,
    *,
    reference_plane: str,
    source: str,
    metadata: Mapping[str, MetadataValue] | None = None,
    phase_model: PhaseModel = "direct",
    polynomial_order: int = 5,
    minimum_relative_magnitude: float = 1.0e-8,
    maximum_correction_magnitude: float = 1.0e3,
    maximum_interpolation_gap_hz: float | None = None,
    maximum_expected_group_delay_s: float | None = None,
) -> CalibrationProfile:
    """Build an inverse complex calibration from measured transfer data.

    ``direct`` retains the unwrapped phase samples. ``polynomial`` fits a
    smooth polynomial to that phase on a normalised frequency axis.  The
    latter rejects phase noise but cannot reproduce sharp resonances.
    """

    frequencies = np.asarray(frequencies_hz, dtype=float)
    transfer = np.asarray(measured_transfer, dtype=np.complex128)
    _validate_frequency_axis(frequencies)
    _validate_complex_vector(transfer, frequencies, name="measured_transfer")
    if phase_model not in {"direct", "polynomial"}:
        raise ValueError("phase_model must be 'direct' or 'polynomial'")
    if not 0.0 < minimum_relative_magnitude < 1.0:
        raise ValueError("minimum_relative_magnitude must lie in (0, 1)")
    if (
        not math.isfinite(maximum_correction_magnitude)
        or maximum_correction_magnitude < 1.0
    ):
        raise ValueError(
            "maximum_correction_magnitude must be finite and at least one"
        )
    frequency_steps_hz = np.diff(frequencies)
    if maximum_interpolation_gap_hz is None:
        maximum_interpolation_gap_hz = float(
            5.0 * np.median(frequency_steps_hz)
        )
    if (
        not math.isfinite(maximum_interpolation_gap_hz)
        or maximum_interpolation_gap_hz <= 0.0
    ):
        raise ValueError(
            "maximum_interpolation_gap_hz must be finite and positive"
        )
    if float(np.max(frequency_steps_hz)) > maximum_interpolation_gap_hz:
        raise ValueError(
            "frequency axis contains an unmeasured gap larger than the "
            "configured interpolation limit"
        )
    if maximum_expected_group_delay_s is not None:
        if (
            not math.isfinite(maximum_expected_group_delay_s)
            or maximum_expected_group_delay_s <= 0.0
        ):
            raise ValueError(
                "maximum_expected_group_delay_s must be finite and positive"
            )
        if (
            2.0
            * np.pi
            * float(np.max(frequency_steps_hz))
            * maximum_expected_group_delay_s
            >= np.pi
        ):
            raise ValueError(
                "calibration frequency spacing is ambiguous for the declared "
                "maximum expected group delay"
            )
    magnitudes = np.abs(transfer)
    maximum_magnitude = float(np.max(magnitudes))
    if maximum_magnitude <= 0.0:
        raise ValueError("measured_transfer must contain non-zero values")
    if float(np.min(magnitudes)) <= maximum_magnitude * minimum_relative_magnitude:
        raise ValueError(
            "measured_transfer contains a null too deep for stable inversion"
        )

    unwrapped_phase = np.unwrap(np.angle(transfer))
    if phase_model == "direct":
        fitted_phase = unwrapped_phase.copy()
    else:
        if isinstance(polynomial_order, bool) or not isinstance(
            polynomial_order, (int, np.integer)
        ):
            raise TypeError("polynomial_order must be an integer")
        if polynomial_order < 1 or polynomial_order >= frequencies.size:
            raise ValueError(
                "polynomial_order must lie between one and sample_count - 1"
            )
        frequency_center = float(np.mean(frequencies))
        frequency_scale = float(np.ptp(frequencies) / 2.0)
        normalised_frequency = (frequencies - frequency_center) / frequency_scale
        coefficients = np.polynomial.polynomial.polyfit(
            normalised_frequency, unwrapped_phase, polynomial_order
        )
        fitted_phase = np.polynomial.polynomial.polyval(
            normalised_frequency, coefficients
        )

    centered_frequency = frequencies - float(np.mean(frequencies))
    slope = float(
        np.linalg.lstsq(
            np.column_stack((centered_frequency, np.ones_like(frequencies))),
            fitted_phase,
            rcond=None,
        )[0][0]
    )
    constant_group_delay_s = -slope / (2.0 * np.pi)
    edge_order = 2 if frequencies.size >= 3 else 1
    group_delay_s = -np.gradient(
        fitted_phase, frequencies, edge_order=edge_order
    ) / (2.0 * np.pi)
    correction = np.exp(-1j * fitted_phase) / magnitudes
    if float(np.max(np.abs(correction))) > maximum_correction_magnitude:
        raise ValueError(
            "required correction gain exceeds maximum_correction_magnitude"
        )
    phase_fit_rms_rad = float(
        np.sqrt(np.mean(np.square(unwrapped_phase - fitted_phase)))
    )
    return CalibrationProfile(
        frequencies_hz=frequencies,
        complex_correction=correction,
        measured_transfer=transfer,
        unwrapped_measured_phase_rad=unwrapped_phase,
        fitted_phase_rad=fitted_phase,
        group_delay_s=group_delay_s,
        constant_group_delay_s=constant_group_delay_s,
        reference_plane=reference_plane,
        source=source,
        phase_model=phase_model,
        phase_fit_rms_rad=phase_fit_rms_rad,
        maximum_interpolation_gap_hz=float(maximum_interpolation_gap_hz),
        maximum_correction_magnitude=float(maximum_correction_magnitude),
        maximum_expected_group_delay_s=maximum_expected_group_delay_s,
        metadata=_normalise_metadata(metadata),
    )


def load_s2p_calibration_profile(
    path: str | Path,
    *,
    reference_plane: str,
    source: str | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
    phase_model: PhaseModel = "direct",
    polynomial_order: int = 5,
    minimum_relative_magnitude: float = 1.0e-8,
    maximum_correction_magnitude: float = 1.0e3,
    maximum_interpolation_gap_hz: float | None = None,
    maximum_expected_group_delay_s: float | None = None,
) -> CalibrationProfile:
    """Load S21 and build a calibration profile on its native frequency grid."""

    frequencies, s21 = load_s2p_s21(path)
    return build_calibration_profile(
        frequencies,
        s21,
        reference_plane=reference_plane,
        source=source or str(Path(path).resolve()),
        metadata=metadata,
        phase_model=phase_model,
        polynomial_order=polynomial_order,
        minimum_relative_magnitude=minimum_relative_magnitude,
        maximum_correction_magnitude=maximum_correction_magnitude,
        maximum_interpolation_gap_hz=maximum_interpolation_gap_hz,
        maximum_expected_group_delay_s=maximum_expected_group_delay_s,
    )


def apply_calibration(
    frequencies_hz: ArrayLike,
    raw_response: ArrayLike | CalibrationResult,
    profile: CalibrationProfile,
) -> CalibrationResult:
    """Apply one profile while retaining immutable raw/calibrated evidence.

    Passing a prior :class:`CalibrationResult` is rejected to guard against
    accidental double calibration.  Extracting its calibrated array and
    deliberately treating that array as new raw input cannot be detected.
    """

    if isinstance(raw_response, CalibrationResult):
        raise ValueError("calibration has already been applied to this result")
    if not isinstance(profile, CalibrationProfile):
        raise TypeError("profile must be a CalibrationProfile")
    frequencies = np.asarray(frequencies_hz, dtype=float)
    raw = np.asarray(raw_response, dtype=np.complex128)
    _validate_frequency_axis(frequencies)
    _validate_complex_vector(raw, frequencies, name="raw_response")
    correction = profile.correction_at(frequencies)
    calibrated = raw * correction
    if not np.all(np.isfinite(calibrated.real)) or not np.all(
        np.isfinite(calibrated.imag)
    ):
        raise ValueError("calibrated response contains non-finite values")
    return CalibrationResult(
        frequencies_hz=frequencies,
        raw_response=raw,
        calibrated_response=calibrated,
        applied_correction=correction,
        profile_id=profile.profile_id,
        reference_plane=profile.reference_plane,
        source=profile.source,
    )


__all__ = [
    "CalibrationProfile",
    "CalibrationResult",
    "apply_calibration",
    "build_calibration_profile",
    "load_s2p_calibration_profile",
]
