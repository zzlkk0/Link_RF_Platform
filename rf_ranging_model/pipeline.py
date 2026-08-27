"""End-to-end Environment -> Channel -> RFIC -> Ranging -> AI pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .ai.correction import LinearDistanceCorrector
from .ai.features import RangingFeatures, extract_features
from .ai.nlos_classifier import HeuristicNLOSClassifier
from .ble.pbr import PBRResult, estimate_distance
from .ble.robust import (
    RobustPBRConfig,
    RobustPBRResult,
    estimate_distance_robust,
)
from .channel.multipath import delay_spread_rms, frequency_response
from .constants import C0
from .rfic.impairments import RFICImpairmentConfig, apply_rfic_impairments
from .rfic.calibration import (
    CalibrationProfile,
    CalibrationResult,
    apply_calibration,
)
from .types import PropagationPath
from .uwb.cir import (
    combine_frequency_response_frames,
    frequency_response_to_cir,
)
from .uwb.first_path import FirstPathResult, FirstPathStatus
from .uwb.tof import estimate_distance_from_cir


@dataclass(frozen=True, slots=True)
class PipelineResult:
    ble_channel_response: NDArray[np.complex128]
    ble_measured_response: NDArray[np.complex128]
    uwb_channel_response: NDArray[np.complex128]
    uwb_measured_response: NDArray[np.complex128]
    cir_delay_s: NDArray[np.float64]
    cir: NDArray[np.complex128]
    ble_pbr: PBRResult
    ble_raw_distance_m: float
    ble_calibrated_distance_m: float
    uwb_raw_distance_m: float
    uwb_distance_m: float
    uwb_first_path: FirstPathResult
    rfic_group_delay_calibration_s: float
    features: RangingFeatures
    nlos_probability: float
    corrected_ble_distance_m: float
    ble_robust: RobustPBRResult | None = None
    ble_raw_pbr: PBRResult | None = None
    ble_ranging_response: NDArray[np.complex128] | None = None
    uwb_ranging_response: NDArray[np.complex128] | None = None
    raw_cir_delay_s: NDArray[np.float64] | None = None
    raw_cir: NDArray[np.complex128] | None = None
    uwb_raw_first_path: FirstPathResult | None = None
    ble_calibration_result: CalibrationResult | None = None
    uwb_calibration_result: CalibrationResult | None = None
    rfic_calibration_mode: str = "none"
    uwb_frame_count: int = 4
    uwb_frame_coherence: float = 1.0
    uwb_measured_frames: NDArray[np.complex128] | None = None
    ble_robust_calibrated_distance_m: float | None = None


class RangingPipeline:
    """Coordinates deterministic models without hiding intermediate evidence."""

    def __init__(
        self,
        *,
        classifier: HeuristicNLOSClassifier | None = None,
        corrector: LinearDistanceCorrector | None = None,
    ) -> None:
        self.classifier = classifier or HeuristicNLOSClassifier()
        self.corrector = corrector or LinearDistanceCorrector()

    def run(
        self,
        paths: Sequence[PropagationPath],
        ble_frequencies_hz: ArrayLike,
        uwb_frequencies_hz: ArrayLike,
        *,
        rfic_config: RFICImpairmentConfig | None = None,
        seed: int | None = None,
        path_factor: float = 1.0,
        uwb_zero_pad_factor: int = 8,
        uwb_threshold_ratio: float = 0.18,
        uwb_frame_count: int = 4,
        uwb_frame_combining: Literal["coherent", "phase_aligned"] = (
            "phase_aligned"
        ),
        uwb_minimum_frame_coherence: float = 0.75,
        uwb_early_path_enhancement: bool = False,
        calibrate_rfic_group_delay: bool = False,
        rfic_group_delay_calibration_s: float | None = None,
        ble_rfic_calibration_profile: CalibrationProfile | None = None,
        uwb_rfic_calibration_profile: CalibrationProfile | None = None,
        ble_robust_config: RobustPBRConfig | None = None,
    ) -> PipelineResult:
        cfg = rfic_config or RFICImpairmentConfig()
        ble_frequencies = np.asarray(ble_frequencies_hz, dtype=float)
        uwb_frequencies = np.asarray(uwb_frequencies_hz, dtype=float)
        if (
            isinstance(uwb_frame_count, bool)
            or not isinstance(uwb_frame_count, (int, np.integer))
            or int(uwb_frame_count) < 1
        ):
            raise ValueError("uwb_frame_count must be a positive integer")
        uwb_frame_count = int(uwb_frame_count)
        if (
            not np.isfinite(uwb_minimum_frame_coherence)
            or not 0.0 <= uwb_minimum_frame_coherence <= 1.0
        ):
            raise ValueError(
                "uwb_minimum_frame_coherence must lie in [0, 1]"
            )
        if not isinstance(uwb_early_path_enhancement, bool):
            raise ValueError("uwb_early_path_enhancement must be a boolean")
        seed_sequence = np.random.SeedSequence(seed)
        child_sequences = seed_sequence.spawn(1 + uwb_frame_count)
        ble_rng = np.random.default_rng(child_sequences[0])
        uwb_rngs = [
            np.random.default_rng(child)
            for child in child_sequences[1:]
        ]

        ble_channel = frequency_response(ble_frequencies, paths)
        uwb_channel = frequency_response(uwb_frequencies, paths)
        ble_measured = apply_rfic_impairments(
            ble_frequencies, ble_channel, cfg, rng=ble_rng
        )
        uwb_measured_frames = np.asarray(
            [
                apply_rfic_impairments(
                    uwb_frequencies,
                    uwb_channel,
                    cfg,
                    rng=uwb_rng,
                )
                for uwb_rng in uwb_rngs
            ],
            dtype=np.complex128,
        )
        uwb_measured, uwb_frame_coherence = combine_frequency_response_frames(
            uwb_measured_frames,
            combining=uwb_frame_combining,
        )

        if (
            ble_rfic_calibration_profile is not None
            or uwb_rfic_calibration_profile is not None
        ) and (
            calibrate_rfic_group_delay
            or rfic_group_delay_calibration_s is not None
        ):
            raise ValueError(
                "scalar and frequency-dependent RFIC calibration cannot be "
                "applied together"
            )

        quality_config = ble_robust_config or RobustPBRConfig(
            path_factor=path_factor
        )
        if not np.isclose(
            quality_config.path_factor, path_factor, rtol=0.0, atol=0.0
        ):
            raise ValueError(
                "ble_robust_config.path_factor must match pipeline path_factor"
            )

        ble_calibration_result = (
            apply_calibration(
                ble_frequencies,
                ble_measured,
                ble_rfic_calibration_profile,
            )
            if ble_rfic_calibration_profile is not None
            else None
        )
        uwb_calibration_result = (
            apply_calibration(
                uwb_frequencies,
                uwb_measured,
                uwb_rfic_calibration_profile,
            )
            if uwb_rfic_calibration_profile is not None
            else None
        )
        ble_ranging_response = (
            ble_calibration_result.calibrated_response
            if ble_calibration_result is not None
            else ble_measured
        )
        uwb_ranging_response = (
            uwb_calibration_result.calibrated_response
            if uwb_calibration_result is not None
            else uwb_measured
        )

        ble_raw_result = estimate_distance(
            ble_frequencies,
            ble_measured,
            path_factor=path_factor,
            max_distance_m=quality_config.max_distance_m,
            coarse_oversampling=quality_config.coarse_oversampling,
            magnitude_weighting=quality_config.magnitude_weighting,
        )
        ble_result = estimate_distance(
            ble_frequencies,
            ble_ranging_response,
            path_factor=path_factor,
            max_distance_m=quality_config.max_distance_m,
            coarse_oversampling=quality_config.coarse_oversampling,
            magnitude_weighting=quality_config.magnitude_weighting,
        )
        ble_robust = estimate_distance_robust(
            ble_frequencies,
            ble_ranging_response,
            config=quality_config,
        )
        if calibrate_rfic_group_delay:
            if rfic_group_delay_calibration_s is None:
                raise ValueError(
                    "rfic_group_delay_calibration_s must be explicitly "
                    "provided when RFIC group-delay calibration is enabled"
                )
            calibration_delay_s = float(
                rfic_group_delay_calibration_s
            )
            if not np.isfinite(calibration_delay_s):
                raise ValueError(
                    "rfic_group_delay_calibration_s must be finite"
                )
        else:
            if rfic_group_delay_calibration_s is not None:
                raise ValueError(
                    "set calibrate_rfic_group_delay=True to apply an "
                    "explicit RFIC group-delay calibration"
                )
            calibration_delay_s = 0.0
        ble_calibrated_distance_m = (
            ble_result.distance_m
            - C0 * calibration_delay_s / path_factor
        )
        ble_robust_calibrated_distance_m = (
            float(
                ble_robust.distance_m
                - C0 * calibration_delay_s / path_factor
            )
            if ble_robust.distance_m is not None
            else None
        )
        if ble_robust.availability and (
            ble_robust_calibrated_distance_m is None
            or not np.isfinite(ble_robust_calibrated_distance_m)
            or ble_robust_calibrated_distance_m < 0.0
            or ble_robust_calibrated_distance_m > quality_config.max_distance_m
        ):
            calibration_flag = "calibrated_distance_out_of_range"
            ble_robust = replace(
                ble_robust,
                availability=False,
                status="rejected",
                rejection_reason=calibration_flag,
                confidence=0.0,
                quality_flags=tuple(
                    dict.fromkeys((*ble_robust.quality_flags, calibration_flag))
                ),
            )
        raw_cir_delay, raw_cir = frequency_response_to_cir(
            uwb_frequencies,
            uwb_measured,
            zero_pad_factor=uwb_zero_pad_factor,
        )
        uwb_raw_distance_m, raw_first_path = estimate_distance_from_cir(
            raw_cir_delay,
            raw_cir,
            path_factor=path_factor,
            threshold_ratio=uwb_threshold_ratio,
            early_path_enhancement=uwb_early_path_enhancement,
            clean_frequency_sample_count=int(uwb_frequencies.size),
        )
        cir_delay, cir = frequency_response_to_cir(
            uwb_frequencies,
            uwb_ranging_response,
            zero_pad_factor=uwb_zero_pad_factor,
        )
        uwb_distance, first_path = estimate_distance_from_cir(
            cir_delay,
            cir,
            path_factor=path_factor,
            threshold_ratio=uwb_threshold_ratio,
            calibration_delay_s=calibration_delay_s,
            early_path_enhancement=uwb_early_path_enhancement,
            clean_frequency_sample_count=int(uwb_frequencies.size),
        )
        if uwb_frame_count > 1 and (
            uwb_frame_coherence < uwb_minimum_frame_coherence
        ):
            if raw_first_path.status is FirstPathStatus.PASS:
                raw_first_path = replace(
                    raw_first_path,
                    status=FirstPathStatus.LOW_CONFIDENCE,
                    rejection_reason="LOW_FRAME_COHERENCE",
                    confidence=min(raw_first_path.confidence, 0.25),
                )
            if first_path.status is FirstPathStatus.PASS:
                first_path = replace(
                    first_path,
                    status=FirstPathStatus.LOW_CONFIDENCE,
                    rejection_reason="LOW_FRAME_COHERENCE",
                    confidence=min(first_path.confidence, 0.25),
                )

        feature_snr = cfg.snr_db if cfg.snr_db is not None else 120.0
        features = extract_features(
            ble_ranging_response,
            phase_residual_rms_rad=ble_result.residual_rms_rad,
            delay_spread_s=delay_spread_rms(paths),
            snr_db=feature_snr,
            cfo_hz=cfg.cfo_hz,
            iq_gain_mismatch_db=cfg.iq_gain_mismatch_db,
        )
        probability = self.classifier.predict_probability(features)
        corrected = (
            self.corrector.correct(
                float(ble_robust_calibrated_distance_m), features
            )
            if ble_robust.availability
            and ble_robust_calibrated_distance_m is not None
            else float("nan")
        )
        return PipelineResult(
            ble_channel_response=ble_channel,
            ble_measured_response=ble_measured,
            uwb_channel_response=uwb_channel,
            uwb_measured_response=uwb_measured,
            cir_delay_s=cir_delay,
            cir=cir,
            ble_pbr=ble_result,
            ble_raw_distance_m=ble_raw_result.distance_m,
            ble_calibrated_distance_m=float(ble_calibrated_distance_m),
            uwb_raw_distance_m=float(uwb_raw_distance_m),
            uwb_distance_m=float(uwb_distance),
            uwb_first_path=first_path,
            rfic_group_delay_calibration_s=float(calibration_delay_s),
            features=features,
            nlos_probability=probability,
            corrected_ble_distance_m=corrected,
            ble_robust=ble_robust,
            ble_raw_pbr=ble_raw_result,
            ble_ranging_response=ble_ranging_response,
            uwb_ranging_response=uwb_ranging_response,
            raw_cir_delay_s=raw_cir_delay,
            raw_cir=raw_cir,
            uwb_raw_first_path=raw_first_path,
            ble_calibration_result=ble_calibration_result,
            uwb_calibration_result=uwb_calibration_result,
            rfic_calibration_mode=(
                "frequency_profile"
                if ble_calibration_result is not None
                or uwb_calibration_result is not None
                else "scalar"
                if calibrate_rfic_group_delay
                else "none"
            ),
            uwb_frame_count=uwb_frame_count,
            uwb_frame_coherence=float(uwb_frame_coherence),
            uwb_measured_frames=uwb_measured_frames,
            ble_robust_calibrated_distance_m=(
                ble_robust_calibrated_distance_m
            ),
        )
