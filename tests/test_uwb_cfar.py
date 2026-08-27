from __future__ import annotations

import math
import unittest

import numpy as np

from rf_ranging_model.channel.multipath import frequency_response
from rf_ranging_model.constants import C0
from rf_ranging_model.rfic.impairments import (
    RFICImpairmentConfig,
    apply_rfic_impairments,
)
from rf_ranging_model.types import PropagationPath
from rf_ranging_model.uwb.cir import (
    combine_frequency_response_frames,
    frequency_response_frames_to_cir,
    frequency_response_to_cir,
)
from rf_ranging_model.uwb.first_path import (
    FirstPathStatus,
    detect_first_path,
)
from rf_ranging_model.uwb.tof import estimate_distance_from_cir


class UWBFirstPathCFARTests(unittest.TestCase):
    frequencies_hz = np.linspace(3.5e9, 4.5e9, 1001)

    def _cir(
        self,
        paths: list[PropagationPath],
        *,
        snr_db: float | None = None,
        seed: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        channel = frequency_response(self.frequencies_hz, paths)
        measured = apply_rfic_impairments(
            self.frequencies_hz,
            channel,
            RFICImpairmentConfig(snr_db=snr_db),
            rng=np.random.default_rng(seed),
        )
        return frequency_response_to_cir(
            self.frequencies_hz,
            measured,
            zero_pad_factor=8,
        )

    def _cfr(
        self,
        paths: list[PropagationPath],
        *,
        snr_db: float | None = None,
        seed: int = 1,
    ) -> np.ndarray:
        channel = frequency_response(self.frequencies_hz, paths)
        return apply_rfic_impairments(
            self.frequencies_hz,
            channel,
            RFICImpairmentConfig(snr_db=snr_db),
            rng=np.random.default_rng(seed),
        )

    def test_ideal_first_lobe_peak_precision_is_preserved(self) -> None:
        errors_m: list[float] = []
        for distance_m in (1.0, 3.0, 5.0, 8.0, 12.0, 20.0, 149.0):
            delays, cir = self._cir(
                [PropagationPath(distance_m / C0, 1.0 + 0.0j)]
            )
            result = detect_first_path(delays, cir)
            errors_m.append(C0 * result.delay_s - distance_m)
            self.assertEqual(result.status, FirstPathStatus.PASS)
            self.assertTrue(result.available)
        self.assertLess(max(np.abs(errors_m)), 0.04)

    def test_pure_noise_has_no_false_ranges_in_deterministic_sweep(self) -> None:
        delays = np.arange(4096, dtype=float) * 0.125e-9
        false_ranges = 0
        for seed in range(128):
            rng = np.random.default_rng(seed)
            noise = (
                rng.normal(size=delays.size)
                + 1j * rng.normal(size=delays.size)
            ) / np.sqrt(2.0)
            result = detect_first_path(delays, noise)
            false_ranges += int(result.available)
            self.assertEqual(result.status, FirstPathStatus.NO_PATH)
            self.assertTrue(math.isnan(result.delay_s))
        self.assertEqual(false_ranges, 0)

        # The platform path is a Hann-windowed, zero-padded IFFT, whose delay
        # samples are correlated. Exercise that representation separately.
        cfr_false_ranges = 0
        for seed in range(64):
            rng = np.random.default_rng(10_000 + seed)
            noise_cfr = (
                rng.normal(size=self.frequencies_hz.size)
                + 1j * rng.normal(size=self.frequencies_hz.size)
            ) / np.sqrt(2.0)
            cfr_delays, noise_cir = frequency_response_to_cir(
                self.frequencies_hz, noise_cfr, zero_pad_factor=8
            )
            cfr_result = detect_first_path(cfr_delays, noise_cir)
            cfr_false_ranges += int(cfr_result.available)
        self.assertEqual(cfr_false_ranges, 0)

        distance_m, result = estimate_distance_from_cir(delays, noise)
        self.assertTrue(math.isnan(distance_m))
        self.assertEqual(result.status, FirstPathStatus.NO_PATH)
        with self.assertRaisesRegex(ValueError, "no available first path"):
            estimate_distance_from_cir(
                delays, noise, require_available=True
            )

    def test_weak_first_path_before_strong_late_path_is_retained_as_low_confidence(
        self,
    ) -> None:
        distance_m = 5.0
        direct_delay_s = distance_m / C0
        delays, cir = self._cir(
            [
                PropagationPath(direct_delay_s, 0.10 + 0.0j, "weak_first"),
                PropagationPath(
                    direct_delay_s + 10.0e-9,
                    1.0 + 0.0j,
                    "strong_late",
                ),
            ]
        )
        result = detect_first_path(
            delays,
            cir,
            early_path_enhancement=True,
        )
        error_m = C0 * result.delay_s - distance_m
        self.assertTrue(result.available)
        self.assertEqual(result.status, FirstPathStatus.LOW_CONFIDENCE)
        self.assertLess(result.confidence, 1.0)
        self.assertLess(abs(error_m), 0.04)
        self.assertLess(result.peak_to_global_ratio, 0.18)
        self.assertIn(
            "FIRST_LOBE_BELOW_PRIMARY_THRESHOLD",
            result.rejection_reason or "",
        )

    def test_phase_aligned_cfr_combiner_removes_only_common_phase(self) -> None:
        direct_delay_s = 5.0 / C0
        response = frequency_response(
            self.frequencies_hz,
            [
                PropagationPath(direct_delay_s, 0.2 + 0.0j),
                PropagationPath(direct_delay_s + 8.0e-9, 1.0j),
            ],
        )
        phases = np.asarray([0.0, 0.7, -1.2, 2.1])
        frames = np.asarray(
            [response * np.exp(1j * phase) for phase in phases]
        )
        combined, coherence = combine_frequency_response_frames(frames)
        np.testing.assert_allclose(combined, response, rtol=1.0e-12, atol=1.0e-12)
        self.assertAlmostEqual(coherence, 1.0, places=12)

        unaligned, unaligned_coherence = combine_frequency_response_frames(
            frames,
            combining="coherent",
        )
        self.assertLess(np.linalg.norm(unaligned), 0.75 * np.linalg.norm(response))
        self.assertLess(unaligned_coherence, 0.75)

        with self.assertRaisesRegex(ValueError, "1-D or 2-D"):
            combine_frequency_response_frames(np.zeros((2, 3, 4)))
        with self.assertRaisesRegex(ValueError, "frequency-sample axis"):
            frequency_response_frames_to_cir(
                self.frequencies_hz[:-1], frames
            )

    def test_five_db_multiframe_weak_first_path_stress(self) -> None:
        distance_m = 5.0
        direct_delay_s = distance_m / C0
        frame_count = 4
        errors_m: list[float] = []
        late_locks = 0
        unavailable = 0
        coherences: list[float] = []
        for trial in range(200):
            late_phase = 2.0 * np.pi * (trial % 16) / 16.0
            paths = [
                PropagationPath(direct_delay_s, 0.10 + 0.0j, "weak_first"),
                PropagationPath(
                    direct_delay_s + 10.0e-9,
                    np.exp(1j * late_phase),
                    "strong_late",
                ),
            ]
            frames = []
            for frame in range(frame_count):
                response = self._cfr(
                    paths,
                    snr_db=5.0,
                    seed=trial * frame_count + frame,
                )
                # Receiver restarts commonly add an arbitrary common phase.
                common_phase = 0.83 * frame + 0.17 * trial
                frames.append(response * np.exp(1j * common_phase))
            delays, cir, coherence = frequency_response_frames_to_cir(
                self.frequencies_hz,
                np.asarray(frames),
                combining="phase_aligned",
                zero_pad_factor=8,
            )
            result = detect_first_path(delays, cir)
            coherences.append(coherence)
            if not result.available:
                unavailable += 1
                continue
            error_m = C0 * result.delay_s - distance_m
            errors_m.append(error_m)
            late_locks += int(abs(error_m) > 1.0)

        correct_first_rate = float(
            np.mean(np.abs(np.asarray(errors_m)) <= 0.10)
        )
        p95_m = float(np.quantile(np.abs(errors_m), 0.95))
        self.assertEqual(unavailable, 0)
        self.assertGreaterEqual(correct_first_rate, 0.95)
        self.assertLessEqual(late_locks / 200.0, 0.01)
        self.assertLessEqual(p95_m, 0.10)
        self.assertGreater(float(np.median(coherences)), 0.75)

    def test_single_frame_weak_early_evidence_is_not_promoted_to_pass(self) -> None:
        distance_m = 5.0
        direct_delay_s = distance_m / C0
        delays, cir = self._cir(
            [
                PropagationPath(direct_delay_s, 0.10 + 0.0j, "weak_first"),
                PropagationPath(
                    direct_delay_s + 10.0e-9,
                    1.0 + 0.0j,
                    "strong_late",
                ),
            ],
            snr_db=5.0,
            seed=0,
        )
        result = detect_first_path(
            delays,
            cir,
            early_path_enhancement=True,
        )
        self.assertEqual(result.status, FirstPathStatus.LOW_CONFIDENCE)
        self.assertIn(
            "EARLY_PATH",
            result.rejection_reason or "",
        )
        # One noisy frame is below the reliable recovery gate.  The detector
        # must expose ambiguity rather than claim that the late peak is PASS.
        self.assertFalse(result.early_path_enhanced)

        late_path_passes = 0
        for trial in range(200):
            late_phase = 2.0 * np.pi * (trial % 16) / 16.0
            delays, cir = self._cir(
                [
                    PropagationPath(
                        direct_delay_s, 0.10 + 0.0j, "weak_first"
                    ),
                    PropagationPath(
                        direct_delay_s + 10.0e-9,
                        np.exp(1j * late_phase),
                        "strong_late",
                    ),
                ],
                snr_db=5.0,
                seed=100_000 + trial,
            )
            trial_result = detect_first_path(
                delays,
                cir,
                early_path_enhancement=True,
            )
            if trial_result.available:
                trial_error_m = C0 * trial_result.delay_s - distance_m
                late_path_passes += int(
                    abs(trial_error_m) > 1.0
                    and trial_result.status is FirstPathStatus.PASS
                )
        self.assertEqual(late_path_passes, 0)

        clean_delays, clean_cir = self._cir(
            [PropagationPath(direct_delay_s, 1.0 + 0.0j, "single_los")]
        )
        conservative_single_los = detect_first_path(
            clean_delays,
            clean_cir,
            early_path_enhancement=True,
        )
        self.assertEqual(
            conservative_single_los.status,
            FirstPathStatus.LOW_CONFIDENCE,
        )
        self.assertIn(
            "EARLY_PATH_NOT_CONFIRMED_IN_CONSERVATIVE_MODE",
            conservative_single_los.rejection_reason or "",
        )

    def test_low_snr_transitions_from_range_to_no_path(self) -> None:
        distance_m = 5.0
        path = PropagationPath(distance_m / C0, 1.0 + 0.0j)

        delays, cir = self._cir([path], snr_db=-12.0, seed=1)
        detected = detect_first_path(delays, cir)
        self.assertTrue(detected.available)
        self.assertEqual(detected.status, FirstPathStatus.PASS)
        self.assertLess(abs(C0 * detected.delay_s - distance_m), 0.08)

        delays, cir = self._cir([path], snr_db=-15.0, seed=1)
        rejected = detect_first_path(delays, cir)
        self.assertFalse(rejected.available)
        self.assertEqual(rejected.status, FirstPathStatus.NO_PATH)
        self.assertIn(
            rejected.rejection_reason,
            {
                "PEAK_BELOW_CFAR_THRESHOLD",
                "NO_RUN_MEETS_MIN_CONSECUTIVE_SAMPLES",
            },
        )

        delays, cir = self._cir([path], snr_db=-18.0, seed=1)
        very_low_snr = detect_first_path(delays, cir)
        self.assertEqual(very_low_snr.status, FirstPathStatus.NO_PATH)

    def test_near_equal_multipath_is_flagged_as_ambiguous(self) -> None:
        distance_m = 5.0
        direct_delay_s = distance_m / C0
        delays, cir = self._cir(
            [
                PropagationPath(direct_delay_s, 1.0 + 0.0j, "first"),
                PropagationPath(
                    direct_delay_s + 2.0e-9,
                    1.0 + 0.0j,
                    "overlapping_equal_path",
                ),
            ]
        )
        result = detect_first_path(delays, cir)
        error_m = C0 * result.delay_s - distance_m
        self.assertTrue(result.available)
        self.assertEqual(result.status, FirstPathStatus.LOW_CONFIDENCE)
        self.assertLess(result.confidence, 1.0)
        self.assertGreaterEqual(result.ambiguity_ratio, 0.80)
        self.assertIn(
            "COMPETING_OR_OVERLAPPING_LOBE",
            result.rejection_reason or "",
        )
        # Equal paths inside the c/B-scale main lobe are not resolvable here;
        # the detector must surface that boundary instead of claiming PASS.
        self.assertGreater(abs(error_m), 0.30)

    def test_search_boundary_clipping_is_low_confidence(self) -> None:
        distance_m = 5.0
        direct_delay_s = distance_m / C0
        delays, cir = self._cir(
            [PropagationPath(direct_delay_s, 1.0 + 0.0j)]
        )
        result = detect_first_path(
            delays,
            cir,
            search_start_s=direct_delay_s - 1.0e-9,
            search_end_s=direct_delay_s + 5.0e-9,
        )
        self.assertTrue(result.available)
        self.assertEqual(result.status, FirstPathStatus.LOW_CONFIDENCE)
        self.assertLess(result.confidence, 1.0)
        self.assertIn(
            "LOBE_CLIPPED_BY_SEARCH_BOUNDARY",
            result.rejection_reason or "",
        )
        self.assertLess(abs(C0 * result.delay_s - distance_m), 0.04)

    def test_zero_energy_and_isolated_spike_return_no_path(self) -> None:
        delays = np.arange(256, dtype=float) * 0.125e-9
        zero_result = detect_first_path(delays, np.zeros(delays.size))
        self.assertEqual(zero_result.status, FirstPathStatus.NO_PATH)
        self.assertEqual(zero_result.rejection_reason, "ZERO_ENERGY")

        isolated = np.zeros(delays.size, dtype=np.complex128)
        isolated[100] = 1.0 + 0.0j
        spike_result = detect_first_path(delays, isolated)
        self.assertEqual(spike_result.status, FirstPathStatus.NO_PATH)
        self.assertEqual(
            spike_result.rejection_reason,
            "NO_RUN_MEETS_MIN_CONSECUTIVE_SAMPLES",
        )


if __name__ == "__main__":
    unittest.main()
