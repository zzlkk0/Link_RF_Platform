from __future__ import annotations

import unittest

import numpy as np

from rf_ranging_model.ble import (
    RobustPBRConfig,
    allowed_cs_frequencies_hz,
    estimate_distance_robust,
)
from rf_ranging_model.channel.multipath import frequency_response
from rf_ranging_model.constants import C0
from rf_ranging_model.rfic.impairments import (
    RFICImpairmentConfig,
    apply_rfic_impairments,
)
from rf_ranging_model.rfic.calibration import build_calibration_profile
from rf_ranging_model.types import PropagationPath


class RobustPBRTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frequencies = allowed_cs_frequencies_hz()

    def response(self, distance_m: float, path_factor: float = 1.0) -> np.ndarray:
        delay_s = path_factor * distance_m / C0
        return np.exp(-1j * 2.0 * np.pi * self.frequencies * delay_s)

    def test_ideal_official_map_preserves_precision_and_reports_evidence(self) -> None:
        for distance_m in (0.5, 50.0, 149.0):
            with self.subTest(path_factor=1.0, distance_m=distance_m):
                result = estimate_distance_robust(
                    self.frequencies, self.response(distance_m)
                )
                self.assertTrue(result.availability)
                self.assertEqual(result.status, "available")
                self.assertIsNone(result.rejection_reason)
                self.assertLess(abs(result.distance_m - distance_m), 1.0e-8)
                self.assertGreater(result.confidence, 0.8)
                self.assertEqual(result.frequency_grid_spacing_hz, 1.0e6)
                self.assertAlmostEqual(
                    result.unambiguous_distance_m, C0 / 1.0e6
                )
                self.assertIsNotNone(result.secondary_peak_delay_s)
                self.assertGreater(result.peak_ratio, 4.0)

        round_trip = estimate_distance_robust(
            self.frequencies,
            self.response(100.0, path_factor=2.0),
            config=RobustPBRConfig(path_factor=2.0, max_distance_m=120.0),
        )
        self.assertTrue(round_trip.availability)
        self.assertLess(abs(round_trip.distance_m - 100.0), 1.0e-8)
        self.assertAlmostEqual(
            round_trip.unambiguous_distance_m, C0 / (2.0e6)
        )

    def test_unsorted_tones_are_safely_reordered_or_explicitly_rejected(self) -> None:
        rng = np.random.default_rng(41)
        order = rng.permutation(self.frequencies.size)
        response = self.response(80.0)
        sorted_result = estimate_distance_robust(
            self.frequencies[order], response[order]
        )
        self.assertTrue(sorted_result.availability)
        self.assertFalse(sorted_result.input_was_sorted)
        self.assertLess(abs(sorted_result.distance_m - 80.0), 1.0e-8)

        rejected = estimate_distance_robust(
            self.frequencies[order],
            response[order],
            config=RobustPBRConfig(sort_frequencies=False),
        )
        self.assertFalse(rejected.availability)
        self.assertEqual(rejected.status, "invalid_input")
        self.assertEqual(
            rejected.rejection_reason, "frequencies_not_strictly_increasing"
        )

    def test_missing_tones_and_contiguous_notch_remain_usable(self) -> None:
        response = self.response(80.0)
        rng = np.random.default_rng(8)
        random_half = np.sort(
            rng.choice(self.frequencies.size, 36, replace=False)
        )
        notch = np.flatnonzero(
            ~(
                (self.frequencies >= 2.430e9)
                & (self.frequencies <= 2.450e9)
            )
        )
        for label, selected in (("random_half", random_half), ("notch", notch)):
            with self.subTest(channel_map=label):
                result = estimate_distance_robust(
                    self.frequencies[selected], response[selected]
                )
                self.assertTrue(result.availability)
                self.assertEqual(result.frequency_grid_spacing_hz, 1.0e6)
                self.assertLess(abs(result.distance_m - 80.0), 1.0e-8)
                self.assertGreaterEqual(
                    result.peak_ratio,
                    RobustPBRConfig().minimum_peak_ratio,
                )

    def test_actual_sparse_map_controls_the_unambiguous_range(self) -> None:
        channel_indices = np.rint(
            (self.frequencies - 2.402e9) / 1.0e6
        ).astype(int)
        selected = channel_indices % 4 == 0
        sparse_frequencies = self.frequencies[selected]
        sparse_response = self.response(20.0, path_factor=2.0)[selected]

        unsafe = estimate_distance_robust(
            sparse_frequencies,
            sparse_response,
            config=RobustPBRConfig(path_factor=2.0, max_distance_m=100.0),
        )
        self.assertFalse(unsafe.availability)
        self.assertEqual(unsafe.status, "rejected")
        self.assertEqual(
            unsafe.rejection_reason,
            "search_prior_exceeds_unambiguous_range",
        )
        self.assertEqual(unsafe.frequency_grid_spacing_hz, 4.0e6)
        self.assertAlmostEqual(
            unsafe.unambiguous_distance_m, C0 / (2.0 * 4.0e6)
        )

        safe = estimate_distance_robust(
            sparse_frequencies,
            sparse_response,
            config=RobustPBRConfig(path_factor=2.0, max_distance_m=30.0),
        )
        self.assertTrue(safe.availability)
        self.assertLess(abs(safe.distance_m - 20.0), 1.0e-8)

    def test_out_of_prior_estimate_is_rejected_without_throwing(self) -> None:
        result = estimate_distance_robust(
            self.frequencies,
            self.response(100.0),
            config=RobustPBRConfig(max_distance_m=50.0),
        )
        self.assertFalse(result.availability)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(
            result.rejection_reason,
            "estimated_distance_outside_search_prior",
        )
        self.assertIsNone(result.distance_m)

    def test_independent_peak_and_residual_gates_are_configurable(self) -> None:
        direct_delay_s = 10.0 / C0
        ambiguous_response = (
            np.exp(-1j * 2.0 * np.pi * self.frequencies * direct_delay_s)
            + np.exp(0.7j)
            * np.exp(
                -1j
                * 2.0
                * np.pi
                * self.frequencies
                * (direct_delay_s + 20.0e-9)
            )
        )
        ambiguous = estimate_distance_robust(
            self.frequencies,
            ambiguous_response,
            config=RobustPBRConfig(
                minimum_peak_ratio=1.25,
                maximum_residual_rms_rad=10.0,
                minimum_confidence=0.0,
            ),
        )
        self.assertFalse(ambiguous.availability)
        self.assertEqual(ambiguous.rejection_reason, "ambiguous_delay_peaks")
        self.assertEqual(
            ambiguous.model_rejection_reason,
            "multipath_path_competition_too_high",
        )
        self.assertIn(
            "multipath_path_competition_too_high",
            ambiguous.quality_flags,
        )
        self.assertTrue(ambiguous.strong_multipath_evidence)
        self.assertGreaterEqual(ambiguous.path_competition_ratio, 0.98)
        self.assertLess(ambiguous.peak_ratio, 1.01)
        self.assertGreater(ambiguous.independent_peak_count, 0)

        high_residual = estimate_distance_robust(
            self.frequencies,
            ambiguous_response,
            config=RobustPBRConfig(
                minimum_peak_ratio=1.0,
                maximum_residual_rms_rad=0.20,
                minimum_confidence=0.0,
                multipath_model_enabled=False,
            ),
        )
        self.assertFalse(high_residual.availability)
        self.assertEqual(
            high_residual.rejection_reason, "phase_residual_too_high"
        )
        self.assertGreater(high_residual.residual_rms_rad, 0.20)

    def test_resolved_multipath_candidate_improves_effective_phase_slope(self) -> None:
        true_distance_m = 10.0
        direct_delay_s = true_distance_m / C0
        reflection_delay_s = direct_delay_s + 27.82112856561002e-9
        measured = np.exp(-1j * 2.0 * np.pi * self.frequencies * direct_delay_s)
        measured += (
            0.5308916829315422
            * np.exp(2.7142629235622007j)
            * np.exp(
                -1j
                * 2.0
                * np.pi
                * self.frequencies
                * reflection_delay_s
            )
        )

        result = estimate_distance_robust(self.frequencies, measured)
        self.assertTrue(result.availability)
        self.assertEqual(result.estimator_mode, "multipath_matrix_pencil")
        self.assertTrue(result.strong_multipath_evidence)
        self.assertEqual(result.model_order, 2)
        self.assertEqual(len(result.path_delays_s), 2)
        self.assertEqual(len(result.path_amplitudes), 2)
        self.assertLess(abs(result.distance_m - true_distance_m), 1.0e-8)
        self.assertGreater(
            abs(result.estimate.distance_m - true_distance_m),
            0.30,
        )
        self.assertLess(result.reconstruction_residual, 1.0e-8)
        self.assertGreater(result.minimum_path_separation_s, 1.0 / 78.0e6)

        confidence_gated = estimate_distance_robust(
            self.frequencies,
            measured,
            config=RobustPBRConfig(minimum_confidence=0.90),
        )
        self.assertTrue(confidence_gated.strong_multipath_evidence)
        self.assertFalse(confidence_gated.availability)
        self.assertIn(
            "confidence_below_threshold", confidence_gated.quality_flags
        )

    def test_noise_gate_rejects_catastrophic_outliers(self) -> None:
        true_distance_m = 100.0
        clean = self.response(true_distance_m, path_factor=2.0)
        config = RobustPBRConfig(path_factor=2.0, max_distance_m=120.0)
        available_errors: list[float] = []
        rejected_count = 0
        for seed in range(200):
            rng = np.random.default_rng(seed)
            snr_db = 5.0
            noise_scale = np.sqrt(0.5 * 10.0 ** (-snr_db / 10.0))
            measured = clean + noise_scale * (
                rng.normal(size=clean.size)
                + 1j * rng.normal(size=clean.size)
            )
            result = estimate_distance_robust(
                self.frequencies, measured, config=config
            )
            if result.availability:
                available_errors.append(result.distance_m - true_distance_m)
            else:
                rejected_count += 1

        errors = np.asarray(available_errors)
        self.assertGreaterEqual(errors.size, 180)
        self.assertGreater(rejected_count, 0)
        self.assertEqual(float(np.mean(np.abs(errors) > 1.0)), 0.0)
        self.assertLess(float(np.quantile(np.abs(errors), 0.95)), 0.15)

    def test_pure_noise_never_becomes_an_available_sparse_path(self) -> None:
        available_count = 0
        for seed in range(128):
            rng = np.random.default_rng(50_000 + seed)
            pure_noise = (
                rng.normal(size=self.frequencies.size)
                + 1j * rng.normal(size=self.frequencies.size)
            ) / np.sqrt(2.0)
            result = estimate_distance_robust(
                self.frequencies,
                pure_noise,
                config=RobustPBRConfig(max_distance_m=120.0),
            )
            available_count += int(result.availability)
        self.assertEqual(available_count, 0)

    def test_fixed_complex_multipath_uses_independent_reference_calibration(
        self,
    ) -> None:
        rng = np.random.default_rng(20260821)
        accepted_errors: list[float] = []
        accepted_legacy_errors: list[float] = []
        rejection_reasons: list[str] = []
        multipath_mode_count = 0
        calibration_rng = np.random.default_rng(20260822)

        for index in range(200):
            true_distance_m = float(rng.uniform(1.0, 20.0))
            direct_delay_s = true_distance_m / C0
            paths = [PropagationPath(direct_delay_s, 1.0 + 0.0j, "LOS")]
            for reflection_index in range(int(rng.integers(1, 5))):
                excess_delay_s = float(rng.uniform(3.0e-9, 30.0e-9))
                magnitude = float(rng.uniform(0.1, 0.75))
                phase_rad = float(rng.uniform(-np.pi, np.pi))
                paths.append(
                    PropagationPath(
                        direct_delay_s + excess_delay_s,
                        magnitude * np.exp(1j * phase_rad),
                        f"reflection_{reflection_index}",
                    )
                )
            snr_db = float(rng.uniform(15.0, 40.0))
            injected_group_delay_s = float(rng.uniform(0.0, 0.5e-9))
            calibration_reference_plane_offset_s = float(
                rng.normal(0.0, 0.02e-9)
            )
            impairment = RFICImpairmentConfig(
                base_group_delay_s=injected_group_delay_s,
                group_delay_ripple_s=float(rng.uniform(0.0, 0.12e-9)),
                amplitude_ripple_db=float(rng.uniform(0.0, 0.8)),
                phase_noise_std_deg=float(rng.uniform(0.2, 2.0)),
                cfo_hz=float(rng.uniform(-50.0, 50.0)),
                tone_interval_s=8.0e-6,
                snr_db=snr_db,
            )
            calibration_impairment = RFICImpairmentConfig(
                phase_offset_rad=impairment.phase_offset_rad,
                base_group_delay_s=(
                    impairment.base_group_delay_s
                    + calibration_reference_plane_offset_s
                ),
                group_delay_ripple_s=impairment.group_delay_ripple_s,
                group_delay_ripple_period_hz=(
                    impairment.group_delay_ripple_period_hz
                ),
                amplitude_ripple_db=impairment.amplitude_ripple_db,
                iq_gain_mismatch_db=impairment.iq_gain_mismatch_db,
                iq_phase_mismatch_deg=impairment.iq_phase_mismatch_deg,
                phase_noise_std_deg=0.2,
                cfo_hz=0.0,
                tone_interval_s=0.0,
                snr_db=40.0,
            )
            reference_response = apply_rfic_impairments(
                self.frequencies,
                np.ones(self.frequencies.size, dtype=np.complex128),
                calibration_impairment,
                rng=calibration_rng,
            )
            external_calibration_s = build_calibration_profile(
                self.frequencies,
                reference_response,
                reference_plane="BLE test fixture connector plane",
                source="independent synthetic known-through sweep",
            ).constant_group_delay_s
            ble_rng = np.random.default_rng(
                np.random.SeedSequence(index).spawn(2)[0]
            )
            measured = apply_rfic_impairments(
                self.frequencies,
                frequency_response(self.frequencies, paths),
                impairment,
                rng=ble_rng,
            )

            # Truth is used only for scoring after the estimator returns.  The
            # scalar correction comes from an independent noisy reference sweep.
            result = estimate_distance_robust(self.frequencies, measured)
            if result.availability:
                calibrated_candidate_m = (
                    result.distance_m - C0 * external_calibration_s
                )
                calibrated_legacy_m = (
                    result.estimate.distance_m - C0 * external_calibration_s
                )
                accepted_errors.append(
                    calibrated_candidate_m - true_distance_m
                )
                accepted_legacy_errors.append(
                    calibrated_legacy_m - true_distance_m
                )
                multipath_mode_count += int(
                    result.estimator_mode == "multipath_matrix_pencil"
                )
            else:
                rejection_reasons.append(result.rejection_reason or "unspecified")

        errors = np.asarray(accepted_errors)
        legacy_errors = np.asarray(accepted_legacy_errors)
        availability = errors.size / 200.0
        p95_m = float(np.percentile(np.abs(errors), 95))
        legacy_p95_m = float(np.percentile(np.abs(legacy_errors), 95))
        error_gt_1m_rate = float(np.mean(np.abs(errors) > 1.0))

        self.assertGreaterEqual(availability, 0.65)
        self.assertLess(p95_m, 0.75)
        self.assertLessEqual(error_gt_1m_rate, 0.01)
        self.assertLess(p95_m, legacy_p95_m)
        self.assertGreater(multipath_mode_count, 100)
        self.assertIn("unresolved_multipath_components", rejection_reasons)
        self.assertIn(
            "sparse_multipath_challenger_disagrees",
            rejection_reasons,
        )

    def test_malformed_inputs_return_explicit_invalid_status(self) -> None:
        valid_response = self.response(10.0)
        cases: list[tuple[str, np.ndarray, np.ndarray, str]] = []

        nonfinite = valid_response.copy()
        nonfinite[4] = np.nan + 0.0j
        cases.append(("nonfinite", self.frequencies, nonfinite, "nonfinite_input"))

        duplicate_frequencies = self.frequencies.copy()
        duplicate_frequencies[5] = duplicate_frequencies[4]
        cases.append(
            (
                "duplicate",
                duplicate_frequencies,
                valid_response,
                "duplicate_frequencies",
            )
        )
        cases.append(
            (
                "all_zero",
                self.frequencies,
                np.zeros_like(valid_response),
                "all_zero_response",
            )
        )
        cases.append(
            (
                "shape_mismatch",
                self.frequencies[:-1],
                valid_response,
                "frequency_response_shape_mismatch",
            )
        )
        cases.append(
            (
                "too_few",
                self.frequencies[:7],
                valid_response[:7],
                "insufficient_tone_count",
            )
        )

        for label, frequencies, response, reason in cases:
            with self.subTest(case=label):
                result = estimate_distance_robust(frequencies, response)
                self.assertFalse(result.availability)
                self.assertEqual(result.status, "invalid_input")
                self.assertEqual(result.rejection_reason, reason)
                self.assertEqual(result.confidence, 0.0)

        off_grid = self.frequencies.copy()
        off_grid[12] += 0.25
        off_grid_result = estimate_distance_robust(off_grid, valid_response)
        self.assertFalse(off_grid_result.availability)
        self.assertEqual(off_grid_result.status, "rejected")
        self.assertEqual(
            off_grid_result.rejection_reason,
            "frequency_grid_not_identifiable",
        )


if __name__ == "__main__":
    unittest.main()
