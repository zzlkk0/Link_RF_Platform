from __future__ import annotations

import unittest

import numpy as np

from rf_ranging_model.ble.pbr import estimate_distance
from rf_ranging_model.channel.multipath import frequency_response
from rf_ranging_model.constants import C0
from rf_ranging_model.pipeline import RangingPipeline
from rf_ranging_model.rfic.impairments import (
    RFICImpairmentConfig,
    apply_rfic_impairments,
)
from rf_ranging_model.rfic.calibration import build_calibration_profile
from rf_ranging_model.types import PropagationPath
from rf_ranging_model.uwb.cir import frequency_response_to_cir
from rf_ranging_model.uwb.first_path import detect_first_path


class RangingPipelineTests(unittest.TestCase):
    def test_constant_group_delay_maps_to_distance_bias(self) -> None:
        frequencies = np.linspace(2.402e9, 2.480e9, 80)
        true_distance = 5.0
        added_delay = 1.2e-9
        channel = frequency_response(
            frequencies,
            [PropagationPath(true_distance / C0, 1.0 + 0.0j)],
        )
        measured = apply_rfic_impairments(
            frequencies,
            channel,
            RFICImpairmentConfig(base_group_delay_s=added_delay),
        )
        estimate = estimate_distance(frequencies, measured)
        expected = true_distance + C0 * added_delay
        self.assertAlmostEqual(estimate.distance_m, expected, places=8)

    def test_pipeline_exposes_ble_uwb_and_ai_results(self) -> None:
        true_distance = 6.0
        direct_delay = true_distance / C0
        paths = [
            PropagationPath(direct_delay, 1.0 + 0.0j, "LOS"),
            PropagationPath(direct_delay + 7.0e-9, -0.4 + 0.1j, "reflection"),
        ]
        ble_frequencies = np.arange(2.402e9, 2.481e9, 2.0e6)
        uwb_frequencies = np.linspace(3.5e9, 4.5e9, 1001)
        result = RangingPipeline().run(
            paths,
            ble_frequencies,
            uwb_frequencies,
            rfic_config=RFICImpairmentConfig(snr_db=45.0),
            seed=11,
        )
        self.assertEqual(result.ble_measured_response.shape, ble_frequencies.shape)
        self.assertEqual(result.uwb_measured_response.shape, uwb_frequencies.shape)
        self.assertTrue(np.isfinite(result.ble_pbr.distance_m))
        self.assertLess(abs(result.uwb_distance_m - true_distance), 0.8)
        self.assertGreaterEqual(result.nlos_probability, 0.0)
        self.assertLessEqual(result.nlos_probability, 1.0)

    def test_uwb_first_lobe_peak_removes_threshold_leading_edge_bias(self) -> None:
        frequencies = np.linspace(3.5e9, 4.5e9, 1001)
        for true_distance in (1.0, 3.0, 5.0, 8.0, 12.0, 20.0):
            with self.subTest(distance_m=true_distance):
                response = frequency_response(
                    frequencies,
                    [PropagationPath(true_distance / C0, 1.0 + 0.0j)],
                )
                delay_s, cir = frequency_response_to_cir(
                    frequencies, response
                )
                result = detect_first_path(delay_s, cir)
                error_m = C0 * result.delay_s - true_distance
                crossing_error_m = (
                    C0 * result.crossing_delay_s - true_distance
                )
                self.assertLess(abs(error_m), 0.04)
                self.assertLess(crossing_error_m, -0.30)
                self.assertEqual(result.method, "first_lobe_peak")

    def test_pipeline_exposes_phase_aligned_multiframe_uwb_evidence(self) -> None:
        true_distance_m = 5.0
        direct_delay_s = true_distance_m / C0
        result = RangingPipeline().run(
            [
                PropagationPath(direct_delay_s, 0.10 + 0.0j, "weak_first"),
                PropagationPath(
                    direct_delay_s + 10.0e-9,
                    1.0 + 0.0j,
                    "strong_late",
                ),
            ],
            np.arange(2.402e9, 2.481e9, 1.0e6),
            np.linspace(3.5e9, 4.5e9, 1001),
            rfic_config=RFICImpairmentConfig(snr_db=5.0),
            seed=7,
        )

        self.assertEqual(result.uwb_frame_count, 4)
        self.assertIsNotNone(result.uwb_measured_frames)
        self.assertEqual(result.uwb_measured_frames.shape, (4, 1001))
        self.assertGreater(result.uwb_frame_coherence, 0.75)
        self.assertLess(abs(result.uwb_distance_m - true_distance_m), 0.10)
        self.assertEqual(result.uwb_first_path.status.value, "LOW_CONFIDENCE")

    def test_low_frame_coherence_cannot_return_high_confidence_pass(self) -> None:
        result = RangingPipeline().run(
            [PropagationPath(5.0 / C0, 1.0 + 0.0j, "LOS")],
            np.arange(2.402e9, 2.481e9, 1.0e6),
            np.linspace(3.5e9, 4.5e9, 1001),
            rfic_config=RFICImpairmentConfig(
                phase_noise_std_deg=60.0,
                snr_db=40.0,
            ),
            seed=0,
        )

        self.assertLess(result.uwb_frame_coherence, 0.75)
        self.assertEqual(result.uwb_first_path.status.value, "LOW_CONFIDENCE")
        self.assertEqual(
            result.uwb_first_path.rejection_reason,
            "LOW_FRAME_COHERENCE",
        )
        self.assertLessEqual(result.uwb_first_path.confidence, 0.25)

    def test_single_frame_early_path_enhancement_prevents_false_pass(self) -> None:
        true_distance_m = 5.0
        direct_delay_s = true_distance_m / C0
        common = dict(
            paths=[
                PropagationPath(direct_delay_s, 0.10 + 0.0j, "weak_first"),
                PropagationPath(
                    direct_delay_s + 10.0e-9,
                    1.0 + 0.0j,
                    "strong_late",
                ),
            ],
            ble_frequencies_hz=np.arange(2.402e9, 2.481e9, 1.0e6),
            uwb_frequencies_hz=np.linspace(3.5e9, 4.5e9, 1001),
            rfic_config=RFICImpairmentConfig(snr_db=5.0),
            seed=7,
            uwb_frame_count=1,
        )
        legacy = RangingPipeline().run(**common)
        guarded = RangingPipeline().run(
            **common,
            uwb_early_path_enhancement=True,
        )

        self.assertGreater(abs(legacy.uwb_distance_m - true_distance_m), 1.0)
        self.assertEqual(legacy.uwb_first_path.status.value, "PASS")
        self.assertGreater(abs(guarded.uwb_distance_m - true_distance_m), 1.0)
        self.assertEqual(guarded.uwb_first_path.status.value, "LOW_CONFIDENCE")
        self.assertEqual(
            guarded.uwb_first_path.rejection_reason,
            "POSSIBLE_EARLY_PATH_BELOW_RELIABLE_THRESHOLD",
        )

    def test_pipeline_group_delay_calibration_keeps_raw_evidence(self) -> None:
        true_distance = 6.0
        group_delay_s = 1.2e-9
        paths = [
            PropagationPath(true_distance / C0, 1.0 + 0.0j, "LOS")
        ]
        ble = np.arange(2.402e9, 2.481e9, 1.0e6)
        uwb = np.linspace(3.5e9, 4.5e9, 1001)
        result = RangingPipeline().run(
            paths,
            ble,
            uwb,
            rfic_config=RFICImpairmentConfig(
                base_group_delay_s=group_delay_s
            ),
            seed=5,
            calibrate_rfic_group_delay=True,
            rfic_group_delay_calibration_s=group_delay_s,
        )
        expected_raw = true_distance + C0 * group_delay_s
        self.assertAlmostEqual(
            result.ble_raw_distance_m, expected_raw, places=8
        )
        self.assertAlmostEqual(
            result.ble_calibrated_distance_m, true_distance, places=8
        )
        self.assertLess(
            abs(result.uwb_raw_distance_m - expected_raw), 0.04
        )
        self.assertLess(
            abs(result.uwb_distance_m - true_distance), 0.04
        )
        self.assertEqual(
            result.rfic_group_delay_calibration_s, group_delay_s
        )

    def test_scalar_calibration_is_revalidated_after_ranging(self) -> None:
        ble = np.arange(2.402e9, 2.481e9, 1.0e6)
        uwb = np.linspace(3.5e9, 4.5e9, 1001)
        result = RangingPipeline().run(
            [PropagationPath(0.5 / C0, 1.0 + 0.0j, "LOS")],
            ble,
            uwb,
            calibrate_rfic_group_delay=True,
            rfic_group_delay_calibration_s=20.0e-9,
        )

        self.assertFalse(result.ble_robust.availability)
        self.assertEqual(result.ble_robust.status, "rejected")
        self.assertEqual(
            result.ble_robust.rejection_reason,
            "calibrated_distance_out_of_range",
        )
        self.assertEqual(result.ble_robust.confidence, 0.0)
        self.assertLess(result.ble_robust_calibrated_distance_m, 0.0)
        self.assertTrue(np.isnan(result.corrected_ble_distance_m))
        self.assertLess(abs(result.uwb_raw_distance_m - 0.5), 0.04)
        self.assertTrue(np.isnan(result.uwb_distance_m))
        self.assertEqual(result.uwb_first_path.status.value, "NO_PATH")
        self.assertEqual(
            result.uwb_first_path.rejection_reason,
            "CALIBRATION_EXCEEDS_DETECTED_DELAY",
        )

    def test_round_trip_robust_candidate_uses_path_factor_for_calibration(self) -> None:
        true_distance_m = 6.0
        group_delay_s = 1.2e-9
        result = RangingPipeline().run(
            [PropagationPath(2.0 * true_distance_m / C0, 1.0 + 0.0j)],
            np.arange(2.402e9, 2.481e9, 1.0e6),
            np.linspace(3.5e9, 4.5e9, 1001),
            path_factor=2.0,
            rfic_config=RFICImpairmentConfig(
                base_group_delay_s=group_delay_s
            ),
            calibrate_rfic_group_delay=True,
            rfic_group_delay_calibration_s=group_delay_s,
        )

        self.assertTrue(result.ble_robust.availability)
        self.assertAlmostEqual(
            result.ble_robust_calibrated_distance_m,
            true_distance_m,
            places=8,
        )

    def test_pipeline_does_not_use_injected_group_delay_as_oracle(self) -> None:
        true_distance = 4.0
        injected_delay_s = 1.0e-9
        paths = [PropagationPath(true_distance / C0, 1.0 + 0.0j)]
        ble = np.arange(2.402e9, 2.481e9, 1.0e6)
        uwb = np.linspace(3.5e9, 4.5e9, 1001)
        result = RangingPipeline().run(
            paths,
            ble,
            uwb,
            rfic_config=RFICImpairmentConfig(
                base_group_delay_s=injected_delay_s
            ),
            seed=2,
        )
        expected_raw = true_distance + C0 * injected_delay_s
        self.assertAlmostEqual(
            result.ble_calibrated_distance_m, expected_raw, places=8
        )
        self.assertEqual(result.rfic_group_delay_calibration_s, 0.0)
        with self.assertRaises(ValueError):
            RangingPipeline().run(
                paths,
                ble,
                uwb,
                rfic_config=RFICImpairmentConfig(
                    base_group_delay_s=injected_delay_s
                ),
                calibrate_rfic_group_delay=True,
            )

    def test_noise_is_reproducible_through_pipeline_seed(self) -> None:
        distance = 3.0
        paths = [PropagationPath(distance / C0, 1.0 + 0.0j)]
        ble = np.arange(2.402e9, 2.481e9, 2.0e6)
        uwb = np.linspace(3.5e9, 4.5e9, 501)
        cfg = RFICImpairmentConfig(snr_db=20.0, phase_noise_std_deg=1.0)
        first = RangingPipeline().run(paths, ble, uwb, rfic_config=cfg, seed=123)
        second = RangingPipeline().run(paths, ble, uwb, rfic_config=cfg, seed=123)
        np.testing.assert_array_equal(
            first.ble_measured_response, second.ble_measured_response
        )
        np.testing.assert_array_equal(
            first.uwb_measured_response, second.uwb_measured_response
        )

    def test_pipeline_integrates_frequency_dependent_rfic_profiles(self) -> None:
        true_distance_m = 7.0
        paths = [PropagationPath(true_distance_m / C0, 1.0 + 0.0j)]
        ble = np.arange(2.402e9, 2.481e9, 1.0e6)
        uwb = np.linspace(3.5e9, 4.5e9, 1001)
        config = RFICImpairmentConfig(
            base_group_delay_s=1.8e-9,
            group_delay_ripple_s=0.2e-9,
            group_delay_ripple_period_hz=40.0e6,
            amplitude_ripple_db=0.4,
        )
        ble_transfer = apply_rfic_impairments(
            ble, np.ones(ble.size, dtype=complex), config
        )
        uwb_transfer = apply_rfic_impairments(
            uwb, np.ones(uwb.size, dtype=complex), config
        )
        ble_profile = build_calibration_profile(
            ble,
            ble_transfer,
            reference_plane="BLE antenna feed",
            source="independent synthetic BLE sweep",
        )
        uwb_profile = build_calibration_profile(
            uwb,
            uwb_transfer,
            reference_plane="UWB antenna feed",
            source="independent synthetic UWB sweep",
        )
        result = RangingPipeline().run(
            paths,
            ble,
            uwb,
            rfic_config=config,
            ble_rfic_calibration_profile=ble_profile,
            uwb_rfic_calibration_profile=uwb_profile,
        )

        self.assertEqual(result.rfic_calibration_mode, "frequency_profile")
        self.assertIsNotNone(result.ble_calibration_result)
        self.assertIsNotNone(result.uwb_calibration_result)
        self.assertGreater(
            abs(result.ble_raw_distance_m - true_distance_m), 0.2
        )
        self.assertLess(
            abs(result.ble_calibrated_distance_m - true_distance_m), 1.0e-8
        )
        self.assertLess(abs(result.uwb_distance_m - true_distance_m), 0.04)
        self.assertTrue(result.ble_robust.availability)
        np.testing.assert_array_equal(
            result.ble_calibration_result.raw_response,
            result.ble_measured_response,
        )

        with self.assertRaisesRegex(ValueError, "cannot be applied together"):
            RangingPipeline().run(
                paths,
                ble,
                uwb,
                rfic_config=config,
                calibrate_rfic_group_delay=True,
                rfic_group_delay_calibration_s=1.8e-9,
                ble_rfic_calibration_profile=ble_profile,
            )

    def test_pipeline_surfaces_ble_rejection_without_hiding_candidate(self) -> None:
        direct_delay_s = 10.0 / C0
        paths = [
            PropagationPath(direct_delay_s, 1.0 + 0.0j, "first"),
            PropagationPath(
                direct_delay_s + 20.0e-9,
                np.exp(0.7j),
                "competing",
            ),
        ]
        ble = np.arange(2.402e9, 2.481e9, 1.0e6)
        uwb = np.linspace(3.5e9, 4.5e9, 1001)
        result = RangingPipeline().run(paths, ble, uwb)

        self.assertFalse(result.ble_robust.availability)
        self.assertEqual(result.ble_robust.status, "rejected")
        self.assertIn(
            result.ble_robust.rejection_reason,
            {"ambiguous_delay_peaks", "phase_residual_too_high"},
        )
        self.assertTrue(np.isfinite(result.ble_calibrated_distance_m))


if __name__ == "__main__":
    unittest.main()
