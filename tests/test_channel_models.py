from __future__ import annotations

import unittest

import numpy as np

from rf_ranging_model.ble.pbr import estimate_distance
from rf_ranging_model.ble.channel_sounding import (
    allowed_cs_channel_indices,
    allowed_cs_frequencies_hz,
)
from rf_ranging_model.channel.free_space import free_space_response
from rf_ranging_model.channel.multipath import (
    delay_spread_rms,
    frequency_response,
    sampled_cir,
)
from rf_ranging_model.channel.stochastic_channel import generate_saleh_valenzuela
from rf_ranging_model.channel.two_ray import TwoRayGeometry, two_ray_response
from rf_ranging_model.constants import C0
from rf_ranging_model.types import PropagationPath


class ChannelModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frequencies = np.arange(2.402e9, 2.481e9, 1.0e6)

    def test_los_phase_slope_recovers_distance(self) -> None:
        true_distance = 7.25
        response = free_space_response(self.frequencies, true_distance)
        result = estimate_distance(self.frequencies, response)
        self.assertAlmostEqual(result.distance_m, true_distance, places=9)
        self.assertLess(result.residual_rms_rad, 1.0e-10)

    def test_official_cs_channel_map_and_long_range_los(self) -> None:
        frequencies = allowed_cs_frequencies_hz()
        golden_indices = np.r_[np.arange(2, 23), np.arange(26, 77)]
        np.testing.assert_array_equal(
            allowed_cs_channel_indices(), golden_indices
        )
        np.testing.assert_array_equal(
            frequencies, 2.402e9 + golden_indices * 1.0e6
        )
        self.assertEqual(frequencies[0], 2.404e9)
        self.assertEqual(frequencies[-1], 2.478e9)
        for true_distance in (0.5, 25.0, 50.0, 100.0, 149.0):
            with self.subTest(distance_m=true_distance):
                response = free_space_response(frequencies, true_distance)
                result = estimate_distance(frequencies, response)
                self.assertAlmostEqual(
                    result.distance_m, true_distance, places=8
                )
                self.assertEqual(result.estimator, "ambiguity_aware")

    def test_official_cs_channel_map_round_trip(self) -> None:
        frequencies = allowed_cs_frequencies_hz()
        for true_distance in (0.5, 10.0, 25.0, 50.0, 100.0):
            with self.subTest(distance_m=true_distance):
                round_trip_delay = 2.0 * true_distance / C0
                response = np.exp(
                    -1j * 2.0 * np.pi * frequencies * round_trip_delay
                )
                result = estimate_distance(
                    frequencies,
                    response,
                    path_factor=2.0,
                    max_distance_m=100.0,
                )
                self.assertAlmostEqual(
                    result.distance_m, true_distance, places=8
                )

    def test_tone_grid_unambiguous_range_is_enforced(self) -> None:
        frequencies = allowed_cs_frequencies_hz()
        response = np.ones_like(frequencies, dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "unambiguous range"):
            estimate_distance(
                frequencies,
                response,
                path_factor=2.0,
                max_distance_m=150.0,
            )
        even_frequencies = frequencies[frequencies % 2.0e6 == 0.0]
        with self.assertRaisesRegex(ValueError, "unambiguous range"):
            estimate_distance(
                even_frequencies,
                np.ones_like(even_frequencies, dtype=np.complex128),
                path_factor=2.0,
                max_distance_m=100.0,
            )
        true_distance_m = 100.0
        out_of_prior_response = np.exp(
            -1j
            * 2.0
            * np.pi
            * frequencies
            * (true_distance_m / C0)
        )
        with self.assertRaisesRegex(ValueError, "outside.*search prior"):
            estimate_distance(
                frequencies,
                out_of_prior_response,
                max_distance_m=50.0,
            )

    def test_two_ray_creates_fading_and_phase_nonlinearity(self) -> None:
        geometry = TwoRayGeometry(8.0, tx_height_m=1.3, rx_height_m=0.9)
        response = two_ray_response(
            self.frequencies, geometry, reflection_coefficient=-0.72 + 0.08j
        )
        amplitude_span_db = np.ptp(20.0 * np.log10(np.abs(response)))
        result = estimate_distance(
            self.frequencies, response, magnitude_weighting=False
        )
        self.assertGreater(amplitude_span_db, 0.2)
        self.assertGreater(result.residual_rms_rad, 1.0e-4)

    def test_multipath_cir_has_separate_impulses(self) -> None:
        paths = [
            PropagationPath(10.0e-9, 1.0 + 0.0j, "first"),
            PropagationPath(15.0e-9, 0.5 + 0.0j, "second"),
            PropagationPath(23.0e-9, 0.25j, "third"),
        ]
        times, cir = sampled_cir(paths, 1.0e9, duration_s=30.0e-9)
        peak_times = times[np.flatnonzero(np.abs(cir) > 0.1)]
        self.assertTrue(np.any(np.isclose(peak_times, 10.0e-9)))
        self.assertTrue(np.any(np.isclose(peak_times, 15.0e-9)))
        self.assertTrue(np.any(np.isclose(peak_times, 23.0e-9)))
        self.assertGreater(delay_spread_rms(paths), 0.0)

    def test_frequency_response_matches_manual_sum(self) -> None:
        paths = [
            PropagationPath(2.0e-9, 1.0 + 0.0j),
            PropagationPath(5.0e-9, 0.3 - 0.1j),
        ]
        response = frequency_response(self.frequencies, paths)
        manual = sum(
            path.complex_gain
            * np.exp(-1j * 2.0 * np.pi * self.frequencies * path.delay_s)
            for path in paths
        )
        np.testing.assert_allclose(response, manual, rtol=1.0e-13, atol=1.0e-13)

    def test_statistical_channel_is_seed_reproducible(self) -> None:
        first = generate_saleh_valenzuela(5.0, seed=42)
        second = generate_saleh_valenzuela(5.0, seed=42)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 2)
        self.assertAlmostEqual(first[0].delay_s, 5.0 / C0)


if __name__ == "__main__":
    unittest.main()
