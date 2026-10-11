"""Independent weak-path and ambiguity contracts for masked sequence recovery."""
import unittest

import numpy as np

from rf_ranging_model.ble.channel_sounding import allowed_cs_frequencies_hz
from rf_ranging_model.ble.sequence_recovery import SequenceRecoveryConfig, recover_sequence
from rf_ranging_model.constants import C0


class SequenceRecoveryTest(unittest.TestCase):
    def observe(self, distances, gains, seed=412, noise=.001, missing=.3):
        frequencies = allowed_cs_frequencies_hz()
        rng = np.random.default_rng(seed)
        channel = np.exp(-2j * np.pi * frequencies[:, None] * np.asarray(distances)[None, :] / C0) @ np.asarray(gains)
        iq = channel[None, :] * np.exp(1j * rng.uniform(-np.pi, np.pi, (8, 1)))
        iq += noise * (rng.normal(size=iq.shape) + 1j * rng.normal(size=iq.shape))
        mask = rng.random(iq.shape) > missing
        return frequencies, np.where(mask, iq, 0), mask

    def test_weak_path_survives_missing_frequencies(self):
        result = recover_sequence(*self.observe([5, 10], [.12, 1]), config=SequenceRecoveryConfig(max_distance_m=20))
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.distance_m, 5, delta=.12)
        self.assertLess(result.amplitude_ratios[0], .2)
        self.assertGreaterEqual(result.frame_supports[0], .625)

    def test_close_paths_are_not_forced_to_strongest_peak(self):
        result = recover_sequence(*self.observe([5, 6.2], [.22, 1], noise=.0001, missing=.15), config=SequenceRecoveryConfig(max_distance_m=15))
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.distance_m, 5, delta=.25)

    def test_phase_offsets_do_not_change_distance(self):
        frequencies, iq, mask = self.observe([5, 10], [.15, 1])
        first = recover_sequence(frequencies, iq, mask, SequenceRecoveryConfig(max_distance_m=20))
        rotated = iq * np.exp(1j * np.arange(8)[:, None] * .714)
        second = recover_sequence(frequencies, rotated, mask, SequenceRecoveryConfig(max_distance_m=20))
        self.assertAlmostEqual(first.distance_m, second.distance_m, delta=.001)

    def test_single_reflection_is_not_geometric_distance(self):
        result = recover_sequence(*self.observe([7], [1]), config=SequenceRecoveryConfig(max_distance_m=20))
        self.assertAlmostEqual(result.distance_m, 7, delta=.03)
        self.assertIn("not_los", result.reason)

    def test_noise_only_does_not_create_reliable_path(self):
        result = recover_sequence(*self.observe([5], [0], noise=.1), config=SequenceRecoveryConfig(max_distance_m=20))
        self.assertFalse(result.available)

    def test_calibrated_receiver_ripple_is_not_an_early_path(self):
        frequencies, iq, mask = self.observe([25], [1], noise=.002)
        ripple = np.exp(.13 * np.cos(2 * np.pi * (frequencies - frequencies.mean()) / 20e6))
        result = recover_sequence(frequencies, iq * ripple, mask, SequenceRecoveryConfig(max_distance_m=45))
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.distance_m, 25, delta=.1)

    def test_bad_masks_and_nonfinite_values_fail(self):
        frequencies, iq, mask = self.observe([5], [1])
        with self.assertRaises(ValueError):
            recover_sequence(frequencies, iq, mask.astype(float))
        iq[0, 0] = np.nan
        with self.assertRaises(ValueError):
            recover_sequence(frequencies, iq, mask)


if __name__ == "__main__":
    unittest.main()
