from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rf_ranging_model.ble.channel_sounding import allowed_cs_frequencies_hz
from rf_ranging_model.ble.pbr import estimate_distance
from rf_ranging_model.constants import C0
from rf_ranging_model.rfic.calibration import (
    CalibrationResult,
    apply_calibration,
    build_calibration_profile,
    load_s2p_calibration_profile,
)
from rf_ranging_model.uwb.cir import frequency_response_to_cir
from rf_ranging_model.uwb.tof import estimate_distance_from_cir


def _polynomial_front_end(
    frequencies_hz: np.ndarray,
    *,
    center_hz: float,
    scale_hz: float,
    constant_delay_s: float,
) -> np.ndarray:
    normalised = (frequencies_hz - center_hz) / scale_hz
    phase = (
        0.43
        - 2.0 * np.pi * constant_delay_s * (frequencies_hz - center_hz)
        + 0.31 * normalised**2
    )
    magnitude = 10.0 ** (
        (0.28 * np.cos(np.pi * normalised) - 0.06 * normalised) / 20.0
    )
    return np.asarray(magnitude * np.exp(1j * phase), dtype=np.complex128)


def _sinusoidal_front_end(
    frequencies_hz: np.ndarray,
    *,
    center_hz: float,
    constant_delay_s: float,
    ripple_delay_s: float,
    ripple_period_hz: float,
) -> np.ndarray:
    delta = frequencies_hz - center_hz
    phase = (
        -0.27
        - 2.0 * np.pi * constant_delay_s * delta
        - ripple_delay_s
        * ripple_period_hz
        * np.sin(2.0 * np.pi * delta / ripple_period_hz)
    )
    magnitude = 10.0 ** (
        0.35 * np.cos(2.0 * np.pi * delta / 125.0e6) / 20.0
    )
    return np.asarray(magnitude * np.exp(1j * phase), dtype=np.complex128)


def _uwb_distance(
    frequencies_hz: np.ndarray,
    response: np.ndarray,
    *,
    path_factor: float,
) -> float:
    delays, cir = frequency_response_to_cir(
        frequencies_hz, response, zero_pad_factor=8
    )
    distance, _ = estimate_distance_from_cir(
        delays, cir, path_factor=path_factor
    )
    return distance


class RFICFrequencyCalibrationTests(unittest.TestCase):
    def test_polynomial_profile_recovers_group_delay_and_metadata(self) -> None:
        frequencies = np.linspace(2.395e9, 2.485e9, 901)
        center = float(np.mean(frequencies))
        scale = float(np.ptp(frequencies) / 2.0)
        constant_delay = 2.4e-9
        transfer = _polynomial_front_end(
            frequencies,
            center_hz=center,
            scale_hz=scale,
            constant_delay_s=constant_delay,
        )
        profile = build_calibration_profile(
            frequencies,
            transfer,
            reference_plane="antenna connector",
            source="synthetic polynomial VNA sweep",
            metadata={"fixture": "TRL-01", "temperature_c": 25.0},
            phase_model="polynomial",
            polynomial_order=2,
        )

        self.assertAlmostEqual(
            profile.constant_group_delay_s, constant_delay, places=15
        )
        self.assertLess(profile.phase_fit_rms_rad, 1.0e-12)
        self.assertGreater(
            np.ptp(profile.frequency_dependent_group_delay_s), 1.0e-9
        )
        self.assertEqual(profile.reference_plane, "antenna connector")
        self.assertEqual(profile.source, "synthetic polynomial VNA sweep")
        self.assertEqual(profile.metadata["fixture"], "TRL-01")
        self.assertFalse(profile.frequencies_hz.flags.writeable)
        self.assertFalse(profile.complex_correction.flags.writeable)
        self.assertEqual(profile.summary()["profile_id"], profile.profile_id)

    def test_ble_polynomial_calibration_removes_bias_for_both_path_factors(
        self,
    ) -> None:
        calibration_frequencies = np.linspace(2.395e9, 2.485e9, 901)
        center = float(np.mean(calibration_frequencies))
        scale = float(np.ptp(calibration_frequencies) / 2.0)
        constant_delay = 2.4e-9
        calibration_transfer = _polynomial_front_end(
            calibration_frequencies,
            center_hz=center,
            scale_hz=scale,
            constant_delay_s=constant_delay,
        )
        profile = build_calibration_profile(
            calibration_frequencies,
            calibration_transfer,
            reference_plane="BLE antenna feed",
            source="synthetic polynomial calibration",
            phase_model="polynomial",
            polynomial_order=2,
        )
        frequencies = allowed_cs_frequencies_hz()
        front_end = _polynomial_front_end(
            frequencies,
            center_hz=center,
            scale_hz=scale,
            constant_delay_s=constant_delay,
        )
        true_distance_m = 18.25

        for path_factor in (1.0, 2.0):
            with self.subTest(path_factor=path_factor):
                propagation_delay = path_factor * true_distance_m / C0
                channel = np.exp(
                    -1j * 2.0 * np.pi * frequencies * propagation_delay
                )
                raw = channel * front_end
                result = apply_calibration(frequencies, raw, profile)
                raw_distance = estimate_distance(
                    frequencies, raw, path_factor=path_factor
                ).distance_m
                calibrated_distance = estimate_distance(
                    frequencies,
                    result.calibrated_response,
                    path_factor=path_factor,
                ).distance_m

                self.assertGreater(abs(raw_distance - true_distance_m), 0.25)
                self.assertLess(
                    abs(calibrated_distance - true_distance_m), 1.0e-8
                )
                np.testing.assert_allclose(
                    result.raw_response, raw, rtol=0.0, atol=0.0
                )
                np.testing.assert_allclose(
                    result.calibrated_response, channel, rtol=1.0e-12, atol=1.0e-12
                )
                self.assertEqual(result.profile_id, profile.profile_id)
                self.assertEqual(
                    result.evidence()["reference_plane"], "BLE antenna feed"
                )

    def test_uwb_direct_sinusoidal_calibration_meets_grid_accuracy(
        self,
    ) -> None:
        calibration_frequencies = np.linspace(3.49e9, 4.51e9, 2041)
        center = 4.0e9
        constant_delay = 1.8e-9
        ripple_delay = 0.32e-9
        ripple_period = 80.0e6
        calibration_transfer = _sinusoidal_front_end(
            calibration_frequencies,
            center_hz=center,
            constant_delay_s=constant_delay,
            ripple_delay_s=ripple_delay,
            ripple_period_hz=ripple_period,
        )
        profile = build_calibration_profile(
            calibration_frequencies,
            calibration_transfer,
            reference_plane="UWB antenna connector",
            source="synthetic sinusoidal S21",
            phase_model="direct",
        )
        frequencies = np.linspace(3.5e9, 4.5e9, 1001)
        front_end = _sinusoidal_front_end(
            frequencies,
            center_hz=center,
            constant_delay_s=constant_delay,
            ripple_delay_s=ripple_delay,
            ripple_period_hz=ripple_period,
        )
        true_distance_m = 8.0

        self.assertAlmostEqual(
            profile.constant_group_delay_s, constant_delay, delta=2.0e-12
        )
        self.assertAlmostEqual(
            float(np.max(profile.frequency_dependent_group_delay_s)),
            ripple_delay,
            delta=2.0e-12,
        )
        for path_factor in (1.0, 2.0):
            with self.subTest(path_factor=path_factor):
                propagation_delay = path_factor * true_distance_m / C0
                channel = np.exp(
                    -1j * 2.0 * np.pi * frequencies * propagation_delay
                )
                raw = channel * front_end
                result = apply_calibration(frequencies, raw, profile)
                raw_distance = _uwb_distance(
                    frequencies, raw, path_factor=path_factor
                )
                calibrated_distance = _uwb_distance(
                    frequencies,
                    result.calibrated_response,
                    path_factor=path_factor,
                )

                self.assertGreater(abs(raw_distance - true_distance_m), 0.20)
                self.assertLess(abs(calibrated_distance - true_distance_m), 0.04)
                np.testing.assert_allclose(
                    result.calibrated_response, channel, rtol=2.0e-12, atol=2.0e-12
                )

    def test_log_magnitude_and_unwrapped_phase_interpolate_off_grid(self) -> None:
        calibration_frequencies = np.linspace(2.4e9, 2.5e9, 11)
        center = 2.45e9
        transfer = np.exp(
            1.5e-9 * (calibration_frequencies - center)
            + 1j
            * (
                0.2
                - 2.0
                * np.pi
                * 1.3e-9
                * (calibration_frequencies - center)
            )
        )
        profile = build_calibration_profile(
            calibration_frequencies,
            transfer,
            reference_plane="interpolation plane",
            source="off-grid analytic vector",
        )
        target = np.linspace(2.405e9, 2.495e9, 10)
        target_transfer = np.exp(
            1.5e-9 * (target - center)
            + 1j * (0.2 - 2.0 * np.pi * 1.3e-9 * (target - center))
        )
        channel = np.exp(-1j * 2.0 * np.pi * target * 7.0e-9)
        result = apply_calibration(target, channel * target_transfer, profile)

        np.testing.assert_allclose(
            result.calibrated_response, channel, rtol=2.0e-14, atol=2.0e-14
        )

    def test_rejects_extrapolation_repeat_application_and_non_finite_input(
        self,
    ) -> None:
        frequencies = np.linspace(2.4e9, 2.5e9, 101)
        transfer = np.exp(-1j * 2.0 * np.pi * frequencies * 1.0e-9)
        profile = build_calibration_profile(
            frequencies,
            transfer,
            reference_plane="test plane",
            source="synthetic safety vector",
        )
        target = np.linspace(2.41e9, 2.49e9, 17)
        first = apply_calibration(target, np.ones(target.size, dtype=complex), profile)
        self.assertIsInstance(first, CalibrationResult)

        with self.assertRaisesRegex(ValueError, "already been applied"):
            apply_calibration(target, first, profile)
        with self.assertRaisesRegex(ValueError, "outside"):
            apply_calibration(
                np.linspace(2.39e9, 2.49e9, 17),
                np.ones(17, dtype=complex),
                profile,
            )
        non_finite_response = np.ones(target.size, dtype=complex)
        non_finite_response[4] = complex(np.nan, 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_calibration(target, non_finite_response, profile)
        bad_transfer = transfer.copy()
        bad_transfer[10] = 0.0
        with self.assertRaisesRegex(ValueError, "null too deep"):
            build_calibration_profile(
                frequencies,
                bad_transfer,
                reference_plane="test plane",
                source="invalid null vector",
            )

    def test_loads_s21_profile_with_reference_plane_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.s2p"
            rows = ["! calibration fixture", "# GHz S RI R 50"]
            for index, frequency_ghz in enumerate(
                (2.40, 2.42, 2.44, 2.46, 2.48)
            ):
                phase = -0.08 * index
                s21 = np.exp(1j * phase)
                rows.append(
                    f"{frequency_ghz:.2f} 0 0 {s21.real:.16g} "
                    f"{s21.imag:.16g} 0 0 0 0"
                )
            path.write_text("\n".join(rows), encoding="utf-8")
            profile = load_s2p_calibration_profile(
                path,
                reference_plane="fixture output",
                source="VNA run 2026-08-22",
                metadata={"operator": "acceptance"},
            )

        self.assertEqual(profile.frequencies_hz.size, 5)
        self.assertEqual(profile.reference_plane, "fixture output")
        self.assertEqual(profile.source, "VNA run 2026-08-22")
        self.assertEqual(profile.metadata["operator"], "acceptance")
        np.testing.assert_allclose(
            profile.measured_transfer * profile.complex_correction,
            np.ones(5, dtype=complex),
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_rejects_phase_unwrap_ambiguity_for_declared_delay_bound(self) -> None:
        frequencies = np.linspace(2.4e9, 2.5e9, 6)
        transfer = np.exp(-1j * 2.0 * np.pi * frequencies * 40.0e-9)
        with self.assertRaisesRegex(ValueError, "spacing is ambiguous"):
            build_calibration_profile(
                frequencies,
                transfer,
                reference_plane="coarse sweep",
                source="phase ambiguity safety vector",
                maximum_expected_group_delay_s=40.0e-9,
            )

    def test_rejects_large_unmeasured_gap_and_excessive_inverse_gain(self) -> None:
        frequencies = np.r_[
            np.linspace(2.40e9, 2.41e9, 11),
            np.linspace(2.48e9, 2.49e9, 11),
        ]
        with self.assertRaisesRegex(ValueError, "unmeasured gap"):
            build_calibration_profile(
                frequencies,
                np.ones(frequencies.size, dtype=complex),
                reference_plane="gapped sweep",
                source="gap safety vector",
            )

        dense = np.linspace(2.4e9, 2.5e9, 101)
        weak_transfer = np.ones(dense.size, dtype=complex)
        weak_transfer[50] = 1.0e-4
        with self.assertRaisesRegex(ValueError, "correction gain"):
            build_calibration_profile(
                dense,
                weak_transfer,
                reference_plane="deep notch",
                source="inverse gain safety vector",
                maximum_correction_magnitude=1.0e3,
            )


if __name__ == "__main__":
    unittest.main()
