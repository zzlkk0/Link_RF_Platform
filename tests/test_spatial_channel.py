"""Independent geometry and occlusion checks for antenna diversity."""
import unittest

import numpy as np

from rf_ranging_model.constants import C0
from rf_ranging_model.channel.multipath import frequency_response
from rf_ranging_model.channel.spatial import AttenuatingPanel, PointScatterer, SpatialAntenna, SpatialScene


class SpatialChannelTests(unittest.TestCase):
    def test_two_antennas_share_scatterers_but_have_distinct_cfr(self):
        scene = SpatialScene((0, 0, 0), (PointScatterer((2, 2, 0), .8 + .2j),), amplitude_reference_m=5)
        antennas = (SpatialAntenna((0, 0, 0)), SpatialAntenna((0, .06, 0)))
        first, second = scene.antenna_paths((5, 0, 0), antennas)
        for index, paths in enumerate((first, second)):
            receiver = np.array([5, .06 * index, 0])
            self.assertAlmostEqual(paths[0].delay_s * C0, np.linalg.norm(receiver))
            self.assertAlmostEqual(paths[1].delay_s * C0, np.sqrt(8) + np.linalg.norm(receiver - [2, 2, 0]))
        self.assertLessEqual(abs(first[1].delay_s - second[1].delay_s) * C0, .06 + 1e-10)
        frequencies = np.linspace(2.402e9, 2.48e9, 72)
        self.assertGreater(np.max(abs(frequency_response(frequencies, first) - frequency_response(frequencies, second))), .1)

    def test_coincident_antennas_have_identical_noiseless_paths(self):
        scene = SpatialScene((0, 0, 0), (PointScatterer((1, 2, 0), .6j),))
        antenna = SpatialAntenna((0, 0, 0))
        self.assertEqual(*scene.antenna_paths((5, 0, 0), (antenna, antenna)))

    def test_opaque_panel_removes_direct_not_bypassing_path(self):
        panel = AttenuatingPanel((2, -.1, -.5), (2.1, .1, .5), opaque=True)
        scene = SpatialScene((0, 0, 0), (PointScatterer((2, 2, 0), 1),), (panel,))
        paths = scene.paths((5, 0, 0), SpatialAntenna((0, 0, 0)))
        self.assertEqual([path.label for path in paths], ["scatterer_0"])
        self.assertGreater(paths[0].delay_s * C0, 5)

    def test_partial_panel_uses_amplitude_not_power_db(self):
        panel = AttenuatingPanel((2, -.1, -.5), (2.1, .1, .5), attenuation_db=20)
        scene = SpatialScene((0, 0, 0), panels=(panel,), amplitude_reference_m=5)
        self.assertAlmostEqual(abs(scene.paths((5, 0, 0), SpatialAntenna((0, 0, 0)))[0].complex_gain), .1)

    def test_panel_edge_can_block_only_one_antenna(self):
        panel = AttenuatingPanel((2, -.1, -.5), (2.1, .01, .5), opaque=True)
        scene = SpatialScene((0, 0, 0), panels=(panel,))
        first, second = scene.antenna_paths((5, 0, 0), (SpatialAntenna((0, 0, 0)), SpatialAntenna((0, .06, 0))))
        self.assertEqual(first, [])
        self.assertEqual(second[0].label, "LOS")

    def test_opposite_polarization_and_rotation_are_not_independent_scenes(self):
        antenna = SpatialAntenna((0, 0, 0), (0, 1, 0))
        scene = SpatialScene((0, 0, 0), (PointScatterer((2, 2, 0), 1, np.pi / 2),))
        paths = scene.paths((5, 0, 0), antenna)
        self.assertEqual([path.label for path in paths], ["scatterer_0"])

    def test_empty_blocked_channel_stays_empty(self):
        panel = AttenuatingPanel((2, -1, -1), (3, 1, 1), opaque=True)
        self.assertEqual(SpatialScene((0, 0, 0), panels=(panel,)).paths((5, 0, 0), SpatialAntenna((0, 0, 0))), [])

    def test_invalid_geometry_fails(self):
        with self.assertRaises(ValueError):
            SpatialAntenna((0, 0, 0), (0, 0, 0))
        with self.assertRaises(ValueError):
            AttenuatingPanel((0, 0, 0), (0, 1, 1))
        with self.assertRaises(ValueError):
            SpatialScene((0, 0, 0)).paths((0, 0, 0), SpatialAntenna((0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
