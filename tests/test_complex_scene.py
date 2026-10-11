"""Independent contracts for the V7 effective complex-scene extension."""
import unittest

import numpy as np

from rf_ranging_model.channel.complex_scene import ComplexScene, MovingPanel, bilateral_product
from rf_ranging_model.channel.spatial import AttenuatingPanel, PointScatterer, SpatialAntenna, SpatialScene


class ComplexSceneTests(unittest.TestCase):
    def test_invalid_bounce_indices_and_scale_rejected(self):
        scene = SpatialScene((0, 0, 1), (PointScatterer((1, 2, 1), .5),))
        for pairs in (((0, 0),), ((0, 1),), ((-1, 0),)):
            with self.assertRaises(ValueError):
                ComplexScene(scene, bounce_pairs=pairs)
        for scale in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                ComplexScene(scene, second_bounce_scale=scale)

    def test_antenna_offsets_share_the_same_two_bounce_geometry(self):
        scene = ComplexScene(SpatialScene((0, 0, 1),
            (PointScatterer((1, 2, 1), .8), PointScatterer((3, 2, 1), .7))), bounce_pairs=((0, 1),))
        first = {path.label: path for path in scene.paths((4, 0, 1), SpatialAntenna((0, 0, 0)))}
        second = {path.label: path for path in scene.paths((4, 0, 1), SpatialAntenna((0, .06, 0)))}
        self.assertIn("double_0_1", first)
        for label in first:
            difference = abs(first[label].delay_s - second[label].delay_s) * 299792458
            self.assertLessEqual(difference, .060000001)

    def test_moving_panel_rejects_invalid_time_and_velocity(self):
        panel = AttenuatingPanel((0, 0, 0), (1, 1, 1), 10)
        with self.assertRaises(ValueError):
            MovingPanel(panel, (np.inf, 0, 0))
        for elapsed in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                MovingPanel(panel).at(elapsed)

    def test_bilateral_masks_exclude_nonfinite_or_one_sided_samples(self):
        values = np.array([1 + 1j, np.nan, 3 + 2j])
        output, mask = bilateral_product(values, np.ones(3, complex), np.ones(3, bool), np.array([True, True, False]))
        np.testing.assert_array_equal(mask, [True, False, False])
        np.testing.assert_array_equal(output, [1 + 1j, 0, 0])
        with self.assertRaises(ValueError):
            bilateral_product(values, np.ones(2), np.ones(3, bool), np.ones(3, bool))
        with self.assertRaises(ValueError):
            bilateral_product(values, values, np.ones(3, int), np.ones(3, bool))


if __name__ == "__main__":
    unittest.main()
