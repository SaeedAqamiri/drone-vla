import unittest
import cv2
import numpy as np

from navlab.lab.ground_tracker import VisualGroundTracker
from navlab.lab.navigator import _image_to_local


class GroundTrackerTest(unittest.TestCase):
    def test_image_projection_center_across_quarter_turns(self):
        tile = {"tile_px": 512, "gsd": .5,
                "bounds": {"min_east": 100, "max_north": 200}}
        image_shape = (320, 512)
        for angle in (0, 90, 180, 270):
            matrix = _image_to_local(tile, np.eye(3), image_shape, angle, 512, .5)
            value = matrix @ np.array([image_shape[1]/2, image_shape[0]/2, 1])
            value = value[:2] / value[2]
            rotated_h, rotated_w = image_shape if angle in (0, 180) else image_shape[::-1]
            expected = (100 + rotated_w/2*.5, 200 - rotated_h/2*.5)
            np.testing.assert_allclose(value, expected, atol=.51)

    def test_tracks_translation_and_discards_unanchored_frame(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, -10], [0, 1, 4]]), (256, 256))
        tracker = VisualGroundTracker(max_dim=256, min_inliers=25)
        self.assertIsNone(tracker.predict(image, 0))
        tracker.finish(np.eye(3))
        delta = tracker.predict(shifted, 1)
        self.assertIsNotNone(delta, tracker.last_diagnostic)
        np.testing.assert_allclose(delta, (10, -4), atol=.4)
        tracker.finish(None)
        self.assertIsNotNone(tracker.previous)
        blank = np.zeros_like(image)
        self.assertIsNone(tracker.predict(blank, 2))
        tracker.finish(None)
        self.assertIsNone(tracker.previous)


if __name__ == "__main__":
    unittest.main()
