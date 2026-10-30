import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sat2aerial.matching import match_images
from sat2aerial.metrics import aligned_edge_iou, ssim
from sat2aerial.synthetic import make_aerial, make_satellite


class MatchingTest(unittest.TestCase):
    def test_synthetic_pair_matches(self):
        rng = np.random.default_rng(3)
        satellite = make_satellite(rng, size=384)
        aerial = make_aerial(satellite, rng)
        with tempfile.TemporaryDirectory() as tmp:
            sat_path = Path(tmp) / "sat.png"
            aerial_path = Path(tmp) / "aerial.png"
            cv2.imwrite(str(sat_path), satellite)
            cv2.imwrite(str(aerial_path), aerial)
            match, homography = match_images(sat_path, aerial_path, mode="edges")
            self.assertTrue(match.success)
            self.assertGreater(match.inliers, 20)
            self.assertIsNotNone(homography)
            self.assertAlmostEqual(match.scale or 0.0, 1.0, delta=0.15)
            iou = aligned_edge_iou(aerial_path, sat_path, homography)
            self.assertIsNotNone(iou)
            self.assertGreater(iou, 0.2)

    def test_identical_metrics(self):
        rng = np.random.default_rng(5)
        image = make_satellite(rng, size=256)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            cv2.imwrite(str(path), image)
            self.assertAlmostEqual(ssim(path, path), 1.0, places=6)
            match, homography = match_images(path, path, mode="gray")
            self.assertTrue(match.success)
            self.assertIsNotNone(homography)
            iou = aligned_edge_iou(path, path, np.eye(3))
            self.assertGreater(iou, 0.95)


if __name__ == "__main__":
    unittest.main()
