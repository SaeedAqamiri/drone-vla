import unittest

import cv2
import numpy as np

from sat2aerial.geometry import register_generated_to_source


class GeneratedGeometryTest(unittest.TestCase):
    def test_recovers_small_translation_and_rejects_unrelated_image(self):
        rng = np.random.default_rng(12)
        source = rng.integers(0, 256, (384, 384, 3), dtype=np.uint8)
        source = cv2.GaussianBlur(source, (3, 3), 0)
        moved = cv2.warpAffine(source, np.float32([[1, 0, 7], [0, 1, -5]]), (384, 384))
        aligned, diag = register_generated_to_source(source, moved)
        self.assertTrue(diag["accepted"], diag)
        self.assertIsNotNone(aligned)
        before = np.mean(np.abs(source[30:-30, 30:-30].astype(float)-moved[30:-30, 30:-30]))
        after = np.mean(np.abs(source[30:-30, 30:-30].astype(float)-aligned[30:-30, 30:-30]))
        self.assertLess(after, before * 0.2)
        unrelated = rng.integers(0, 256, source.shape, dtype=np.uint8)
        aligned, diag = register_generated_to_source(source, unrelated)
        self.assertIsNone(aligned)
        self.assertFalse(diag["accepted"])


if __name__ == "__main__":
    unittest.main()
