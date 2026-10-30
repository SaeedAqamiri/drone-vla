import unittest

from navlab.fusion import Fusion2D


class FusionTest(unittest.TestCase):
    def test_initial_update(self):
        fusion = Fusion2D()
        result = fusion.update(1.0, 2.0, 10.0, sigma=1.5)
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["north"], 1.0)
        self.assertAlmostEqual(result["east"], 2.0)

    def test_predict_and_outlier(self):
        fusion = Fusion2D(gate=9.21)
        fusion.update(0.0, 0.0, 10.0)
        fusion.update(1.0, 0.0, 11.0)
        predicted = fusion.predict_only(12.0)
        self.assertIsNotNone(predicted)
        self.assertGreater(predicted["north"], 1.0)
        rejected = fusion.update(100.0, 100.0, 13.0)
        self.assertFalse(rejected["accepted"])
        self.assertGreater(fusion.rejected, 0)

    def test_uninitialized_predict(self):
        self.assertIsNone(Fusion2D().predict_only(1.0))


if __name__ == "__main__":
    unittest.main()
