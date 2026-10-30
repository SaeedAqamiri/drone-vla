import unittest

from sat2aerial.sizing import SizingInput, compute


class SizingTest(unittest.TestCase):
    def test_defaults(self):
        result = compute(SizingInput())
        width, height = result["drone"]["footprint_m"]
        self.assertGreater(width, 280)
        self.assertLess(width, 310)
        self.assertGreater(result["mission"]["hours"], 27)
        self.assertLess(result["mission"]["hours"], 29)
        self.assertGreater(result["satellite"]["tiles"], 1000)
        self.assertGreater(result["generation"]["one_gpu_days"], 1)

    def test_gsd_scaling(self):
        fine = compute(SizingInput(satellite_gsd_m=0.5))
        coarse = compute(SizingInput(satellite_gsd_m=1.0))
        self.assertGreater(fine["satellite"]["tiles"], coarse["satellite"]["tiles"] * 3)


if __name__ == "__main__":
    unittest.main()
