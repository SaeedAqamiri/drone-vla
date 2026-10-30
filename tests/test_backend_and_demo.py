import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sat2aerial.backends import make_backend
from sat2aerial.dataset import load_pairs
from sat2aerial.synthetic import generate_demo, make_satellite


class BackendAndDemoTest(unittest.TestCase):
    def test_dummy_backend(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sat_path = root / "sat.png"
            cv2.imwrite(str(sat_path), make_satellite(rng, size=192))
            backend = make_backend("dummy", seed=0)
            out = root / "out.png"
            meta = backend.generate(sat_path, "prompt", out, (128, 128), 1)
            self.assertTrue(out.exists())
            self.assertEqual(meta["backend"], "dummy")
            image = cv2.imread(str(out))
            self.assertEqual(image.shape[:2], (128, 128))

    def test_demo_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = generate_demo(Path(tmp) / "demo", sites=2, size=192, seed=0)
            pairs = load_pairs(root)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(len(pairs[0].aerials), 2)


if __name__ == "__main__":
    unittest.main()
