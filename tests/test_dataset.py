import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sat2aerial.dataset import load_pairs, pick_style_refs


def _write(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), value, np.uint8)
    cv2.imwrite(str(path), image)


class DatasetTest(unittest.TestCase):
    def test_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "alpha" / "satellite.png", 10)
            _write(root / "alpha" / "aerial_00.png", 20)
            _write(root / "alpha" / "aerial_01.png", 30)
            _write(root / "beta" / "sat.png", 40)
            _write(root / "beta" / "drone.png", 50)
            pairs = load_pairs(root)
            self.assertEqual([pair.site_id for pair in pairs], ["alpha", "beta"])
            self.assertEqual(len(pairs[0].aerials), 2)
            self.assertEqual(len(pairs[1].aerials), 1)
            refs = pick_style_refs(pairs, "alpha", 2)
            self.assertEqual(len(refs), 2)
            self.assertTrue(all("beta" in str(path) for path in refs))

    def test_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "images" / "sat1.png", 10)
            _write(root / "images" / "air1.png", 20)
            (root / "manifest.csv").write_text(
                "site,satellite,aerial\none,images/sat1.png,images/air1.png\n",
                encoding="utf-8",
            )
            pairs = load_pairs(root)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].site_id, "one")
            self.assertEqual(pairs[0].aerials, [root / "images" / "air1.png"])

    def test_missing_pairs_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_pairs(Path(tmp))


if __name__ == "__main__":
    unittest.main()
