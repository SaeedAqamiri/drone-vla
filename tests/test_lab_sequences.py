import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from navlab.lab.sequences import _timestamp, uav_visloc_queries


class LabSequencesTest(unittest.TestCase):
    def test_compact_single_digit_hour_preserves_seconds(self):
        self.assertEqual(_timestamp("2018071250119 ", 1) - _timestamp("2018071250114 ", 0), 5)

    def test_uav_visloc_filter_sort_and_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_dir = root / "03"
            drone_dir = map_dir / "drone"
            drone_dir.mkdir(parents=True)
            rows = [
                {
                    "num": "3",
                    "filename": "03_0003.JPG",
                    "date": "2018-10-23T08:32:39",
                    "lat": "32.3",
                    "lon": "119.9",
                    "height": "466.0",
                    "Omega": "0.5",
                    "Kappa": "0.4",
                    "Phi1": "-40.0",
                    "Phi2": "-52.0",
                },
                {
                    "num": "1",
                    "filename": "03_0001.JPG",
                    "date": "2018-10-23T08:32:25",
                    "lat": "32.1",
                    "lon": "119.8",
                    "height": "467.0",
                    "Omega": "0.0",
                    "Kappa": "0.0",
                    "Phi1": "-39.0",
                    "Phi2": "-51.0",
                },
                {
                    "num": "2",
                    "filename": "03_0002.JPG",
                    "date": "2018-10-23T08:32:32",
                    "lat": "32.2",
                    "lon": "119.7",
                    "height": "468.0",
                    "Omega": "25.0",
                    "Kappa": "0.0",
                    "Phi1": "-38.0",
                    "Phi2": "-50.0",
                },
            ]
            for row in rows:
                image = np.zeros((32, 32, 3), np.uint8)
                cv2.imwrite(str(drone_dir / row["filename"]), image)
            with (map_dir / "03.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            queries = uav_visloc_queries(root, "03", near_nadir_deg=15.0)
            self.assertEqual([query.id for query in queries], ["03_0001", "03_0003"])
            self.assertAlmostEqual(queries[0].altitude, 467.0)
            self.assertAlmostEqual(queries[0].yaw, -39.0)
            self.assertGreater(queries[1].timestamp, queries[0].timestamp)
            self.assertTrue(queries[0].meta["attitude_present"])

    def test_missing_attitude_is_not_a_measured_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drone = root / "07" / "drone"
            drone.mkdir(parents=True)
            cv2.imwrite(str(drone / "07_0001.JPG"), np.zeros((8, 8, 3), np.uint8))
            row = {"num": "1", "filename": "07_0001.JPG", "date": "2018071250114",
                   "lat": "40", "lon": "115", "height": "688"}
            with (root / "07" / "07.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            query = uav_visloc_queries(root, "07")[0]
            self.assertFalse(query.meta["attitude_present"])


if __name__ == "__main__":
    unittest.main()
