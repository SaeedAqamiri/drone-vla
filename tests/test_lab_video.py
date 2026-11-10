import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from navlab.lab.sequences import video_queries


class LabVideoTest(unittest.TestCase):
    def test_video_with_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "flight.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 64),
            )
            for index in range(20):
                frame = np.full((64, 64, 3), index * 10, np.uint8)
                writer.write(frame)
            writer.release()
            telemetry_path = root / "telemetry.csv"
            with telemetry_path.open("w", newline="", encoding="utf-8") as handle:
                writer_csv = csv.DictWriter(
                    handle, fieldnames=["timestamp", "lat", "lon", "alt", "yaw"]
                )
                writer_csv.writeheader()
                for index in range(20):
                    writer_csv.writerow(
                        {
                            "timestamp": index / 10.0,
                            "lat": 32.0 + index * 1e-5,
                            "lon": 119.0 + index * 1e-5,
                            "alt": 100.0,
                            "yaw": 0.0,
                        }
                    )
            queries = video_queries(
                video_path,
                root / "out",
                telemetry_csv=telemetry_path,
                sample_fps=10.0,
            )
            self.assertEqual(len(queries), 20)
            self.assertTrue(all(query.has_truth for query in queries))
            self.assertTrue(queries[0].path.exists())
            self.assertAlmostEqual(queries[0].altitude, 100.0)

    def test_video_without_telemetry_marks_no_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "flight.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (32, 32),
            )
            for _ in range(5):
                writer.write(np.zeros((32, 32, 3), np.uint8))
            writer.release()
            queries = video_queries(video_path, root / "out", sample_fps=5.0)
            self.assertEqual(len(queries), 5)
            self.assertFalse(any(query.has_truth for query in queries))


if __name__ == "__main__":
    unittest.main()
