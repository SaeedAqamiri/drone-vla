import unittest
from pathlib import Path

from navlab.lab.types import QueryFrame
from scripts.prepare_visloc_local import first_level_segment, initial_velocity


def frame(num, timestamp, pitch=0.0, roll=0.0, measured=True):
    return QueryFrame(str(num), Path("unused"), timestamp, 0, 0, 0, 0,
                      {"num": num, "attitude_present": measured,
                       "pitch_deg": pitch, "roll_deg": roll})


class LevelSegmentTest(unittest.TestCase):
    def test_requires_measured_consecutive_frames_and_time(self):
        queries = [frame(1, 1, measured=False), frame(2, 2), frame(3, 3, roll=float("nan")),
                   frame(4, 4), frame(6, 6), frame(7, 50), frame(8, 51), frame(9, 52)]
        selected = first_level_segment(queries, 3, 3, 10)
        self.assertEqual([q.id for q in selected], ["7", "8", "9"])

    def test_angle_bound_is_strict(self):
        with self.assertRaises(ValueError):
            first_level_segment([frame(1, 1), frame(2, 2, pitch=3), frame(3, 3)], 3, 3, 10)

    def test_index_selects_a_distinct_contiguous_run(self):
        queries = [frame(i, i) for i in (1, 2, 3, 4, 5)] + [frame(i, i) for i in (7, 8, 9, 10)]
        first = first_level_segment(queries, 3, 3, 10, 0)
        second = first_level_segment(queries, 3, 3, 10, 1)
        self.assertEqual([q.id for q in first], ["1", "2", "3"])
        self.assertEqual([q.id for q in second], ["7", "8", "9"])

    def test_velocity_fit_handles_irregular_times(self):
        times = [1000000000, 1000000002, 1000000005, 1000000009]
        points = [(500000 + 3*(t-times[0]), 3200000 - 4*(t-times[0])) for t in times]
        vx, vy = initial_velocity(points, times)
        self.assertAlmostEqual(vx, 3)
        self.assertAlmostEqual(vy, -4)
        with self.assertRaises(ValueError):
            initial_velocity([(0, 0), (1, 1)], [1, 1])


if __name__ == "__main__":
    unittest.main()
