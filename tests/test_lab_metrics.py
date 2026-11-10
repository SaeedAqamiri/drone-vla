import unittest

from navlab.lab.metrics import compute_summary
from navlab.lab.runner import RunnerConfig
from navlab.lab.types import FrameResult


def _frame(index, accepted, error, reason, timestamp):
    return FrameResult(
        index=index,
        query_id=f"q{index}",
        timestamp=timestamp,
        truth_east=0.0,
        truth_north=0.0,
        prior_east=None,
        prior_north=None,
        candidates=1,
        attempted=True,
        matched=accepted or reason == "no_match",
        accepted=accepted,
        best_tile="t",
        best_east=None,
        best_north=None,
        error_m=error,
        prior_error_m=None,
        latency_ms=10.0,
        reason=reason,
    )


class LabMetricsTest(unittest.TestCase):
    def test_unlabelled_accepted_fix_does_not_count_as_false(self):
        result = compute_summary([_frame(0, True, None, "ok", 0.0)])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["false_fix_rate"])
        self.assertEqual(result["ground_truth_accepted"], 0)

    def test_summary_counts_and_false_fixes(self):
        config = RunnerConfig(max_error_m=20.0)
        frames = [
            _frame(0, True, 5.0, "ok", 0.0),
            _frame(1, True, 50.0, "ok", 2.0),
            _frame(2, False, None, "no_match", 4.0),
            _frame(3, True, 10.0, "ok", 6.0),
        ]
        summary = compute_summary(frames, config)
        self.assertEqual(summary["frames"], 4)
        self.assertEqual(summary["accepted"], 3)
        self.assertEqual(summary["matched"], 4)
        self.assertAlmostEqual(summary["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["false_fix_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["time_to_first_fix_s"], 0.0)
        self.assertAlmostEqual(summary["mean_fix_interval_s"], 3.0)
        self.assertEqual(summary["failure_counts"]["no_match"], 1)


if __name__ == "__main__":
    unittest.main()
