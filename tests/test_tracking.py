from __future__ import annotations

import unittest

import numpy as np

from sat2aerial.tracking import FilterConfig, gate_and_smooth, robust_trajectory


def _straight_track(count: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    index = np.arange(count, dtype=float)
    truth = np.stack([index * 10.0, index * -3.0], axis=1)
    noisy = truth + rng.normal(0.0, 3.0, truth.shape)
    return truth, noisy


class TrackingTests(unittest.TestCase):
    def test_gate_rejects_gross_outliers(self) -> None:
        truth, noisy = _straight_track()
        measurements = noisy.copy()
        outliers = [7, 13, 22, 31]
        measurements[outliers] += 120.0
        valid = np.ones(len(measurements), dtype=bool)

        filtered, updated, gated = gate_and_smooth(
            measurements,
            valid,
            FilterConfig(measure_std_m=5.0, process_std_m=2.0, velocity_std_m=20.0, gate_sigma=3.0),
        )
        self.assertGreaterEqual(int(gated[outliers].sum()), 3)
        errors = np.hypot(filtered[:, 0] - truth[:, 0], filtered[:, 1] - truth[:, 1])
        errors = errors[np.isfinite(errors)]
        self.assertLess(float(np.median(errors)), 6.0)
        self.assertGreater(int(updated.sum()), 30)

    def test_filter_coasts_through_invalid_samples(self) -> None:
        truth, noisy = _straight_track()
        valid = np.ones(len(noisy), dtype=bool)
        valid[20:25] = False
        filtered, updated, _ = gate_and_smooth(noisy, valid)
        self.assertFalse(updated[20:25].any())
        # every sample still has an estimate because the model coasts
        self.assertTrue(np.all(np.isfinite(filtered[:, 0])))
        errors = np.hypot(filtered[20:25, 0] - truth[20:25, 0], filtered[20:25, 1] - truth[20:25, 1])
        self.assertLess(float(np.median(errors)), 8.0)

    def test_robust_trajectory_keeps_the_line_and_drops_outliers(self) -> None:
        truth, noisy = _straight_track()
        measurements = noisy.copy()
        measurements[[5, 18, 27]] += 150.0
        valid = np.ones(len(measurements), dtype=bool)

        fitted, inliers, info = robust_trajectory(measurements, valid, threshold=12.0)
        self.assertGreater(info["inlier_fraction"], 0.85)
        self.assertFalse(inliers[[5, 18, 27]].any())
        errors = np.hypot(fitted[:, 0] - truth[:, 0], fitted[:, 1] - truth[:, 1])
        self.assertLess(float(np.median(errors)), 4.0)

    def test_robust_trajectory_recovers_the_slopes(self) -> None:
        _, noisy = _straight_track()
        _, _, info = robust_trajectory(noisy, np.ones(len(noisy), dtype=bool), threshold=12.0)
        self.assertAlmostEqual(info["vx_m_per_sample"], 10.0, delta=1.0)
        self.assertAlmostEqual(info["vy_m_per_sample"], -3.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
