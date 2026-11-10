"""Innovation gating and smoothing for per-frame map-matching fixes.

A single-frame matcher occasionally locks onto the wrong, similar-looking
parcel. A navigator never trusts one frame in isolation: it predicts where the
vehicle should be from the previous fixes and rejects measurements that fall
outside the innovation gate. This module is that layer, kept independent of the
matcher so it can also be applied offline to a recorded ``results.csv``.

The model is constant velocity in the local ENU frame. ``process_std_m`` is the
per-sample acceleration uncertainty, not the vehicle speed, so it stays small
for a straight survey line and grows for acrobatic flight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FilterConfig:
    #: 1-sigma of one matcher fix, metres
    measure_std_m: float = 20.0
    #: 1-sigma of the unmodelled acceleration per sample, metres
    process_std_m: float = 8.0
    #: 1-sigma of the unknown initial velocity, metres/sample
    velocity_std_m: float = 120.0
    #: innovation gate in sigmas
    gate_sigma: float = 3.0


def robust_line(
    values: np.ndarray,
    valid: np.ndarray,
    threshold: float = 15.0,
    iterations: int = 400,
    seed: int = 0,
) -> dict:
    """RANSAC line ``value = intercept + slope * index`` over valid samples.

    Survey flights are straight lines at constant speed, so a line is a strong
    motion model. Two-point hypotheses are cheap and need no linear algebra;
    the returned inlier mask is what a temporal filter would keep.
    """
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    indices = np.flatnonzero(valid)
    if indices.size < 2:
        return {"slope": 0.0, "intercept": float(values[indices[0]]) if indices.size else 0.0,
                "inliers": np.zeros(len(values), dtype=bool), "residual_median": None, "ok": False}
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(values), dtype=bool)
    best_cost = -1.0
    for _ in range(iterations):
        first, second = rng.choice(indices, size=2, replace=False)
        if first == second:
            continue
        slope = (values[second] - values[first]) / float(second - first)
        intercept = values[first] - slope * first
        residual = np.abs(values[indices] - (intercept + slope * indices))
        mask = residual <= threshold
        cost = float(mask.sum())
        if cost > best_cost:
            best_cost = cost
            best_mask = np.zeros(len(values), dtype=bool)
            best_mask[indices[mask]] = True
            best_slope, best_intercept = slope, intercept
    # A line that explains fewer than 15% of the samples is not a model of the
    # trajectory; without this guard a pair of outliers produces a 25 km line.
    if best_cost < max(4.0, 0.15 * float(indices.size)):
        return {"slope": float("nan"), "intercept": float("nan"),
                "inliers": np.zeros(len(values), dtype=bool), "residual_median": None, "ok": False}
    indices = np.flatnonzero(best_mask)
    if indices.size >= 2:
        # least-squares refit on the inliers
        design = np.vstack([np.ones(indices.size), indices]).T
        coefficients, *_ = np.linalg.lstsq(design, values[indices], rcond=None)
        best_intercept, best_slope = float(coefficients[0]), float(coefficients[1])
    residual = np.abs(values[best_mask] - (best_intercept + best_slope * np.flatnonzero(best_mask)))
    return {
        "slope": float(best_slope),
        "intercept": float(best_intercept),
        "inliers": best_mask,
        "residual_median": float(np.median(residual)) if residual.size else None,
        "ok": True,
    }


def robust_trajectory(
    measurements: np.ndarray,
    valid: np.ndarray,
    threshold: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit both axes independently and keep samples that are inliers of both."""
    measurements = np.asarray(measurements, dtype=float)
    east = robust_line(measurements[:, 0], valid, threshold)
    north = robust_line(measurements[:, 1], valid, threshold)
    inliers = east["inliers"] & north["inliers"]
    count = len(measurements)
    index = np.arange(count)
    valid_count = int((valid & np.isfinite(measurements).all(axis=1)).sum())
    ok = bool(
        east["ok"]
        and north["ok"]
        and int(inliers.sum()) >= max(4.0, 0.15 * float(valid_count))
    )
    if ok:
        fitted = np.stack(
            [east["intercept"] + east["slope"] * index, north["intercept"] + north["slope"] * index],
            axis=1,
        )
    else:
        fitted = np.full((count, 2), np.nan)
    info = {
        "ok": ok,
        "inliers": int(inliers.sum()),
        "valid": int((valid & np.isfinite(measurements).all(axis=1)).sum()),
        "inlier_fraction": round(float(inliers.sum()) / max(1.0, float((valid).sum())), 4),
        "residual_east": east["residual_median"],
        "residual_north": north["residual_median"],
        "vx_m_per_sample": round(east["slope"], 3) if ok else None,
        "vy_m_per_sample": round(north["slope"], 3) if ok else None,
    }
    return fitted, inliers, info


def gate_and_smooth(
    measurements: np.ndarray,
    valid: np.ndarray,
    config: FilterConfig | None = None,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a constant-velocity Kalman filter with innovation gating.

    ``measurements`` is ``(N, 2)`` of east/north metres (NaN allowed where no
    measurement exists) and ``valid`` is the matcher's own accept flag. Returns
    ``(filtered, updated, gated)``:

    * ``filtered``: the state estimate at every sample, ``(N, 2)``; NaN before
      the first accepted fix;
    * ``updated``: the measurement was fused;
    * ``gated``: a valid measurement was rejected by the innovation gate.
    """
    config = config or FilterConfig()
    measurements = np.asarray(measurements, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    count = len(measurements)
    filtered = np.full((count, 2), np.nan)
    updated = np.zeros(count, dtype=bool)
    gated = np.zeros(count, dtype=bool)

    transition = np.array(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    process = np.diag(
        [
            config.process_std_m**2,
            config.process_std_m**2,
            (config.process_std_m * 0.3) ** 2,
            (config.process_std_m * 0.3) ** 2,
        ]
    )
    observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    measurement_cov = np.eye(2) * config.measure_std_m**2

    state: np.ndarray | None = None
    covariance: np.ndarray | None = None
    if initial_state is not None:
        state = np.array(initial_state, dtype=float)
        covariance = np.diag(
            [
                config.measure_std_m**2,
                config.measure_std_m**2,
                4.0**2,
                4.0**2,
            ]
        )
        filtered[0] = state[:2]

    for index in range(count):
        sample = measurements[index]
        usable = bool(valid[index]) and np.all(np.isfinite(sample))

        if state is None:
            if not usable:
                continue
            state = np.array([sample[0], sample[1], 0.0, 0.0])
            covariance = np.diag(
                [
                    config.measure_std_m**2,
                    config.measure_std_m**2,
                    config.velocity_std_m**2,
                    config.velocity_std_m**2,
                ]
            )
            filtered[index] = state[:2]
            updated[index] = True
            continue

        if index > 0:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
            filtered[index] = state[:2]

        if not usable:
            continue
        innovation = sample - observation @ state
        innovation_cov = observation @ covariance @ observation.T + measurement_cov
        distance = float(innovation @ np.linalg.solve(innovation_cov, innovation))
        if distance > config.gate_sigma**2:
            gated[index] = True
            continue
        gain = covariance @ observation.T @ np.linalg.inv(innovation_cov)
        state = state + gain @ innovation
        covariance = (np.eye(4) - gain @ observation) @ covariance
        filtered[index] = state[:2]
        updated[index] = True

    return filtered, updated, gated


def summarize(errors: np.ndarray) -> dict[str, float | int | None]:
    """Median/p90/max of a finite error array."""
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return {"frames": 0, "median": None, "p90": None, "max": None}
    return {
        "frames": int(errors.size),
        "median": round(float(np.median(errors)), 3),
        "p90": round(float(np.percentile(errors, 90)), 3),
        "max": round(float(np.max(errors)), 3),
    }
