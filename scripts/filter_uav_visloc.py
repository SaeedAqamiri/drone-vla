#!/usr/bin/env python3
"""Apply innovation gating to recorded UAV-VisLoc per-frame fixes.

Reads each ``<site>/eval/results.csv`` written by ``localize`` and reports what
a constant-velocity Kalman filter with an innovation gate would have kept::

    python scripts/filter_uav_visloc.py --root data/uav_visloc/processed_v1 \
        --measure-std 20 --process-std 8 --gate-sigma 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sat2aerial.tracking import (  # noqa: E402
    FilterConfig,
    gate_and_smooth,
    robust_trajectory,
    summarize,
)


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    measurements = np.array(
        [[float(row["est_x_m"]), float(row["est_y_m"])] for row in rows], dtype=float
    )
    truth = np.array([[float(row["x_m"]), float(row["y_m"])] for row in rows], dtype=float)
    valid = np.array([row["success"] in {"True", "true", "1"} for row in rows], dtype=bool)
    return measurements, truth, valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None, help="write filtered summary json/csv here")
    parser.add_argument("--measure-std", type=float, default=20.0)
    parser.add_argument("--process-std", type=float, default=8.0)
    parser.add_argument("--velocity-std", type=float, default=120.0)
    parser.add_argument("--gate-sigma", type=float, default=3.0)
    args = parser.parse_args(argv)

    config = FilterConfig(
        measure_std_m=args.measure_std,
        process_std_m=args.process_std,
        velocity_std_m=args.velocity_std,
        gate_sigma=args.gate_sigma,
    )
    root = Path(args.root)
    results = []
    print(f"filter: measure {config.measure_std_m} m, process {config.process_std_m} m, "
          f"gate {config.gate_sigma} sigma, velocity {config.velocity_std_m} m/sample")
    print(f"{'site':>4} {'frames':>6} {'inl%':>5} {'raw med':>8} {'line med':>8} "
          f"{'kf kept':>7} {'kf gated':>8} {'kf med':>8} {'kf p90':>8}")
    for site in sorted(path for path in root.iterdir() if path.is_dir()):
        results_path = site / "eval" / "results.csv"
        if not results_path.is_file():
            continue
        measurements, truth, valid = load(results_path)
        raw_errors = np.hypot(measurements[:, 0] - truth[:, 0], measurements[:, 1] - truth[:, 1])
        raw_errors = np.where(valid, raw_errors, np.nan)

        fitted, inliers, info = robust_trajectory(measurements, valid)
        if info["ok"]:
            line_errors = np.hypot(fitted[:, 0] - truth[:, 0], fitted[:, 1] - truth[:, 1])
            # warm-start the online filter from the robust line: a real navigator
            # begins with a position and velocity, not with zero velocity
            initial = np.array(
                [
                    fitted[0, 0],
                    fitted[0, 1],
                    info["vx_m_per_sample"],
                    info["vy_m_per_sample"],
                ]
            )
            filtered, updated, gated = gate_and_smooth(
                measurements, valid, config, initial_state=initial
            )
            filtered_errors = np.hypot(filtered[:, 0] - truth[:, 0], filtered[:, 1] - truth[:, 1])
            filtered_errors = np.where(np.isfinite(filtered[:, 0]), filtered_errors, np.nan)
        else:
            # no straight-line consensus: the per-frame fixes are not merely
            # noisy, they are inconsistent, and no temporal filter can fix that
            line_errors = np.full(len(measurements), np.nan)
            filtered_errors = np.full(len(measurements), np.nan)
            updated = np.zeros(len(measurements), dtype=bool)
            gated = np.zeros(len(measurements), dtype=bool)

        raw = summarize(raw_errors)
        line = summarize(line_errors)
        smooth = summarize(filtered_errors)
        row = {
            "site": site.name,
            "frames": len(measurements),
            "inlier_fraction": info["inlier_fraction"],
            "raw_median_m": raw["median"],
            "line_median_m": line["median"],
            "line_p90_m": line["p90"],
            "kf_updated": int(updated.sum()),
            "kf_gated": int(gated.sum()),
            "kf_median_m": smooth["median"],
            "kf_p90_m": smooth["p90"],
            "kf_max_m": smooth["max"],
            "vx_m_per_sample": info["vx_m_per_sample"],
            "vy_m_per_sample": info["vy_m_per_sample"],
        }
        results.append(row)
        print(f"{row['site']:>4} {row['frames']:>6} {row['inlier_fraction'] * 100:>5.0f} "
              f"{str(row['raw_median_m']):>8} {str(row['line_median_m']):>8} "
              f"{row['kf_updated']:>7} {row['kf_gated']:>8} "
              f"{str(row['kf_median_m']):>8} {str(row['kf_p90_m']):>8}")

    if results:
        medians = [row["kf_median_m"] for row in results if row["kf_median_m"] is not None]
        if medians:
            print(f"\nflights {len(results)} | filtered median over flights "
                  f"{sorted(medians)[len(medians) // 2]} m")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "filtered.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        fields = list(results[0])
        with (out / "filtered.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
