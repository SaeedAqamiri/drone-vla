"""Offline evaluation of the visual keyframe localizer on a recorded map."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from navlab.localizer import MapLocalizer


def make_holdout(source: Path, target: Path, every: int = 2) -> list[dict]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    map_frames = [item for index, item in enumerate(frames) if index % every == 0]
    live_frames = [item for index, item in enumerate(frames) if index % every == 1]
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(
        json.dumps({"z0": manifest.get("z0"), "frames": map_frames}, indent=2),
        encoding="utf-8",
    )
    return live_frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="outputs/navlab/maps/3")
    parser.add_argument("--eval-dir", default="outputs/navlab/maps/3_eval")
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--radius", type=float, nargs="+", default=[10.0, 20.0, 35.0])
    parser.add_argument("--out", default="outputs/navlab/maps/3_eval_report.json")
    args = parser.parse_args()

    source = Path(args.map)
    live_frames = make_holdout(source, Path(args.eval_dir), args.every)
    eval_manifest = json.loads(
        (Path(args.eval_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    print(f"map frames: {len(eval_manifest.get('frames', []))}")
    print(f"live frames: {len(live_frames)}")

    reports = []
    for radius in args.radius:
        localizer = MapLocalizer(args.eval_dir)
        prior_e, prior_n = 0.0, 0.0
        last_kf = -1
        errors: list[float] = []
        latencies: list[float] = []
        matched = 0
        for item in live_frames:
            located = localizer.locate(
                item["frame"], prior_e, prior_n, radius, last_kf
            )
            if located is None:
                continue
            matched += 1
            error = math.hypot(
                located["east"] - float(item["x"]),
                located["north"] - float(item["y"]),
            )
            errors.append(error)
            latencies.append(located["latency_ms"])
            prior_e, prior_n = located["east"], located["north"]
            last_kf = located["keyframe"]
        report = {
            "radius_m": radius,
            "live": len(live_frames),
            "matched": matched,
            "match_rate": round(matched / max(len(live_frames), 1), 3),
            "mean_error_m": round(statistics.mean(errors), 2) if errors else None,
            "median_error_m": round(statistics.median(errors), 2) if errors else None,
            "p95_error_m": round(sorted(errors)[int(len(errors) * 0.95)], 2) if errors else None,
            "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))

    Path(args.out).write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
