"""Score a flythrough: how far is the match-inferred position from the truth?

``orthofly`` writes, per frame, the camera image, a reference crop, and the true
pose. This module runs the matcher on every pair and turns the homography into
a metric position fix:

1. ``match_images`` returns ``H`` mapping **reference pixels -> query pixels**.
2. The query centre (the camera's optical axis) is mapped back through
   ``H^-1`` into reference pixels.
3. The offset from the centre of the reference crop is converted to metres with
   the reference GSD, so ``error_m`` is the real horizontal position error.

Two things to keep in mind when reading the report:

* With ``--ref-jitter 0`` the reference crop is centred on the true position, so
  the score is an upper bound: it measures *matchability*, not absolute fixes.
  Use a jitter comparable to your expected position uncertainty for the honest
  number.
* ``median_reproj`` is in pixels, so it does not tell you the metric error;
  only ``error_m`` does.

``--matcher ncc`` swaps sparse features for coarse-to-fine dense correlation.
That is the mode that survives real drone-vs-satellite imagery: SIFT dies on the
appearance gap, while normalised cross-correlation still locks on when both
sides are resampled to the satellite's detail limit. Its report also carries
``bias_m`` (the constant satellite-vs-GPS offset) and ``error_m_debiased``
(scatter after removing it), because a mosaic with a broken georeference
otherwise hides how well the matcher itself did.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .matching import match_images


@dataclass
class LocalizeConfig:
    run: Path
    out: Path
    detector: str = "sift"
    mode: str = "gray"
    max_dim: int = 1024
    max_frames: int | None = None
    min_inliers: int = 15
    min_ratio: float = 0.15
    min_coverage: float = 0.03
    matcher: str = "features"
    min_score: float = 0.12
    rotate_search: float = 0.0
    rotate_step: float = 2.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image.shape[1], image.shape[0]


def _pair_path(directory: Path, stem: str) -> Path | None:
    """Find a pair image regardless of whether it was written as PNG or JPEG."""
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _resize_scale(size: tuple[int, int], max_dim: int) -> float:
    return min(1.0, max_dim / float(max(size)))


def evaluate(config: LocalizeConfig) -> dict[str, Any]:
    if config.matcher == "ncc":
        return evaluate_correlation(config)
    run = Path(config.run)
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    map_meta = meta["map"]
    ref_gsd = float(map_meta["gsd_m"])
    # geographic rasters have anisotropic pixels in metres; keep both axes
    ref_gsd_x = float(map_meta.get("gsd_x_m") or ref_gsd)
    ref_gsd_y = float(map_meta.get("gsd_y_m") or ref_gsd)
    ref_size = int(
        (meta.get("config") or {}).get("ref_size_px")
        or (meta.get("map") or {}).get("ref_size_px")
        or 512
    )

    with (run / "poses.csv").open(encoding="utf-8") as handle:
        poses = list(csv.DictReader(handle))
    if config.max_frames is not None:
        poses = poses[: config.max_frames]
    if not poses:
        raise ValueError(f"no rows in {run / 'poses.csv'}")

    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    latencies: list[float] = []
    import time

    for pose in poses:
        stem = pose["frame"]
        ref_path = _pair_path(run / "pairs" / "satellite", stem)
        query_path = _pair_path(run / "pairs" / "aerial", f"{stem}_00")
        if ref_path is None or query_path is None:
            continue

        started = time.perf_counter()
        result, homography = match_images(
            ref_path,
            query_path,
            mode=config.mode,
            max_dim=config.max_dim,
            detector=config.detector,
            min_inliers=config.min_inliers,
            min_ratio=config.min_ratio,
            min_coverage=config.min_coverage,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)

        query_w, query_h = _image_size(query_path)
        ref_w, ref_h = _image_size(ref_path)
        scale_query = _resize_scale((query_w, query_h), config.max_dim)
        scale_ref = _resize_scale((ref_w, ref_h), config.max_dim)
        centre = np.array([query_w * scale_query / 2.0, query_h * scale_query / 2.0, 1.0])

        error_m = None
        east = north = None
        if homography is not None:
            try:
                inverse = np.linalg.inv(homography)
                mapped = inverse @ centre
                mapped = mapped[:2] / mapped[2]
                # resized reference pixels -> original reference pixels -> metres
                delta = (mapped - np.array([ref_w * scale_ref / 2.0, ref_h * scale_ref / 2.0])) / max(scale_ref, 1e-9)
                east = float(delta[0]) * ref_gsd_x
                north = float(-delta[1]) * ref_gsd_y
                error_m = float(math.hypot(east, north))
                if result.inliers < 4:
                    error_m = None
            except np.linalg.LinAlgError:
                error_m = None

        # Absolute fix = reference-crop centre + offset of the optical axis in it.
        ref_center_x = float(pose.get("ref_center_x_m") or pose["x_m"])
        ref_center_y = float(pose.get("ref_center_y_m") or pose["y_m"])
        est_x = est_y = None
        if east is not None and north is not None:
            est_x = ref_center_x + east
            est_y = ref_center_y + north
            error_m = float(math.hypot(est_x - float(pose["x_m"]), est_y - float(pose["y_m"])))

        if error_m is not None:
            errors.append(error_m)

        rows.append(
            {
                **{key: pose[key] for key in ("frame", "lat", "lon", "x_m", "y_m", "alt_m", "gsd_m")},
                "true_x_m": pose["x_m"],
                "true_y_m": pose["y_m"],
                "est_east_m": None if east is None else round(east, 3),
                "est_north_m": None if north is None else round(north, 3),
                "est_x_m": None if est_x is None else round(est_x, 3),
                "est_y_m": None if est_y is None else round(est_y, 3),
                "error_m": None if error_m is None else round(error_m, 3),
                "inliers": result.inliers,
                "inlier_ratio": round(result.inlier_ratio, 4),
                "coverage": round(result.coverage, 4),
                "median_reproj_px": None if result.median_reproj is None else round(result.median_reproj, 3),
                "success": result.success,
                "latency_ms": round(latencies[-1], 1),
            }
        )

    if not rows:
        raise ValueError("no pairs matched the poses table; run orthofly with --pairs-dir")

    success = [row for row in rows if row["success"]]
    gated_errors = [float(row["error_m"]) for row in success if row["error_m"] is not None]
    gross_passed = sum(1 for row in success if row["error_m"] is not None and float(row["error_m"]) > 5.0)
    summary = {
        "run": str(run),
        "detector": config.detector,
        "mode": config.mode,
        "ref_gsd_m": ref_gsd,
        "ref_gsd_x_m": ref_gsd_x,
        "ref_gsd_y_m": ref_gsd_y,
        "ref_size_px": ref_size,
        "frames": len(rows),
        "fixed": len(errors),
        "fix_rate": round(len(errors) / len(rows), 4),
        "success_rate": round(len(success) / len(rows), 4),
        "error_m": {
            "median": round(_percentile(errors, 0.5), 3) if errors else None,
            "mean": round(float(np.mean(errors)), 3) if errors else None,
            "p90": round(_percentile(errors, 0.9), 3) if errors else None,
            "max": round(max(errors), 3) if errors else None,
        },
        "under_2m": round(sum(1 for value in errors if value <= 2.0) / len(rows), 4),
        "under_5m": round(sum(1 for value in errors if value <= 5.0) / len(rows), 4),
        "gated": {
            "frames": len(gated_errors),
            "median": round(_percentile(gated_errors, 0.5), 3) if gated_errors else None,
            "p90": round(_percentile(gated_errors, 0.9), 3) if gated_errors else None,
            "max": round(max(gated_errors), 3) if gated_errors else None,
            # the number that actually matters: accepted matches that are metres wrong
            "gross_passed_5m": gross_passed,
        },
        "inliers_median": float(np.median([row["inliers"] for row in rows])),
        "latency_ms_median": round(_percentile(latencies, 0.5), 1),
        "all_success": len(success) == len(rows),
    }

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _fit_to_reference(ref_shape: tuple[int, int], query: np.ndarray) -> np.ndarray:
    """Crop the query's centre to what the reference can actually cover.

    Satellite corridors in real datasets are often thinner than the drone
    footprint, so the template must be limited to the band both images share.
    """
    ref_h, ref_w = ref_shape[:2]
    height, width = query.shape[:2]
    if height <= ref_h and width <= ref_w:
        return query
    out_h, out_w = min(height, ref_h), min(width, ref_w)
    top = (height - out_h) // 2
    left = (width - out_w) // 2
    return query[top : top + out_h, left : left + out_w]


def _correlate_once(
    ref_image: np.ndarray,
    query_image: np.ndarray,
    coarse: int = 640,
    mode: str = "gray",
) -> tuple[float, float, float] | None:
    from .matching import _prepare

    height, width = ref_image.shape[:2]
    if query_image.shape[0] > height or query_image.shape[1] > width:
        return None

    def prepare(scale: float) -> tuple[np.ndarray, np.ndarray]:
        ref = cv2.resize(ref_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        query = cv2.resize(query_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return _prepare(ref, mode), _prepare(query, mode)

    def search(ref: np.ndarray, query: np.ndarray, x0: int, y0: int, margin: int) -> tuple[float, int, int]:
        if margin:
            crop = ref[
                y0 : y0 + query.shape[0] + 2 * margin,
                x0 : x0 + query.shape[1] + 2 * margin,
            ]
        else:
            crop = ref
        scores = cv2.matchTemplate(crop, query, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        return float(score), int(x0 + location[0]), int(y0 + location[1])

    scale = min(1.0, coarse / float(max(height, width)))
    ref_small, query_small = prepare(scale)
    score, left, top = search(ref_small, query_small, 0, 0, 0)

    fine = min(1.0, 1600 / float(max(height, width)))
    if fine > scale:
        ref_fine, query_fine = prepare(fine)
        margin = 6
        x0 = max(0, int(round(left * fine / scale)) - margin)
        y0 = max(0, int(round(top * fine / scale)) - margin)
        fine_score, fine_left, fine_top = search(ref_fine, query_fine, x0, y0, margin)
        if fine_score >= score:
            score, left, top = fine_score, fine_left, fine_top
            scale = fine

    centre_x = left / scale + query_image.shape[1] / 2.0
    centre_y = top / scale + query_image.shape[0] / 2.0
    return score, centre_x, centre_y


def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_w = int(height * sin + width * cos)
    out_h = int(height * cos + width * sin)
    matrix[0, 2] += out_w / 2.0 - width / 2.0
    matrix[1, 2] += out_h / 2.0 - height / 2.0
    return cv2.warpAffine(image, matrix, (out_w, out_h), flags=cv2.INTER_AREA)


def _correlate(
    ref_image: np.ndarray,
    query_image: np.ndarray,
    coarse: int = 640,
    mode: str = "gray",
    angles: tuple[float, ...] = (0.0,),
    fine_step: float = 2.0,
) -> tuple[float, float, float] | None:
    """Coarse-to-fine normalised cross-correlation, optionally over rotations.

    Returns ``(score, centre_x, centre_y)`` in reference pixels. Dense
    correlation is the fallback for real imagery, where sparse features across a
    4x GSD gap are too weak to survive the ratio test. Contrast normalisation
    (CLAHE) is applied *at the working resolution*, never before resizing: its
    tile size is a fraction of the image, so preparing a large crop and then
    shrinking it changes the intensities enough to kill the correlation.

    ``angles`` rotates the query about its optical axis; real drone frames can
    be mounted 90 degrees away from the map's north-up frame, and correlation is
    not rotation invariant. When more than a handful of angles are given, a
    cheap coarse pass picks the basin and only that basin is refined.
    """
    angles = tuple(float(value) for value in angles)
    if not angles:
        angles = (0.0,)

    coarse_step = abs(angles[1] - angles[0]) if len(angles) > 1 else 0.0
    if len(angles) > 8 and coarse_step > 0:
        scored = []
        for angle in angles:
            fitted = _fit_to_reference(ref_image.shape, _rotate(query_image, angle))
            result = _correlate_once(ref_image, fitted, coarse=320, mode=mode)
            if result is not None:
                scored.append((result[0], angle))
        if scored:
            best_angle = max(scored)[1]
            angles = tuple(
                float(value)
                for value in np.arange(
                    best_angle - coarse_step,
                    best_angle + coarse_step + 1e-9,
                    max(0.5, fine_step),
                )
            )

    best = None
    for angle in angles:
        rotated = _rotate(query_image, angle)
        fitted = _fit_to_reference(ref_image.shape, rotated)
        result = _correlate_once(ref_image, fitted, coarse=coarse, mode=mode)
        if result is None:
            continue
        if best is None or result[0] > best[0]:
            best = result
    return best


def evaluate_correlation(config: LocalizeConfig) -> dict[str, Any]:
    """Score the same pairs with dense correlation instead of sparse features."""
    import time

    run = Path(config.run)
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    map_meta = meta["map"]
    ref_gsd = float(map_meta["gsd_m"])
    ref_gsd_x = float(map_meta.get("gsd_x_m") or ref_gsd)
    ref_gsd_y = float(map_meta.get("gsd_y_m") or ref_gsd)

    with (run / "poses.csv").open(encoding="utf-8") as handle:
        poses = list(csv.DictReader(handle))
    if config.max_frames is not None:
        poses = poses[: config.max_frames]
    if not poses:
        raise ValueError(f"no rows in {run / 'poses.csv'}")

    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    scores: list[float] = []
    latencies: list[float] = []

    for pose in poses:
        stem = pose["frame"]
        ref_path = _pair_path(run / "pairs" / "satellite", stem)
        query_path = _pair_path(run / "pairs" / "aerial", f"{stem}_00")
        if ref_path is None or query_path is None:
            continue
        ref_image = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        query_image = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
        if ref_image is None or query_image is None:
            continue

        try:
            query_gsd = float(pose.get("gsd_m") or 0.0)
        except ValueError:
            query_gsd = 0.0
        if query_gsd <= 0:
            query_gsd = ref_gsd
        scale = query_gsd / ref_gsd
        if abs(scale - 1.0) > 0.02:
            query_image = cv2.resize(
                query_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )

        started = time.perf_counter()
        if config.rotate_search > 0:
            # coarse angles first; _correlate refines around the best basin
            coarse_step = max(10.0, config.rotate_step * 4.0)
            angles = tuple(
                float(value) for value in np.arange(0.0, config.rotate_search, coarse_step)
            )
        else:
            angles = (0.0,)
        matched = _correlate(
            ref_image,
            query_image,
            angles=angles,
            fine_step=max(0.5, config.rotate_step),
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        if matched is None:
            continue

        score, centre_x, centre_y = matched
        ref_h, ref_w = ref_image.shape[:2]
        # the query centre (optical axis) in reference pixels, relative to the crop
        east = (centre_x - ref_w / 2.0) * ref_gsd_x
        north = -(centre_y - ref_h / 2.0) * ref_gsd_y

        ref_center_x = float(pose.get("ref_center_x_m") or pose["x_m"])
        ref_center_y = float(pose.get("ref_center_y_m") or pose["y_m"])
        est_x = ref_center_x + east
        est_y = ref_center_y + north
        error_m = float(math.hypot(est_x - float(pose["x_m"]), est_y - float(pose["y_m"])))
        success = score >= config.min_score

        scores.append(score)
        if success:
            errors.append(error_m)

        rows.append(
            {
                **{key: pose[key] for key in ("frame", "lat", "lon", "x_m", "y_m", "alt_m", "gsd_m")},
                "est_east_m": round(east, 3),
                "est_north_m": round(north, 3),
                "est_x_m": round(est_x, 3),
                "est_y_m": round(est_y, 3),
                "error_m": round(error_m, 3),
                "score": round(score, 4),
                "success": success,
                "latency_ms": round(latencies[-1], 1),
            }
        )

    if not rows:
        raise ValueError("no pairs matched the poses table; run orthofly with --pairs-dir")

    success_rows = [row for row in rows if row["success"]]
    gross_passed = sum(1 for row in success_rows if row["error_m"] > 5.0)

    # The satellite mosaic and the drone GPS can disagree by a constant offset
    # (UAV-VisLoc v2 indeed has a broken latitude axis). Separating that offset
    # from the per-frame scatter says whether the matcher locks on or not.
    offsets_x = [float(row["est_x_m"]) - float(row["x_m"]) for row in success_rows]
    offsets_y = [float(row["est_y_m"]) - float(row["y_m"]) for row in success_rows]
    bias_x = _percentile(offsets_x, 0.5) if offsets_x else 0.0
    bias_y = _percentile(offsets_y, 0.5) if offsets_y else 0.0
    residuals = [
        float(math.hypot(east - bias_x, north - bias_y))
        for east, north in zip(offsets_x, offsets_y)
    ]

    summary = {
        "run": str(run),
        "matcher": "ncc",
        "mode": "gray",
        "min_score": config.min_score,
        "rotate_search_deg": config.rotate_search,
        "ref_gsd_m": ref_gsd,
        "ref_gsd_x_m": ref_gsd_x,
        "ref_gsd_y_m": ref_gsd_y,
        "frames": len(rows),
        "fixed": len(success_rows),
        "fix_rate": round(len(success_rows) / len(rows), 4),
        "success_rate": round(len(success_rows) / len(rows), 4),
        "error_m": {
            "median": round(_percentile(errors, 0.5), 3) if errors else None,
            "mean": round(float(np.mean(errors)), 3) if errors else None,
            "p90": round(_percentile(errors, 0.9), 3) if errors else None,
            "max": round(max(errors), 3) if errors else None,
        },
        "bias_m": {
            "east": round(bias_x, 3),
            "north": round(bias_y, 3),
            "magnitude": round(float(math.hypot(bias_x, bias_y)), 3),
        },
        "error_m_debiased": {
            "median": round(_percentile(residuals, 0.5), 3) if residuals else None,
            "p90": round(_percentile(residuals, 0.9), 3) if residuals else None,
            "max": round(max(residuals), 3) if residuals else None,
        },
        "under_2m": round(sum(1 for value in errors if value <= 2.0) / len(rows), 4),
        "under_5m": round(sum(1 for value in errors if value <= 5.0) / len(rows), 4),
        "gated": {
            "frames": len(errors),
            "median": round(_percentile(errors, 0.5), 3) if errors else None,
            "p90": round(_percentile(errors, 0.9), 3) if errors else None,
            "max": round(max(errors), 3) if errors else None,
            "gross_passed_5m": gross_passed,
        },
        "score": {
            "median": round(_percentile(scores, 0.5), 4) if scores else None,
            "p10": round(_percentile(scores, 0.1), 4) if scores else None,
            "max": round(max(scores), 4) if scores else None,
        },
        "latency_ms_median": round(_percentile(latencies, 0.5), 1),
        "all_success": len(success_rows) == len(rows),
    }

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def add_arguments(parser: Any) -> None:
    parser.add_argument("--run", required=True, help="orthofly output directory")
    parser.add_argument("--out", default=None, help="defaults to <run>/eval")
    parser.add_argument("--matcher", default="features", choices=["features", "ncc"])
    parser.add_argument("--detector", default="sift", choices=["sift", "orb", "akaze"])
    parser.add_argument("--match-mode", default="gray", choices=["edges", "gray"])
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--min-ratio", type=float, default=0.15)
    parser.add_argument("--min-coverage", type=float, default=0.03)
    parser.add_argument("--min-score", type=float, default=0.12, help="ncc matcher: correlation peak needed for a fix")
    parser.add_argument(
        "--rotate-search",
        type=float,
        default=0.0,
        help="ncc matcher: try rotations in [0, this) degrees (0 = no rotation search)",
    )
    parser.add_argument("--rotate-step", type=float, default=2.0, help="degrees between rotation samples")
    parser.add_argument("--quiet", action="store_true")


def run_from_args(args: Any) -> int:
    out = Path(args.out) if args.out else Path(args.run) / "eval"
    summary = evaluate(
        LocalizeConfig(
            run=Path(args.run),
            out=out,
            detector=args.detector,
            mode=args.match_mode,
            max_dim=args.max_dim,
            max_frames=args.max_frames,
            min_inliers=args.min_inliers,
            min_ratio=args.min_ratio,
            min_coverage=args.min_coverage,
            matcher=args.matcher,
            min_score=args.min_score,
            rotate_search=args.rotate_search,
            rotate_step=args.rotate_step,
        )
    )
    if not args.quiet:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif summary.get("matcher") == "ncc":
        print(
            f"success {summary['success_rate']:.2f} | median {summary['error_m']['median']} m "
            f"(debiased {summary['error_m_debiased']['median']} m) "
            f"| p90 {summary['error_m']['p90']} m | bias {summary['bias_m']['magnitude']} m "
            f"| gross_passed {summary['gated']['gross_passed_5m']} | score {summary['score']['median']}"
        )
    else:
        gated = summary["gated"]
        print(
            f"success {summary['success_rate']:.2f} | gated median {gated['median']} m "
            f"| gated p90 {gated['p90']} m | gross_passed {gated['gross_passed_5m']} "
            f"| inliers {summary['inliers_median']:.0f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="sat2aerial.localize", description=__doc__.splitlines()[0])
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
