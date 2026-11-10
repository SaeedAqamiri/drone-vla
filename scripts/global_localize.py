#!/usr/bin/env python3
"""Global localization: find an aerial photo's coordinates anywhere in a satellite raster.

No prior, no annotation: the query is correlated against every position of the
whole mosaic, over a small sweep of assumed scales (the camera GSD is never
known exactly) and a sweep of rotations (a drone's yaw is arbitrary). The best
peak is refined at native resolution and projected to lon/lat with the raster's
own transform.

This is the end-to-end operation the map-matching stack exists for::

    python scripts/global_localize.py --run data/uav_visloc/processed_v1/07 \\
        --frames 07_0001 --scales 0.8,1.0,1.25 --rotate-step 30 --work-gsd 0.9

``--run`` supplies the geodesy (``meta.json`` + ``poses.csv``) so the script can
also print the error against the dataset's own annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sat2aerial.localize import _rotate  # noqa: E402


def read_band_stack(path: Path, work_gsd: float, native_gsd: float) -> tuple[np.ndarray, float]:
    """Decimate the raster to roughly ``work_gsd`` and return gray + real gsd."""
    factor = max(1, int(round(work_gsd / native_gsd)))
    with rasterio.open(path) as dataset:
        width = dataset.width // factor
        height = dataset.height // factor
        data = dataset.read(indexes=[1, 2, 3], out_shape=(3, height, width),
                            resampling=Resampling.average)
    rgb = np.transpose(data, (1, 2, 0))
    if rgb.dtype != np.uint8:
        low, high = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb.astype(np.float32) - low) / max(1e-6, high - low) * 255, 0, 255)
        rgb = rgb.astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray, native_gsd * factor


def prepare(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def pixel_to_lonlat(path: Path, factor: int, col: float, row: float) -> tuple[float, float]:
    with rasterio.open(path) as dataset:
        x, y = dataset.transform * ((col + 0.5) * factor, (row + 0.5) * factor)
        crs = dataset.crs
    if crs is None:
        return float("nan"), float("nan")
    lon, lat = warp_transform(crs, "EPSG:4326", [x], [y])
    return float(lon[0]), float(lat[0])


def search(
    ref: np.ndarray,
    query_rgb: np.ndarray,
    frame_gsd: float,
    work_gsd: float,
    scales: list[float],
    angles: list[float],
) -> list[dict]:
    """Correlate the query at every scale/rotation; return the best peaks."""
    peaks: list[dict] = []
    for factor in scales:
        base_scale = (frame_gsd * factor) / work_gsd
        for angle in angles:
            rotated = _rotate(query_rgb, angle)
            target = (max(1, int(round(rotated.shape[1] * base_scale))),
                      max(1, int(round(rotated.shape[0] * base_scale))))
            if target[0] >= ref.shape[1] or target[1] >= ref.shape[0]:
                continue
            template = cv2.resize(rotated, target, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            scores = cv2.matchTemplate(ref, prepare(gray), cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            peaks.append({
                "score": float(score),
                "scale": factor,
                "angle": angle,
                "col": location[0] + template.shape[1] / 2.0,
                "row": location[1] + template.shape[0] / 2.0,
                "template": template.shape,
            })
    peaks.sort(key=lambda item: -item["score"])
    return peaks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--frames", default="", help="comma separated stems, default all")
    parser.add_argument("--scales", default="0.8,1.0,1.25")
    parser.add_argument("--rotate-step", type=float, default=30.0)
    parser.add_argument("--work-gsd", type=float, default=0.9)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args(argv)

    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    raster = Path(meta["satellite"])
    native_gsd = float(meta["map"].get("native_gsd_m") or meta["map"]["gsd_m"])
    frame_gsd = float(meta["map"]["frame_gsd_m"])

    ref, work_gsd = read_band_stack(raster, args.work_gsd, native_gsd)
    ref = prepare(ref)
    factor = max(1, int(round(work_gsd / native_gsd)))
    print(f"raster {ref.shape[1]}x{ref.shape[0]} at {work_gsd:.3f} m/px (factor {factor}); "
          f"frame gsd {frame_gsd:.4f}")

    wanted = {item.strip() for item in args.frames.split(",") if item.strip()}
    poses = [row for row in csv.DictReader((run / "poses.csv").open(encoding="utf-8"))
             if not wanted or row["frame"] in wanted]
    scales = [float(item) for item in args.scales.split(",")]
    angles = list(np.arange(0.0, 360.0, args.rotate_step).tolist())

    rows = []
    for pose in poses:
        stem = pose["frame"]
        query_path = run / "pairs" / "aerial" / f"{stem}_00.jpg"
        query = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
        if query is None:
            continue
        peaks = search(ref, query, frame_gsd, work_gsd, scales, angles)
        if not peaks:
            print(f"{stem}: no valid template")
            continue
        true_col = float(pose["map_col"]) / factor
        true_row = float(pose["map_row"]) / factor
        print(f"\n{stem}  truth col {true_col:.0f} row {true_row:.0f}")
        best = peaks[0]
        for peak in peaks[: args.top]:
            east = (peak["col"] - true_col) * work_gsd
            north = -(peak["row"] - true_row) * work_gsd
            distance = float(np.hypot(east, north))
            lon, lat = pixel_to_lonlat(raster, factor, peak["col"], peak["row"])
            print(f"   peak {peak['score']:.3f} scale x{peak['scale']:.2f} angle {peak['angle']:>5.0f}"
                  f"  offset E{east:+8.1f} N{north:+8.1f}  err {distance:8.1f} m"
                  f"  lat {lat:.7f} lon {lon:.7f}")
        east = (best["col"] - true_col) * work_gsd
        north = -(best["row"] - true_row) * work_gsd
        rows.append({
            "frame": stem,
            "score": round(best["score"], 4),
            "scale": best["scale"],
            "angle": best["angle"],
            "error_m": round(float(np.hypot(east, north)), 2),
        })

    if rows:
        errors = [row["error_m"] for row in rows]
        print(f"\nframes {len(rows)} | median error {np.median(errors):.1f} m | "
              f"under 10 m {np.mean(np.array(errors) < 10):.2f} | "
              f"median peak {np.median([row['score'] for row in rows]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
