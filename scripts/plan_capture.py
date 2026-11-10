#!/usr/bin/env python3
"""Estimate how many frames a mapping flight produces, and how big they are.

Answers the practical question before a data-collection sortie: at this speed
and altitude, how many images, at what frame rate, with how much storage, and
how does the drone GSD compare with the satellite map it must be matched to?

Example::

    python scripts/plan_capture.py --length-km 10 --speed 200 --altitudes 100 200 300
"""

from __future__ import annotations

import argparse
import math


def footprint(altitude_m: float, fov_deg: float) -> float:
    return 2.0 * altitude_m * math.tan(math.radians(fov_deg) / 2.0)


def gsd(altitude_m: float, sensor_mm: float, focal_mm: float, pixels: int) -> float:
    return altitude_m * (sensor_mm / 1000.0) / (focal_mm / 1000.0) / float(pixels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--length-km", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=200.0, help="ground speed, m/s")
    parser.add_argument("--altitudes", type=float, nargs="+", default=[100.0, 200.0, 300.0])
    parser.add_argument("--hfov", type=float, default=63.0, help="horizontal FOV, degrees")
    parser.add_argument("--vfov", type=float, default=46.0, help="vertical FOV, degrees")
    parser.add_argument("--overlap", type=float, nargs="+", default=[0.70, 0.80])
    parser.add_argument("--sensor-mm", type=float, default=35.9, help="sensor width (full frame 35.9)")
    parser.add_argument("--focal-mm", type=float, default=35.0)
    parser.add_argument("--pixels", type=int, default=7952, help="image width in pixels")
    parser.add_argument("--megapixels", type=float, default=42.0)
    parser.add_argument("--jpeg-mb", type=float, default=15.0, help="size of one JPEG")
    parser.add_argument("--satellite-gsd", type=float, default=0.30, help="reference map GSD, m/px")
    args = parser.parse_args(argv)

    length = args.length_km * 1000.0
    print(f"flight {args.length_km:.1f} km at {args.speed:.0f} m/s "
          f"({length / args.speed:.0f} s airborne), FOV {args.hfov:.0f}x{args.vfov:.0f} deg")
    print(f"camera: {args.megapixels:.0f} MP, {args.pixels} px wide, {args.focal_mm:.0f} mm lens\n")

    header = f"{'alt m':>6} {'across m':>9} {'along m':>8} {'GSD cm':>7}"
    for overlap in args.overlap:
        header += f" {'px':>7} {'fps':>5}"
    header += f" {'GB':>6} {'vs map':>7}"
    print(header)

    for altitude in args.altitudes:
        across = footprint(altitude, args.hfov)
        along = footprint(altitude, args.vfov)
        pixel_gsd = gsd(altitude, args.sensor_mm, args.focal_mm, args.pixels)
        row = f"{altitude:>6.0f} {across:>9.1f} {along:>8.1f} {pixel_gsd * 100:>7.1f}"

        worst = 0
        for overlap in args.overlap:
            step = along * (1.0 - overlap)
            frames = int(math.ceil(length / step)) + 1
            worst = max(worst, frames)
            row += f" {frames:>7d} {args.speed / step:>5.1f}"
        storage = worst * args.jpeg_mb / 1024.0
        row += f" {storage:>6.1f} {args.satellite_gsd / pixel_gsd:>7.1f}"
        print(row)

    print("\n'vs map' is the satellite GSD divided by the drone GSD: how many times")
    print("finer the drone sees the ground. Above ~10x the appearance gap is large")
    print("enough that sparse features fail and dense correlation or synthesis is needed.")

    print("\nsynthesis workload for augmentation (not the flight count):")
    for altitude in args.altitudes:
        corridor_m2 = length * max(across for across in [footprint(altitude, args.hfov)])
        tiles = corridor_m2 / (512 * args.satellite_gsd) ** 2
        print(f"  {altitude:>4.0f} m: {corridor_m2 / 1e6:>6.2f} km2 corridor "
              f"-> {tiles:>5.0f} crops of 512 px at {args.satellite_gsd:.2f} m/px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
