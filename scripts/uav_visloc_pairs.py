from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image


def latlon_to_pixel(dataset, lon: float, lat: float) -> tuple[int, int]:
    row, col = dataset.index(lon, lat)
    return int(col), int(row)


def pixel_size_m(dataset, lat: float) -> tuple[float, float]:
    xres, yres = dataset.res
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return abs(xres) * m_per_deg_lon, abs(yres) * m_per_deg_lat


def crop_satellite(
    dataset,
    lon: float,
    lat: float,
    altitude_m: float,
    fov_deg: float,
    aspect: float,
    margin: float,
    out_size: tuple[int, int],
    offset_east_m: float = 0.0,
    offset_north_m: float = 0.0,
    crop_width_m: float | None = None,
) -> tuple[np.ndarray, dict]:
    res_x, res_y = pixel_size_m(dataset, lat)
    footprint_w = 2.0 * altitude_m * math.tan(math.radians(fov_deg) / 2.0)
    footprint_h = footprint_w / aspect
    crop_w_m = crop_width_m if crop_width_m else footprint_w * margin
    crop_h_m = crop_w_m / aspect
    crop_w = max(16, int(round(crop_w_m / res_x)))
    crop_h = max(16, int(round(crop_h_m / res_y)))

    col, row = latlon_to_pixel(dataset, lon, lat)
    col += int(round(offset_east_m / res_x))
    row -= int(round(offset_north_m / res_y))
    left = col - crop_w // 2
    top = row - crop_h // 2
    window = Window(left, top, crop_w, crop_h)
    patch = dataset.read(window=window, boundless=True, fill_value=0)
    patch = np.transpose(patch[:3], (1, 2, 0))
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    patch = cv2.resize(patch, out_size, interpolation=cv2.INTER_AREA)
    meta = {
        "crop_px": [crop_w, crop_h],
        "crop_m": [round(crop_w_m, 1), round(crop_h_m, 1)],
        "footprint_m": [round(footprint_w, 1), round(footprint_h, 1)],
        "window": [left, top, crop_w, crop_h],
        "sat_res_m": [round(res_x, 3), round(res_y, 3)],
        "offset_east_m": offset_east_m,
        "offset_north_m": offset_north_m,
    }
    return patch, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--drone-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--fov", type=float, default=73.0)
    parser.add_argument("--margin", type=float, default=1.35)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--offset-east", type=float, default=0.0)
    parser.add_argument("--offset-north", type=float, default=0.0)
    parser.add_argument("--crop-width-m", type=float, default=None)
    args = parser.parse_args()

    out_root = Path(args.out)
    pairs_root = out_root / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with rasterio.open(args.satellite) as dataset, open(args.csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for index, record in enumerate(reader):
            if index >= args.limit:
                break
            filename = record["filename"].strip()
            image_path = Path(args.drone_dir) / filename
            if not image_path.exists():
                continue
            latitude = float(record["lat"])
            longitude = float(record["lon"])
            altitude = float(record["height"])
            frame_id = Path(filename).stem
            site_dir = pairs_root / frame_id
            site_dir.mkdir(parents=True, exist_ok=True)

            satellite_crop, meta = crop_satellite(
                dataset,
                longitude,
                latitude,
                altitude,
                args.fov,
                3.0 / 2.0,
                args.margin,
                (args.width, args.height),
                offset_east_m=args.offset_east,
                offset_north_m=args.offset_north,
                crop_width_m=args.crop_width_m,
            )
            aerial = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if aerial is None:
                continue
            aerial = cv2.resize(aerial, (args.width, args.height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(site_dir / "satellite.png"), satellite_crop)
            cv2.imwrite(str(site_dir / "aerial.png"), aerial)
            rows.append(
                {
                    "site": frame_id,
                    "satellite": str((site_dir / "satellite.png").relative_to(out_root)),
                    "aerial": str((site_dir / "aerial.png").relative_to(out_root)),
                    "lat": latitude,
                    "lon": longitude,
                    "height": altitude,
                    "yaw_deg": float(record.get("Phi1") or 0.0),
                    **meta,
                }
            )
    manifest = out_root / "manifest.csv"
    fieldnames = [
        "site",
        "satellite",
        "aerial",
        "lat",
        "lon",
        "height",
        "yaw_deg",
        "crop_px",
        "crop_m",
        "footprint_m",
        "window",
        "sat_res_m",
        "offset_east_m",
        "offset_north_m",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "meta.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"pairs: {len(rows)}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
