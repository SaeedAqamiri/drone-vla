#!/usr/bin/env python3
"""Aerial photo in, coordinates out.

Given one nadir aerial photograph and a georeferenced satellite raster, this
finds the photograph's position in the raster and prints its latitude/longitude.

How it works
------------
The raster is cut into overlapping tiles of a fixed pixel size. Each tile is a
small, self-contained map: it knows its own georeference, so a peak found inside
a tile converts straight to lon/lat. For every tile the query is correlated at
several rotations and scales; the best peak wins, then the top few tiles are
refined at native resolution. The margin between the best and the runner-up
tile is reported as the confidence, because a match that is only marginally
better than the next place on the map is not a fix.

Two reference flavours are available:

* ``--reference raw``   the satellite tiles as they are;
* ``--reference flux``  the same tiles restyled by FLUX.3-Klein into the look of
  a drone camera (see ``scripts/flux_augment_refs.py``). Cache directory keeps
  the restyled tiles so the cost is paid once per map.

Examples
--------
Evaluate against the UAV-VisLoc annotation (prints the error)::

    python scripts/solve_position.py --run data/uav_visloc/processed_v1/07 \
        --frames 07_0001,07_0010 --tile-px 1024 --rotate-step 15

Solve for an arbitrary photo on an arbitrary map::

    python scripts/solve_position.py --map area.tif --image shot.jpg --gsd 0.07
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
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sat2aerial.localize import _rotate  # noqa: E402

FLUX_DIR = REPO / "outputs" / "flux_tiles"


def prepare(gray: np.ndarray, mode: str = "gray") -> np.ndarray:
    """Normalise contrast at the working resolution -- never before resizing.

    ``edges`` correlates gradient magnitude instead of intensity. On scenes with
    large uniform areas (water, bare fields, snow) plain NCC happily locks onto
    any other uniform patch; the gradient removes that free lunch and keeps the
    structure that actually identifies the place.
    """
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if mode != "edges":
        return equalised
    smoothed = cv2.GaussianBlur(equalised, (3, 3), 0)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    high = float(np.percentile(magnitude, 99)) or 1.0
    return np.clip(magnitude / high * 255.0, 0, 255).astype(np.uint8)


def read_tile(dataset, col0: int, row0: int, width: int, height: int) -> np.ndarray | None:
    bands = min(3, dataset.count)
    window = Window(col0, row0, width, height)
    data = dataset.read(indexes=list(range(1, bands + 1)), window=window,
                        boundless=True, fill_value=0)
    if bands == 3:
        rgb = np.transpose(data, (1, 2, 0))
    else:
        rgb = np.repeat(np.transpose(data, (1, 2, 0)), 3, axis=2)
    if rgb.dtype != np.uint8:
        low, high = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb.astype(np.float32) - low) / max(1e-6, high - low) * 255, 0, 255)
        rgb = rgb.astype(np.uint8)
    return rgb


def native_pixel_size(dataset) -> tuple[float, float]:
    """Ground size of one pixel in metres, assuming a geographic CRS if needed."""
    step_x = abs(dataset.transform.a)
    step_y = abs(dataset.transform.e)
    if step_x < 1e-3 and step_y < 1e-3:
        # degrees per pixel: the raster is EPSG:4326 (or similar)
        _, latitude = warp_transform(dataset.crs, "EPSG:4326",
                                     [dataset.bounds.left], [dataset.bounds.top])
        metres_lat = 110_574.0
        metres_lon = 111_320.0 * max(0.05, abs(np.cos(np.radians(latitude[0]))))
        return step_x * metres_lon, step_y * metres_lat
    return step_x, step_y


def tile_grid(dataset, tile_px: int, stride: int, bbox: tuple[float, float, float, float] | None):
    tile_w = min(tile_px, dataset.width)
    tile_h = min(tile_px, dataset.height)
    if bbox is None:
        col_start, row_start = 0, 0
        col_stop, row_stop = dataset.width, dataset.height
    else:
        col_start, row_start, col_stop, row_stop = (int(round(value)) for value in bbox)
        col_start = max(0, col_start)
        row_start = max(0, row_start)
        col_stop = min(dataset.width, col_stop)
        row_stop = min(dataset.height, row_stop)
    tiles = []
    row = row_start
    while row < row_stop:
        col = col_start
        while col < col_stop:
            tiles.append((col, row, min(tile_w, col_stop - col), min(tile_h, row_stop - row)))
            col += stride
        row += stride
    return tiles


def restyle_tiles(tiles: list[tuple[int, int, int, int]], dataset, tile_px: int,
                  cache: Path, prompt: str, steps: int, gateway: str) -> dict[str, Path]:
    """Cut each tile, restyle it with FLUX once, and cache the result."""
    import subprocess

    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for index, (col, row, width, height) in enumerate(tiles):
        key = f"t_{col}_{row}_{width}x{height}"
        target = cache / f"{key}.png"
        out[key] = target
        if target.is_file():
            continue
        rgb = read_tile(dataset, col, row, width, height)
        if rgb is None:
            continue
        source = cache / f"{key}_src.png"
        cv2.imwrite(str(source), rgb)
        command = [
            sys.executable, str(REPO / "scripts" / "flux_augment.py"),
            "--image", str(source), "--out", str(target), "--prompt", prompt,
            "--steps", str(steps), "--gateway", gateway,
            "--width", str(rgb.shape[1]), "--height", str(rgb.shape[0]),
        ]
        print(f"  restyle {index + 1}/{len(tiles)} {key}")
        subprocess.run(command, check=True)
        source.unlink(missing_ok=True)
    return out


def correlate_tile(ref: np.ndarray, query: np.ndarray, angle: float, coarse: float,
                   mode: str = "gray") -> tuple[float, float, float] | None:
    """Best NCC peak of a rotated query inside one tile, in tile pixels.

    ``ref`` and ``query`` are single-channel. The search runs at ``coarse``
    pixels for the long side; the peak is mapped back to tile pixels so the
    caller can add the tile origin.
    """
    rotated = _rotate(query, angle)
    if rotated.shape[0] > ref.shape[0] or rotated.shape[1] > ref.shape[1]:
        # a thin reference (a corridor cut around the flight line) can only cover
        # the central band of the photo; that band is enough, and it is the same
        # restriction localize.py applies for the same reason
        top = max(0, (rotated.shape[0] - ref.shape[0]) // 2)
        left = max(0, (rotated.shape[1] - ref.shape[1]) // 2)
        height = min(rotated.shape[0], ref.shape[0])
        width = min(rotated.shape[1], ref.shape[1])
        rotated = rotated[top : top + height, left : left + width]
    if rotated.shape[0] > ref.shape[0] or rotated.shape[1] > ref.shape[1]:
        return None
    scale = min(1.0, coarse / float(max(ref.shape[:2])))
    ref_small = prepare(cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                        mode)
    query_small = prepare(cv2.resize(rotated, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA), mode)
    if query_small.shape[0] > ref_small.shape[0] or query_small.shape[1] > ref_small.shape[1]:
        return None
    scores = cv2.matchTemplate(ref_small, query_small, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    cx = (location[0] + query_small.shape[1] / 2.0) / scale
    cy = (location[1] + query_small.shape[0] / 2.0) / scale
    return float(score), cx, cy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="processed run (meta.json + poses.csv + pairs/)")
    source.add_argument("--map", help="georeferenced satellite raster")
    parser.add_argument("--image", help="aerial photo (needed with --map)")
    parser.add_argument("--gsd", type=float, default=0.0, help="photo GSD, m/px (with --map)")
    parser.add_argument("--frames", default="", help="comma separated stems (with --run)")
    parser.add_argument("--latitude", type=float, default=None, help="approximate position")
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--radius", type=float, default=0.0, help="search radius around it, m")
    parser.add_argument("--tile-px", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--rotate-step", type=float, default=15.0)
    parser.add_argument("--extra-scales", default="1.0", help="comma separated query scale factors")
    parser.add_argument("--reference", choices=("raw", "flux"), default="raw")
    parser.add_argument("--mode", choices=("gray", "edges"), default="gray",
                        help="correlate intensities or gradient magnitude")
    parser.add_argument("--flux-steps", type=int, default=12)
    parser.add_argument("--gateway", default="http://127.0.0.1:8090")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)

    if args.run:
        run = Path(args.run)
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        raster = Path(meta["satellite"])
        frame_gsd = float(meta["map"]["frame_gsd_m"])
        poses = list(csv.DictReader((run / "poses.csv").open(encoding="utf-8")))
        wanted = {item.strip() for item in args.frames.split(",") if item.strip()}
        jobs = [(row["frame"], run / "pairs" / "aerial" / f"{row['frame']}_00.jpg", row)
                for row in poses if not wanted or row["frame"] in wanted]
    else:
        if not args.image or not args.gsd:
            raise SystemExit("--map needs --image and --gsd")
        raster = Path(args.map)
        frame_gsd = args.gsd
        jobs = [(Path(args.image).stem, Path(args.image), None)]

    with rasterio.open(raster) as dataset:
        native_gsd_x, native_gsd_y = native_pixel_size(dataset)
        native_gsd = 0.5 * (native_gsd_x + native_gsd_y)
        bbox = None
        if args.latitude is not None and args.longitude is not None and args.radius > 0:
            x, y = warp_transform("EPSG:4326", dataset.crs, [args.longitude], [args.latitude])
            col, row = ~dataset.transform @ (x[0], y[0])
            half = args.radius / native_gsd
            bbox = (col - half, row - half, col + half, row + half)
        stride = max(1, int(round(args.tile_px * (1.0 - args.overlap))))
        tiles = tile_grid(dataset, args.tile_px, stride, bbox)
        print(f"raster {dataset.width}x{dataset.height} at {native_gsd:.3f} m/px; "
              f"{len(tiles)} tiles of {args.tile_px} px ({args.tile_px * native_gsd:.0f} m)")

        references: dict[tuple[int, int, int, int], np.ndarray] = {}
        if args.reference == "flux":
            prompt = args.prompt or (
                "aerial photograph of exactly this scene taken by a small fixed-wing drone camera "
                "at nadir, keep every building, road, field boundary and tree in exactly the same "
                "place and the same size, only change the capture style: real-world daylight, "
                "slight atmospheric haze, soft lens, mild sensor noise, no text, no watermark"
            )
            if not args.radius:
                print("warning: restyling a large map is expensive; pass --latitude/--longitude/"
                      "--radius to limit it")
            cache = FLUX_DIR / Path(raster).stem
            paths = restyle_tiles(tiles, dataset, args.tile_px, cache, prompt,
                                  args.flux_steps, args.gateway)
            for index, tile in enumerate(tiles):
                key = f"t_{tile[0]}_{tile[1]}_{tile[2]}x{tile[3]}"
                path = paths.get(key)
                if path is not None and path.is_file():
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        references[tile] = image
        else:
            for tile in tiles:
                rgb = read_tile(dataset, tile[0], tile[1], tile[2], tile[3])
                if rgb is not None:
                    references[tile] = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        print(f"references ready: {len(references)}")

        angles = list(np.arange(0.0, 360.0, args.rotate_step).tolist())
        scales = [float(item) for item in args.extra_scales.split(",")]
        rows = []
        for stem, query_path, pose in jobs:
            query = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
            if query is None:
                continue
            query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
            base = frame_gsd / native_gsd
            per_tile: list[tuple[float, tuple[int, int, int, int], float, float, float]] = []
            # NOTE: scores are not strictly comparable across scales -- a smaller
            # template always correlates better -- so keep --extra-scales at 1.0
            # unless the camera GSD is genuinely unknown.
            for factor in scales:
                scale = base * factor
                target = (max(1, int(round(query.shape[1] * scale))),
                          max(1, int(round(query.shape[0] * scale))))
                resized = cv2.resize(query, target, interpolation=cv2.INTER_AREA)
                for tile, ref in references.items():
                    tile_best = None
                    for angle in angles:
                        result = correlate_tile(ref, resized, angle, coarse=320, mode=args.mode)
                        if result is None:
                            continue
                        if tile_best is None or result[0] > tile_best[0]:
                            tile_best = (result[0], angle, result[1], result[2])
                    if tile_best is None:
                        continue
                    score, angle, cx, cy = tile_best
                    per_tile.append((score, tile, angle, cx, cy))
            per_tile.sort(key=lambda item: -item[0])
            if not per_tile:
                print(f"{stem}: no tile accepted the template")
                continue
            score, tile, angle, cx, cy = per_tile[0]
            col = tile[0] + cx
            row = tile[1] + cy
            x, y = dataset.transform * (col + 0.5, row + 0.5)
            lon, lat = warp_transform(dataset.crs, "EPSG:4326", [x], [y])
            lon, lat = float(lon[0]), float(lat[0])
            print(f"\n{stem}: score {score:.3f} at tile {tile[0]},{tile[1]} "
                  f"angle {angle:.0f} -> lat {lat:.7f} lon {lon:.7f}")
            for rank, (value, box, ang, tx, ty) in enumerate(per_tile[: args.top], start=1):
                print(f"   #{rank} tile {box[0]:>5},{box[1]:>5} score {value:.3f} angle {ang:>5.0f}"
                      f"  ({box[0] + tx:.0f},{box[1] + ty:.0f}) px")
            margin = score - per_tile[1][0] if len(per_tile) > 1 else score
            record = {"frame": stem, "score": round(score, 4), "lat": lat, "lon": lon,
                      "margin": round(margin, 4), "angle": angle}
            if pose is not None:
                true_lat, true_lon = float(pose["lat"]), float(pose["lon"])
                metres_lat = 111_320.0
                metres_lon = 111_320.0 * np.cos(np.radians(true_lat))
                east = (lon - true_lon) * metres_lon
                north = (lat - true_lat) * metres_lat
                record["error_m"] = round(float(np.hypot(east, north)), 2)
                print(f"   error {record['error_m']:.1f} m (E{east:+.1f} N{north:+.1f}), "
                      f"margin {margin:+.3f}")
            rows.append(record)

    if rows:
        errors = [row["error_m"] for row in rows if "error_m" in row]
        if errors:
            print(f"\nframes {len(rows)} | median error {np.median(errors):.2f} m | "
                  f"best {np.min(errors):.2f} m | under 10 m "
                  f"{np.mean(np.array(errors) < 10):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
