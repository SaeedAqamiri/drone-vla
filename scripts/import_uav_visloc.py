#!/usr/bin/env python3
"""Convert a UAV-VisLoc-style flight into an ``orthofly``/``localize`` dataset.

Two layouts are supported:

* **v1** (the original 17 GB release)::

    <root>/01/drone/01_0001.JPG ...
    <root>/01/satellite01.tif
    <root>/01/01.csv          # filename, latitude, longitude, height, ...

* **v2** (the Zenodo re-release)::

    <root>/07/satellite-tile/satellite07_135_270.jpg ...
    <root>/07/07-satellite-tile-coordinates.csv   # patchname, latitude, longitude
    <root>/07/UAV-image/07_0001.JPG ...
    <root>/07/07-UAV-coordinates.csv

Both are exactly a set of co-located satellite/aerial pairs. This script cuts a
reference window around each drone frame and writes the layout the rest of the
repo already speaks::

    <out>/pairs/satellite/<stem>.png
    <out>/pairs/aerial/<stem>_00.png
    <out>/poses.csv        # lat/lon, ENU metres, true + reference-crop centre
    <out>/meta.json        # effective reference GSD, measured camera GSD, warnings

Then::

    python -m sat2aerial localize --run <out> --matcher ncc --rotate-search 360

gives the metric error against the dataset's own GPS annotations — i.e. real
imagery, real terrain, real season, no simulator involved.

The camera GSD is *measured* from consecutive frames (GPS step / pixel shift),
because EXIF is stripped in both releases. ``--frame-gsd`` overrides it.

The reference window is centred on the annotated position by default. Pass
``--offset-x/--offset-y`` metres to shift it, which turns the evaluation from
"can this match at all" into "can it produce an absolute fix".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from navlab.geo import local_crs  # noqa: E402
from navlab.imagery import _native_gsd, _stretch  # noqa: E402

NAME_KEYS = ("filename", "file", "image", "img", "name", "path")
LAT_KEYS = ("latitude", "lat", "gps_lat", "y")
LON_KEYS = ("longitude", "lon", "lng", "long", "gps_lon", "x")
ALT_KEYS = ("height", "altitude", "alt", "elevation", "agl", "z")
HEADING_KEYS = ("phi", "heading", "yaw", "orientation")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

#: the Zenodo v2 layout: ``NN/satellite-tile/satelliteNN_R_C.jpg`` + ``NN/UAV-image/``
V2_TILE_DIR = "satellite-tile"
V2_FRAME_DIR = "UAV-image"


def meters_per_degree(latitude_deg: float) -> tuple[float, float]:
    """WGS84 metres per degree of latitude and longitude at this latitude."""
    phi = math.radians(latitude_deg)
    lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi) - 0.0023 * math.cos(6 * phi)
    lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi) + 0.118 * math.cos(5 * phi)
    return lat, lon


def is_v2(site: Path) -> bool:
    return (site / V2_TILE_DIR).is_dir() and any(site.glob("*-satellite-tile-coordinates.csv"))


def _pick(header: list[str], keys: tuple[str, ...]) -> str | None:
    normalized = {column.strip().lower().replace(" ", "_"): column for column in header}
    for key in keys:
        if key in normalized:
            return normalized[key]
    for key in keys:
        for name, original in normalized.items():
            if name.startswith(key):
                return original
    return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not lines:
        return []
    if "," in lines[0]:
        return list(csv.DictReader(lines))
    header = lines[0].split()
    return [dict(zip(header, line.split())) for line in lines[1:] if line.strip()]


def _find_site_files(site: Path) -> tuple[Path | None, Path | None]:
    """Return ``(satellite_tif, metadata_csv)`` for one flight directory."""
    tif = None
    for path in sorted(site.iterdir()):
        if path.suffix.lower() in {".tif", ".tiff"} and path.is_file():
            tif = path
            break
    csvs = [p for p in sorted(site.iterdir()) if p.suffix.lower() == ".csv" and p.is_file()]
    meta = None
    for candidate in csvs:
        try:
            header = next(csv.reader([candidate.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]]))
        except Exception:
            continue
        if _pick(header, LAT_KEYS) and _pick(header, LON_KEYS):
            meta = candidate
            break
    return tif, meta


def _to_bgr(data: np.ndarray) -> np.ndarray:
    image = np.transpose(data, (1, 2, 0))
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 2:
        image = np.repeat(image[:, :, :1], 3, axis=2)
    else:
        image = image[:, :, :3]
        if image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return _stretch(image)


def _read_window(
    dataset,
    col: float,
    row: float,
    width: int,
    height: int,
    fill: int = 0,
) -> np.ndarray:
    window = Window(col - width // 2, row - height // 2, width, height)
    data = dataset.read(
        indexes=list(range(1, min(3, dataset.count) + 1)),
        window=window,
        boundless=True,
        fill_value=fill,
    )
    return _to_bgr(data)


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return image
    interpolation = cv2.INTER_AREA if image.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


def convert_flight(
    site: Path,
    out: Path,
    ref_meters: float,
    ref_size: int,
    ref_aspect: float,
    query_size: int,
    max_frames: int,
    offset_m: tuple[float, float],
    jpeg_quality: int,
    frame_gsd_arg: str = "auto",
    gsd_mode: str = "native",
    frame_stride: int = 1,
    start_frame: int = 0,
) -> dict:
    """Convert one v1 flight: a GeoTIFF per flight plus a lat/lon CSV per frame."""
    tif, csv_path = _find_site_files(site)
    if tif is None or csv_path is None:
        raise ValueError(f"{site}: need one .tif and one metadata .csv")
    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path}: no rows")

    header = list(rows[0])
    name_key = _pick(header, NAME_KEYS)
    lat_key = _pick(header, LAT_KEYS)
    lon_key = _pick(header, LON_KEYS)
    alt_key = _pick(header, ALT_KEYS)
    heading_key = _pick(header, HEADING_KEYS)
    if name_key is None:
        raise ValueError(f"{csv_path}: no filename column in {header}")

    records = []
    for row in rows:
        try:
            lat = float(row[lat_key])  # type: ignore[index]
            lon = float(row[lon_key])  # type: ignore[index]
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "name": row[name_key].strip(),  # type: ignore[index]
                "lat": lat,
                "lon": lon,
                "alt": float(row[alt_key]) if alt_key and row.get(alt_key) else 0.0,
                "heading": float(row[heading_key]) if heading_key and row.get(heading_key) else 0.0,
            }
        )

    frames: list[tuple[Path, dict]] = []
    for record in records:
        for candidate in (site / "drone" / record["name"], site / record["name"]):
            if candidate.is_file():
                frames.append((candidate, record))
                break
    if not frames:
        raise ValueError(f"{site}: no drone images were found next to {csv_path}")
    if start_frame:
        frames = frames[start_frame:]

    # Measure the camera GSD on *consecutive* frames: striding first would make
    # each pair hundreds of metres apart and turn the measurement into aliasing.
    if str(frame_gsd_arg).lower() == "auto":
        measured = measure_frame_gsd([path for path, _ in frames], [record for _, record in frames])
        if measured is None:
            raise ValueError("could not measure the drone GSD; pass --frame-gsd")
        frame_gsd = measured
    else:
        frame_gsd = float(frame_gsd_arg)

    if frame_stride and frame_stride > 1:
        frames = frames[::frame_stride]
    if max_frames and len(frames) > max_frames:
        frames = frames[:max_frames]

    probe = cv2.imread(str(frames[0][0]), cv2.IMREAD_COLOR)
    if probe is None:
        raise ValueError(f"{frames[0][0]}: unreadable frame")
    frame_width_px = query_size or probe.shape[1]
    frame_height_px = query_size or probe.shape[0]
    frame_w_m = frame_width_px * frame_gsd
    frame_h_m = frame_height_px * frame_gsd

    origin_lat = sum(item["lat"] for item in records) / len(records)
    origin_lon = sum(item["lon"] for item in records) / len(records)
    enu = local_crs(origin_lat, origin_lon)

    from pyproj import Transformer

    to_enu = Transformer.from_crs("EPSG:4326", enu, always_xy=True)
    if gsd_mode not in {"native", "square"}:
        raise ValueError(f"unknown --gsd-mode {gsd_mode!r}")

    with rasterio.open(tif) as dataset:
        _, gsd_x, gsd_y_declared = _native_gsd(dataset)
        # Some products are a square-metre mosaic that was georeferenced with
        # square-degree pixels; the north axis is then 1/cos(lat) too coarse.
        square = gsd_mode == "square"
        gsd_y = gsd_x if square else gsd_y_declared
        raster_h_m = dataset.height * gsd_y
        raster_w_m = dataset.width * gsd_x

        if ref_meters <= 0:
            ref_meters = min(1.15 * frame_h_m, 0.98 * raster_h_m)
        ref_meters = min(ref_meters, 0.98 * raster_h_m)
        if ref_aspect <= 0:
            ref_aspect = max(2.0, 1.2 * frame_w_m / ref_meters)
        ref_meters_x = ref_meters * ref_aspect
        if ref_size <= 0:
            ref_size = max(256, min(3072, int(round(ref_meters_x / gsd_x))))
        ref_size_x = int(ref_size)
        ref_size_y = max(64, int(round(ref_size * ref_meters / ref_meters_x)))
        effective_gsd = ref_meters_x / ref_size_x

        window_w = max(16, int(round(ref_meters_x / gsd_x)))
        window_h = max(16, int(round(ref_meters / gsd_y)))

        to_raster = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
        bounds = dataset.bounds
        lat_centre = (bounds.top + bounds.bottom) / 2.0
        m_lat, _ = meters_per_degree(lat_centre)

        satellite_dir = out / "pairs" / "satellite"
        aerial_dir = out / "pairs" / "aerial"
        satellite_dir.mkdir(parents=True, exist_ok=True)
        aerial_dir.mkdir(parents=True, exist_ok=True)

        pose_rows = []
        written = 0
        for image_path, record in frames:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if query_size:
                frame = _resize(frame, query_size)

            x, y = to_raster.transform(record["lon"], record["lat"])
            col, row = ~dataset.transform @ (x, y)
            if square:
                row = dataset.height / 2.0 + (lat_centre - record["lat"]) * m_lat / gsd_x

            ex, ey = to_enu.transform(record["lon"], record["lat"])
            # reference-crop centre = annotated position + requested offset
            ref_centre_x = ex + offset_m[0]
            ref_centre_y = ey + offset_m[1]

            ref_col = col + offset_m[0] / gsd_x
            ref_row = row - offset_m[1] / gsd_y  # raster rows grow southwards
            reference = _read_window(dataset, ref_col, ref_row, window_w, window_h)
            interpolation = cv2.INTER_AREA if reference.shape[0] > ref_size_y else cv2.INTER_CUBIC
            reference = cv2.resize(reference, (ref_size_x, ref_size_y), interpolation=interpolation)

            stem = image_path.stem
            # JPEG, not PNG: a 3000x2000 aerial photo is 17 MB as PNG but ~1 MB
            # as JPEG, and the write cost dominated the whole import
            quality = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
            cv2.imwrite(str(satellite_dir / f"{stem}.jpg"), reference, quality)
            cv2.imwrite(str(aerial_dir / f"{stem}_00.jpg"), frame, quality)
            pose_rows.append(
                {
                    "frame": stem,
                    "lat": round(record["lat"], 8),
                    "lon": round(record["lon"], 8),
                    "x_m": round(ex, 4),
                    "y_m": round(ey, 4),
                    "alt_m": round(record["alt"], 3),
                    "heading_deg": round(record["heading"], 3),
                    "gsd_m": round(frame_gsd, 5),
                    "map_col": round(col, 3),
                    "map_row": round(row, 3),
                    "ref_center_x_m": round(ref_centre_x, 4),
                    "ref_center_y_m": round(ref_centre_y, 4),
                }
            )
            written += 1

    if not pose_rows:
        raise ValueError(f"{site}: no drone images were found next to {csv_path}")

    with (out / "poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pose_rows[0]))
        writer.writeheader()
        writer.writerows(pose_rows)

    meta = {
        "source": str(site),
        "layout": "uav_visloc_v1",
        "satellite": str(tif),
        "metadata": str(csv_path),
        "origin": {"lat": origin_lat, "lon": origin_lon, "crs": enu},
        "map": {
            "gsd_m": effective_gsd,
            "gsd_x_m": effective_gsd,
            "gsd_y_m": effective_gsd,
            "native_gsd_m": gsd_x,
            "native_gsd_y_m": gsd_y_declared,
            "frame_gsd_m": frame_gsd,
            "ref_meters": ref_meters_x,
            "ref_meters_y": ref_meters,
            "ref_size_px": ref_size_x,
            "ref_size_py": ref_size_y,
            "raster_extent_m": [raster_w_m, raster_h_m],
        },
        "frame_footprint_m": [round(frame_w_m, 2), round(frame_h_m, 2)],
        "gsd_mode": gsd_mode,
        "frames": written,
        "offset_m": list(offset_m),
        "warnings": [],
    }
    anisotropy = gsd_y_declared / gsd_x if gsd_x else 1.0
    if abs(anisotropy - 1.0) > 0.05 and not square:
        meta["warnings"].append(
            f"the raster's north pixel is {anisotropy:.3f}x its east pixel "
            f"({gsd_y_declared:.4f} vs {gsd_x:.4f} m/px). If the source was a square-metre "
            "mosaic georeferenced in degrees, pass --gsd-mode square."
        )
    if ref_meters < 1.05 * frame_h_m:
        meta["warnings"].append(
            f"the reference crop covers {100.0 * ref_meters / frame_h_m:.0f}% of the frame "
            "height; the satellite corridor is thinner than the drone footprint."
        )
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _tile_rc(path: Path) -> tuple[int, int]:
    """``satellite07_135_270.jpg`` -> ``(135, 270)`` (pixel origins in the mosaic).

    The v2 chips are also named ``0_0.jpg``, so the prefix is optional.
    """
    match = re.search(r"(\d+)_(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"{path.name}: no _row_col suffix")
    return int(match.group(1)), int(match.group(2))


def build_mosaic(tile_dir: Path) -> tuple[np.ndarray, int, int]:
    """Paste the v2 chips at the pixel offsets encoded in their names.

    The chip names are the offsets, so the mosaic is unambiguous even though the
    dataset's latitude column is not (see ``convert_flight_v2``).
    """
    files = sorted(p for p in tile_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError(f"{tile_dir}: no satellite chips")
    grid = {}
    for path in files:
        grid[_tile_rc(path)] = path
    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    tile_h, tile_w = first.shape[:2]
    rows = max(r for r, _ in grid) + tile_h
    cols = max(c for _, c in grid) + tile_w
    mosaic = np.zeros((rows, cols, 3), np.uint8)
    for (r, c), path in grid.items():
        chip = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if chip is None:
            continue
        mosaic[r : r + chip.shape[0], c : c + chip.shape[1]] = chip
    return mosaic, tile_h, tile_w


def _crop_array(
    image: np.ndarray,
    col: float,
    row: float,
    width: int,
    height: int,
    fill: int = 0,
) -> np.ndarray:
    left, top = int(round(col)) - width // 2, int(round(row)) - height // 2
    canvas = np.full((height, width, image.shape[2]), fill, np.uint8)
    src_top, src_left = max(0, top), max(0, left)
    dst_top, dst_left = src_top - top, src_left - left
    height = min(height - dst_top, image.shape[0] - src_top)
    width = min(width - dst_left, image.shape[1] - src_left)
    if height > 0 and width > 0:
        canvas[dst_top : dst_top + height, dst_left : dst_left + width] = image[
            src_top : src_top + height, src_left : src_left + width
        ]
    return canvas


def measure_frame_gsd(images: list[Path], records: list[dict], samples: int = 12) -> float | None:
    """Measure the drone camera GSD from consecutive overlapping frames.

    Consecutive exposures are a known ground distance apart (the GPS track) and
    their pixel displacement says how many pixels that is, so the GSD needs no
    EXIF and no assumption about the lens.
    """
    if len(images) < 2 or len(records) < 2:
        return None
    m_lat, m_lon = meters_per_degree(records[0]["lat"])
    gsds = []
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    total = min(len(images) - 1, len(records) - 1)
    step = max(1, total // max(1, samples))
    indices = list(range(0, total, step))[:samples]
    for index in indices:
        a = cv2.imread(str(images[index]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(images[index + 1]), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        ka, da = sift.detectAndCompute(a, None)
        kb, db = sift.detectAndCompute(b, None)
        if da is None or db is None:
            continue
        pairs = matcher.knnMatch(da, db, k=2)
        good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            continue
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        shift = np.median(dst - src, axis=0)
        pixels = float(np.hypot(*shift))
        if pixels < 1.0:
            continue
        ground = math.hypot(
            (records[index + 1]["lat"] - records[index]["lat"]) * m_lat,
            (records[index + 1]["lon"] - records[index]["lon"]) * m_lon,
        )
        if ground > 0.5:
            gsds.append(ground / pixels)
    if not gsds:
        return None
    return float(np.median(gsds))


def convert_flight_v2(
    site: Path,
    out: Path,
    ref_size: int,
    ref_meters: float,
    ref_aspect: float,
    query_size: int,
    max_frames: int,
    offset_m: tuple[float, float],
    jpeg_quality: int,
    frame_gsd_arg: str,
    strip_shift_arg: str,
    frame_stride: int = 1,
    start_frame: int = 0,
) -> dict:
    """Convert one v2 flight: chips -> one GeoTIFF mosaic + a real-image pairs run."""
    tile_csv = next(site.glob("*-satellite-tile-coordinates.csv"))
    uav_csv = next(site.glob("*-UAV-coordinates.csv"))

    mosaic, tile_h, tile_w = build_mosaic(site / V2_TILE_DIR)

    tile_rows = _read_rows(tile_csv)
    chips: dict[tuple[int, int], dict[str, float]] = {}
    for row in tile_rows:
        try:
            row_px, col_px = _tile_rc(Path(row["patchname"].strip()))
        except (KeyError, ValueError):
            continue
        chips[(row_px, col_px)] = {
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
        }
    if not chips:
        raise ValueError(f"{tile_csv}: no usable rows")

    row_ids = sorted({r for r, _ in chips})
    col_ids = sorted({c for _, c in chips})
    if len(row_ids) < 2 or len(col_ids) < 2:
        raise ValueError(f"{tile_csv}: need a 2-D chip grid")
    row0, col0 = row_ids[0], col_ids[0]
    anchor = chips[(row0, col0)]
    m_lat, m_lon = meters_per_degree(anchor["lat"])

    gsd_east = (chips[(row0, col_ids[1])]["lon"] - anchor["lon"]) * m_lon / (col_ids[1] - col0)
    gsd_north = abs(chips[(row_ids[1], col0)]["lat"] - anchor["lat"]) * m_lat / (row_ids[1] - row0)
    gsd = float(gsd_east)
    axis_ratio = gsd_north / gsd_east
    latitude_axis_inconsistent = abs(axis_ratio - 1.0) > 0.05

    uav_rows = _read_rows(uav_csv)
    if not uav_rows:
        raise ValueError(f"{uav_csv}: no rows")
    uav_header = list(uav_rows[0])
    name_key = _pick(uav_header, NAME_KEYS)
    lat_key = _pick(uav_header, LAT_KEYS)
    lon_key = _pick(uav_header, LON_KEYS)
    alt_key = _pick(uav_header, ALT_KEYS)
    if name_key is None or lat_key is None or lon_key is None:
        raise ValueError(f"{uav_csv}: need name/lat/lon columns, got {uav_header}")
    records = []
    for row in uav_rows:
        try:
            lat = float(row[lat_key])
            lon = float(row[lon_key])
        except (TypeError, ValueError):
            continue
        try:
            alt = float(row[alt_key]) if alt_key and row.get(alt_key) else 0.0
        except ValueError:
            alt = 0.0
        records.append(
            {
                "name": row[name_key].strip(),
                "lat": lat,
                "lon": lon,
                "alt": alt,
            }
        )
    if not records:
        raise ValueError(f"{uav_csv}: no rows")

    frames: list[tuple[Path, dict]] = []
    for record in records:
        candidate = site / V2_FRAME_DIR / record["name"]
        if candidate.is_file():
            frames.append((candidate, record))
    if not frames:
        raise ValueError(f"{site}: no UAV frames matched {uav_csv}")
    if start_frame:
        frames = frames[start_frame:]

    if str(frame_gsd_arg).lower() == "auto":
        measured = measure_frame_gsd([path for path, _ in frames], [record for _, record in frames])
        if measured is None:
            raise ValueError("could not measure the drone GSD; pass --frame-gsd")
        frame_gsd = measured
    else:
        frame_gsd = float(frame_gsd_arg)

    if frame_stride and frame_stride > 1:
        frames = frames[::frame_stride]
    if max_frames and len(frames) > max_frames:
        frames = frames[:max_frames]

    probe = cv2.imread(str(frames[0][0]), cv2.IMREAD_COLOR)
    if probe is None:
        raise ValueError(f"{frames[0][0]}: unreadable frame")
    frame_width = query_size or probe.shape[1]
    frame_height = query_size or probe.shape[0]

    # the v2 satellite product is a narrow corridor: it is only ~1.15 frames
    # tall, so the crop can only grow sideways. A wide, short reference lets the
    # frame slide along the strip without including black bands.
    strip_extent_y = mosaic.shape[0] * gsd
    if ref_aspect <= 0:
        ref_aspect = 2.5
    if ref_meters <= 0:
        ref_meters = 1.15 * frame_height * frame_gsd
    ref_meters = min(ref_meters, 0.98 * strip_extent_y)
    ref_meters_x = ref_meters * ref_aspect
    if ref_size <= 0:
        # no point writing the satellite at the drone GSD: its detail stops at
        # the mosaic GSD, so cap the file and let the matcher resample
        ref_size = max(256, min(1024, int(round(ref_meters_x / frame_gsd))))
    ref_size_x = int(ref_size)
    ref_size_y = max(64, int(round(ref_size * ref_meters / ref_meters_x)))

    # map pixel -> local ENU, anchored on the (row0, col0) chip's own coordinate
    left = -(col0 + tile_w / 2.0) * gsd
    top = (row0 + tile_h / 2.0) * gsd

    enu = local_crs(anchor["lat"], anchor["lon"])
    from pyproj import Transformer

    to_enu = Transformer.from_crs("EPSG:4326", enu, always_xy=True)

    positions = [to_enu.transform(record["lon"], record["lat"]) for _, record in frames]
    strip_centre_y = -((mosaic.shape[0] / 2.0) - (row0 + tile_h / 2.0)) * gsd
    mean_y = sum(y for _, y in positions) / len(positions)
    if strip_shift_arg == "auto":
        strip_shift = strip_centre_y - mean_y
    elif strip_shift_arg == "off":
        strip_shift = 0.0
    else:
        strip_shift = float(strip_shift_arg)

    map_path = out / "map.tif"
    out.mkdir(parents=True, exist_ok=True)
    satellite_dir = out / "pairs" / "satellite"
    aerial_dir = out / "pairs" / "aerial"
    satellite_dir.mkdir(parents=True, exist_ok=True)
    aerial_dir.mkdir(parents=True, exist_ok=True)

    transform = rasterio.transform.from_origin(left, top, gsd, gsd)
    profile = {
        "driver": "GTiff",
        "height": mosaic.shape[0],
        "width": mosaic.shape[1],
        "count": 3,
        "dtype": "uint8",
        "crs": enu,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(map_path, "w", **profile) as dataset:
        dataset.write(np.transpose(mosaic, (2, 0, 1)))

    window_w = max(16, int(round(ref_meters_x / gsd)))
    window_h = max(16, int(round(ref_meters / gsd)))
    effective_gsd = ref_meters_x / float(ref_size_x)
    pose_rows = []
    for (path, record), (east, north) in zip(frames, positions):
        north += strip_shift
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if query_size:
            frame = _resize(frame, query_size)

        col = (east + offset_m[0] - left) / gsd - 0.5
        row = (top - (north + offset_m[1])) / gsd - 0.5
        reference = _crop_array(mosaic, col, row, window_w, window_h)
        reference = cv2.resize(reference, (ref_size_x, ref_size_y), interpolation=cv2.INTER_AREA)

        stem = path.stem
        quality = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        cv2.imwrite(str(satellite_dir / f"{stem}.jpg"), reference, quality)
        cv2.imwrite(str(aerial_dir / f"{stem}_00.jpg"), frame, quality)
        pose_rows.append(
            {
                "frame": stem,
                "lat": round(record["lat"], 8),
                "lon": round(record["lon"], 8),
                "x_m": round(east, 4),
                "y_m": round(north, 4),
                "alt_m": round(record["alt"], 3),
                "heading_deg": 0.0,
                "gsd_m": round(frame_gsd, 5),
                "map_col": round(col, 3),
                "map_row": round(row, 3),
                "ref_center_x_m": round(east + offset_m[0], 4),
                "ref_center_y_m": round(north + offset_m[1], 4),
            }
        )

    if not pose_rows:
        raise ValueError(f"{site}: no readable UAV frames")

    with (out / "poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pose_rows[0]))
        writer.writeheader()
        writer.writerows(pose_rows)

    meta = {
        "source": str(site),
        "layout": "uav_visloc_v2",
        "satellite": str(map_path),
        "metadata": str(uav_csv),
        "origin": {"lat": anchor["lat"], "lon": anchor["lon"], "crs": enu},
        "map": {
            "gsd_m": effective_gsd,
            "gsd_x_m": effective_gsd,
            "gsd_y_m": effective_gsd,
            "mosaic_gsd_m": gsd,
            "frame_gsd_m": frame_gsd,
            "ref_meters": ref_meters_x,
            "ref_meters_y": ref_meters,
            "ref_size_px": ref_size_x,
            "ref_size_py": ref_size_y,
            "strip_extent_m": strip_extent_y,
            "width_px": mosaic.shape[1],
            "height_px": mosaic.shape[0],
        },
        "frames": len(pose_rows),
        "offset_m": list(offset_m),
        "strip_shift_m": round(strip_shift, 3),
        "warnings": [],
    }
    if latitude_axis_inconsistent:
        meta["warnings"].append(
            f"v2 latitude axis does not match the chip pixel grid: north step is "
            f"{axis_ratio:.4f}x the east step (1/cos(lat) = {1/math.cos(math.radians(anchor['lat'])):.4f}). "
            "Trusting the longitude axis and shifting the strip by strip_shift_m."
        )
    if frame_gsd > 3 * gsd:
        meta["warnings"].append(
            f"drone GSD {frame_gsd:.4f} m/px is {frame_gsd/gsd:.1f}x finer than the satellite mosaic "
            f"{gsd:.4f} m/px; reference crops are resampled to the mosaic's detail limit."
        )
    if ref_meters >= 0.98 * strip_extent_y:
        meta["warnings"].append(
            f"reference crop is limited by the {strip_extent_y:.1f} m tall satellite strip; "
            "there is almost no room to search north-south."
        )
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="UAV-VisLoc root (containing 01/, 02/, ...) or one flight dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sites", default="", help="comma separated flight ids to convert (default: all)")
    parser.add_argument("--ref-meters", type=float, default=0.0, help="reference window height (ground metres); 0 = auto")
    parser.add_argument("--ref-aspect", type=float, default=0.0, help="reference width/height; 0 = auto (wide enough for the frame)")
    parser.add_argument("--ref-size", type=int, default=0, help="reference window output width in pixels; 0 = auto")
    parser.add_argument("--query-size", type=int, default=0, help="resize drone frames (0 keeps native)")
    parser.add_argument("--max-frames", type=int, default=0, help="per flight (0 = all)")
    parser.add_argument("--frame-stride", type=int, default=1, help="keep every Nth frame (1 = all)")
    parser.add_argument("--start-frame", type=int, default=0, help="skip the first N frames")
    parser.add_argument("--offset-x", type=float, default=0.0, help="shift the reference window east (metres)")
    parser.add_argument("--offset-y", type=float, default=0.0, help="shift the reference window north (metres)")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frame-gsd", default="auto", help="drone camera GSD in m/px, or 'auto' to measure it")
    parser.add_argument(
        "--gsd-mode",
        default="native",
        choices=["native", "square"],
        help="v1 only: 'native' trusts the raster transform; 'square' treats the raster as a "
        "square-metre mosaic whose north axis was georeferenced in degrees (1/cos(lat) too coarse)",
    )
    parser.add_argument(
        "--strip-shift",
        default="auto",
        help="v2 only: 'auto', 'off', or metres to move the satellite strip north. "
        "The v2 latitude axis is inconsistent with its own pixel grid, so the strip "
        "is re-centred on the flight line by default.",
    )
    args = parser.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    wanted = {token.strip() for token in args.sites.split(",") if token.strip()}

    candidates = []
    for child in sorted(src.iterdir()) if src.is_dir() else []:
        if not child.is_dir():
            continue
        if wanted and child.name not in wanted:
            continue
        if _find_site_files(child)[0] is not None or is_v2(child):
            candidates.append(child)
    if is_v2(src) or (src / "drone").is_dir():
        candidates = [src]
    if not candidates:
        raise SystemExit(f"no UAV-VisLoc flight directories under {src}")

    totals = []
    for site in candidates:
        destination = out / site.name if len(candidates) > 1 else out
        if destination.exists() and any(destination.iterdir()) and not args.overwrite:
            raise SystemExit(f"{destination} is not empty; pass --overwrite")
        if is_v2(site):
            meta = convert_flight_v2(
                site=site,
                out=destination,
                ref_size=args.ref_size,
                ref_meters=args.ref_meters,
                ref_aspect=args.ref_aspect,
                query_size=args.query_size,
                max_frames=args.max_frames,
                offset_m=(args.offset_x, args.offset_y),
                jpeg_quality=args.jpeg_quality,
                frame_gsd_arg=args.frame_gsd,
                strip_shift_arg=args.strip_shift,
                frame_stride=args.frame_stride,
                start_frame=args.start_frame,
            )
        else:
            meta = convert_flight(
                site=site,
                out=destination,
                ref_meters=args.ref_meters,
                ref_size=args.ref_size,
                ref_aspect=args.ref_aspect,
                query_size=args.query_size,
                max_frames=args.max_frames,
                offset_m=(args.offset_x, args.offset_y),
                jpeg_quality=args.jpeg_quality,
                frame_gsd_arg=args.frame_gsd,
                gsd_mode=args.gsd_mode,
                frame_stride=args.frame_stride,
                start_frame=args.start_frame,
            )
        totals.append(
            {
                "site": site.name,
                "out": str(destination),
                "layout": meta.get("layout", "uav_visloc_v1"),
                "frames": meta["frames"],
                "warnings": meta.get("warnings", []),
            }
        )
        print(json.dumps(totals[-1], ensure_ascii=False))

    print(json.dumps({"flights": len(totals), "frames": sum(item["frames"] for item in totals)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
