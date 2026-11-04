from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds

from .geo import local_crs

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".jp2", ".j2k", ".webp", ".bmp"}


def save_bytes(data: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def _stretch(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    values = image.astype(np.float32)
    low, high = np.percentile(values, 2), np.percentile(values, 98)
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _read_rgb(path: Path) -> np.ndarray | None:
    try:
        with rasterio.open(path) as dataset:
            bands = min(3, dataset.count)
            data = dataset.read(
                indexes=list(range(1, bands + 1)),
                out_shape=(
                    bands,
                    min(dataset.height, 1200),
                    min(dataset.width, 1200),
                ),
            )
            rgb = np.transpose(data, (1, 2, 0))
            if rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
            rgb = _stretch(rgb)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _ground_resolution(dataset) -> tuple[float, float]:
    transform = dataset.transform
    if dataset.crs and dataset.crs.is_geographic:
        center_lat = (dataset.bounds.bottom + dataset.bounds.top) / 2.0
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))
        col_m = math.hypot(transform.a * m_per_deg_lon, transform.b * m_per_deg_lat)
        row_m = math.hypot(transform.d * m_per_deg_lon, transform.e * m_per_deg_lat)
    else:
        col_m = math.hypot(transform.a, transform.d)
        row_m = math.hypot(transform.b, transform.e)
    return float(col_m), float(row_m)


def inspect_image(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "width": None,
        "height": None,
        "bounds": None,
        "crs": None,
        "gsd_m": None,
    }
    try:
        with rasterio.open(path) as dataset:
            info["width"] = dataset.width
            info["height"] = dataset.height
            info["crs"] = str(dataset.crs) if dataset.crs else None
            if dataset.crs:
                left, bottom, right, top = dataset.bounds
                if dataset.crs.to_epsg() != 4326:
                    transformer = Transformer.from_crs(
                        dataset.crs, "EPSG:4326", always_xy=True
                    )
                    left, bottom = transformer.transform(left, bottom)
                    right, top = transformer.transform(right, top)
                info["bounds"] = {
                    "min_lon": min(left, right),
                    "min_lat": min(bottom, top),
                    "max_lon": max(left, right),
                    "max_lat": max(bottom, top),
                }
                col_m, row_m = _ground_resolution(dataset)
                info["gsd_m"] = (col_m + row_m) / 2.0
                info["gsd_xy_m"] = [col_m, row_m]
                info["rotated"] = bool(
                    abs(dataset.transform.b) > 1e-15 or abs(dataset.transform.d) > 1e-15
                )
                return info
    except Exception:
        pass
    try:
        with Image.open(path) as image:
            info["width"], info["height"] = image.size
    except Exception:
        pass
    return info


def make_preview(path: Path, dest: Path) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = _read_rgb(path)
    if image is None:
        try:
            with Image.open(path) as pil:
                image = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(dest), image)
    return dest


def crop_bounds(
    path: Path,
    bounds: dict[str, float],
    dest: Path,
    tile_px: int = 1024,
) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(path) as dataset:
            if not dataset.crs:
                return None
            min_lon = bounds["min_lon"]
            max_lon = bounds["max_lon"]
            min_lat = bounds["min_lat"]
            max_lat = bounds["max_lat"]
            center_lat = (min_lat + max_lat) / 2.0
            center_lon = (min_lon + max_lon) / 2.0
            crs = local_crs(center_lat, center_lon)
            to_local = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            corners = [
                to_local.transform(lon, lat)
                for lon, lat in (
                    (min_lon, min_lat),
                    (max_lon, min_lat),
                    (max_lon, max_lat),
                    (min_lon, max_lat),
                )
            ]
            xs = [point[0] for point in corners]
            ys = [point[1] for point in corners]
            side = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            center_x = (max(xs) + min(xs)) / 2.0
            center_y = (max(ys) + min(ys)) / 2.0
            min_x = center_x - side / 2.0
            max_x = center_x + side / 2.0
            min_y = center_y - side / 2.0
            max_y = center_y + side / 2.0
            dst_transform = transform_from_bounds(
                min_x, min_y, max_x, max_y, tile_px, tile_px
            )

            to_source = Transformer.from_crs(crs, dataset.crs, always_xy=True)
            source_corners = [
                to_source.transform(x, y)
                for x, y in (
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                )
            ]
            source_xs = [point[0] for point in source_corners]
            source_ys = [point[1] for point in source_corners]
            window = from_bounds(
                min(source_xs),
                min(source_ys),
                max(source_xs),
                max(source_ys),
                dataset.transform,
            )
            window = window.round_offsets().round_lengths()
            pad = 4
            window = Window(
                window.col_off - pad,
                window.row_off - pad,
                window.width + 2 * pad,
                window.height + 2 * pad,
            )
            if window.width < 2 or window.height < 2:
                return None
            bands = min(3, dataset.count)
            source = dataset.read(
                indexes=list(range(1, bands + 1)),
                window=window,
                boundless=True,
                fill_value=0,
            )
            source_transform = dataset.window_transform(window)
            destination = np.zeros((bands, tile_px, tile_px), dtype=source.dtype)
            for band in range(bands):
                reproject(
                    source=source[band],
                    destination=destination[band],
                    src_transform=source_transform,
                    src_crs=dataset.crs,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    resampling=Resampling.average,
                )
            rgb = np.transpose(destination, (1, 2, 0))
            if rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
            rgb = _stretch(rgb)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dest), image)
            return dest
    except Exception:
        return None


def whole_image_png(path: Path, dest: Path, size: int = 1024) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = _read_rgb(path)
    if image is None:
        try:
            with Image.open(path) as pil:
                image = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest), image)
    return dest


def bounds_for_leaflet(bounds: dict[str, float]) -> list[list[float]]:
    return [
        [bounds["min_lat"], bounds["min_lon"]],
        [bounds["max_lat"], bounds["max_lon"]],
    ]
