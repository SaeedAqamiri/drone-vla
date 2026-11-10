from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window, from_bounds

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


def inspect_image(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "width": None,
        "height": None,
        "bounds": None,
        "crs": None,
        "gsd_m": None,
        "gsd_xy_m": None,
        "rotated": False,
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
                rotated, gsd_x, gsd_y = _native_gsd(dataset)
                info["rotated"] = bool(rotated)
                info["gsd_xy_m"] = [float(gsd_x), float(gsd_y)]
                info["gsd_m"] = float((gsd_x + gsd_y) / 2.0)
                return info
    except Exception:
        pass
    try:
        with Image.open(path) as image:
            info["width"], info["height"] = image.size
    except Exception:
        pass
    return info


def _native_gsd(dataset) -> tuple[bool, float, float]:
    """``(rotated, gsd_x_m, gsd_y_m)`` for an open rasterio dataset.

    Uses the affine transform rather than ``dataset.res`` so rotated rasters
    (non-zero ``b``/``d``) report a real pixel footprint instead of ``0, 0``.
    Geographic CRSs are converted to metres at the dataset's centre latitude.
    """
    a, b, _, d, e, _ = dataset.transform[:6]
    rotated = abs(b) > 1e-12 or abs(d) > 1e-12
    size_x = math.hypot(a, d)
    size_y = math.hypot(b, e)
    crs = dataset.crs
    if crs is not None and crs.is_geographic:
        bounds = dataset.bounds
        center_lat = (bounds.bottom + bounds.top) / 2.0
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))
        return rotated, size_x * m_per_deg_lon, size_y * m_per_deg_lat
    return rotated, size_x, size_y


def _ground_resolution(dataset) -> tuple[float, float]:
    """Per-axis ground resolution (metres/pixel) of an open rasterio dataset."""
    _, gsd_x, gsd_y = _native_gsd(dataset)
    return gsd_x, gsd_y


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
            left = bounds["min_lon"]
            right = bounds["max_lon"]
            bottom = bounds["min_lat"]
            top = bounds["max_lat"]
            if dataset.crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(
                    "EPSG:4326", dataset.crs, always_xy=True
                )
                left, bottom = transformer.transform(left, bottom)
                right, top = transformer.transform(right, top)
            window = from_bounds(left, bottom, right, top, dataset.transform)
            window = window.round_offsets().round_lengths()
            if window.width < 2 or window.height < 2:
                return None
            data = dataset.read(
                indexes=[1, 2, 3] if dataset.count >= 3 else [1],
                window=window,
                boundless=True,
                fill_value=0,
            )
            rgb = np.transpose(data, (1, 2, 0))
            if rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
            rgb = _stretch(rgb)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (tile_px, tile_px), interpolation=cv2.INTER_AREA)
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
