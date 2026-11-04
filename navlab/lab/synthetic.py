from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from sat2aerial.synthetic import make_aerial, make_satellite

from .types import QueryFrame


def make_scene(
    root: str | Path,
    *,
    size: int = 2048,
    seed: int = 0,
    gsd_m: float = 0.5,
    origin_lat: float = 32.0,
    origin_lon: float = 119.0,
) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image = make_satellite(rng, size=size)
    lat_per_px = gsd_m / 111_320.0
    lon_per_px = gsd_m / (111_320.0 * math.cos(math.radians(origin_lat)))
    path = root / "scene.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype=image.dtype,
        crs="EPSG:4326",
        transform=from_origin(origin_lon, origin_lat, lon_per_px, lat_per_px),
    ) as dataset:
        dataset.write(np.transpose(image, (2, 0, 1)))
    return {
        "path": str(path),
        "size": size,
        "gsd_m": gsd_m,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "lat_per_px": lat_per_px,
        "lon_per_px": lon_per_px,
    }


def make_queries(
    root: str | Path,
    scene: dict[str, Any],
    *,
    count: int = 12,
    seed: int = 0,
    altitude_m: float = 300.0,
    fov_deg: float = 73.0,
    aspect: float = 1.5,
    query_size: tuple[int, int] = (768, 512),
    start_fraction: float = 0.2,
    end_fraction: float = 0.8,
) -> list[QueryFrame]:
    root = Path(root)
    queries_dir = root / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    scene_path = Path(scene["path"])
    size = int(scene["size"])
    gsd = float(scene["gsd_m"])
    lat0 = float(scene["origin_lat"])
    lon0 = float(scene["origin_lon"])
    lat_per_px = float(scene["lat_per_px"])
    lon_per_px = float(scene["lon_per_px"])
    footprint_w = 2.0 * altitude_m * math.tan(math.radians(fov_deg) / 2.0)
    footprint_h = footprint_w / aspect
    crop_w = max(16, int(round(footprint_w / gsd)))
    crop_h = max(16, int(round(footprint_h / gsd)))
    rng = np.random.default_rng(seed + 1000)
    queries: list[QueryFrame] = []
    with rasterio.open(scene_path) as dataset:
        for index in range(max(1, int(count))):
            fraction = start_fraction + (end_fraction - start_fraction) * (
                index / max(1, count - 1)
            )
            row = int(round(size * (1.0 - fraction)))
            col = int(round(size * fraction))
            row = int(np.clip(row, crop_h // 2, size - crop_h // 2 - 1))
            col = int(np.clip(col, crop_w // 2, size - crop_w // 2 - 1))
            window = Window(
                col - crop_w // 2,
                row - crop_h // 2,
                crop_w,
                crop_h,
            )
            data = dataset.read(window=window)
            image = cv2.cvtColor(np.transpose(data, (1, 2, 0)), cv2.COLOR_RGB2BGR)
            aerial = make_aerial(image, rng, strength=0.035)
            aerial = cv2.resize(aerial, query_size, interpolation=cv2.INTER_AREA)
            query_id = f"syn_{index:04d}"
            query_path = queries_dir / f"{query_id}.png"
            cv2.imwrite(str(query_path), aerial)
            center_lon = lon0 + (col + 0.5) * lon_per_px
            center_lat = lat0 - (row + 0.5) * lat_per_px
            queries.append(
                QueryFrame(
                    id=query_id,
                    path=query_path,
                    timestamp=float(index) * 2.0,
                    lat=center_lat,
                    lon=center_lon,
                    altitude=altitude_m,
                    yaw=0.0,
                    meta={"scene": str(scene_path)},
                )
            )
    payload = {
        "meta": {"kind": "synthetic", "scene": scene, "count": len(queries)},
        "queries": [query.as_dict() for query in queries],
    }
    (root / "queries.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return queries
