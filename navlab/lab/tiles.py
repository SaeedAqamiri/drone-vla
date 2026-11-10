from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window

from ..geo import local_crs
from ..imagery import _ground_resolution


def _stretch(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    values = image.astype(np.float32)
    low, high = np.percentile(values, 2), np.percentile(values, 98)
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _source_center(dataset) -> tuple[float, float]:
    bounds = dataset.bounds
    center_x = (bounds.left + bounds.right) / 2.0
    center_y = (bounds.bottom + bounds.top) / 2.0
    if dataset.crs and dataset.crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
        center_x, center_y = transformer.transform(center_x, center_y)
    return float(center_y), float(center_x)


def _lonlat_bounds(
    to_wgs: Transformer,
    min_east: float,
    min_north: float,
    max_east: float,
    max_north: float,
) -> dict[str, float]:
    corners = [
        to_wgs.transform(x, y)
        for x, y in (
            (min_east, min_north),
            (max_east, min_north),
            (max_east, max_north),
            (min_east, max_north),
        )
    ]
    lons = [point[0] for point in corners]
    lats = [point[1] for point in corners]
    return {
        "min_lon": min(lons),
        "min_lat": min(lats),
        "max_lon": max(lons),
        "max_lat": max(lats),
    }


def _read_tile(vrt, col_off: int, row_off: int, tile_px: int) -> np.ndarray:
    bands = min(3, vrt.count)
    canvas = np.zeros((bands, tile_px, tile_px), dtype=vrt.dtypes[0])
    request = Window(col_off, row_off, tile_px, tile_px)
    extent = Window(0, 0, vrt.width, vrt.height)
    intersection = request.intersection(extent)
    if intersection.width > 0 and intersection.height > 0:
        intersection = Window(
            int(round(intersection.col_off)),
            int(round(intersection.row_off)),
            int(round(intersection.width)),
            int(round(intersection.height)),
        )
        data = vrt.read(
            indexes=list(range(1, bands + 1)),
            window=intersection,
        )
        x0 = int(round(intersection.col_off - col_off))
        y0 = int(round(intersection.row_off - row_off))
        width = int(round(intersection.width))
        height = int(round(intersection.height))
        canvas[:, y0 : y0 + height, x0 : x0 + width] = data[:, :height, :width]
    rgb = np.transpose(canvas, (1, 2, 0))
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    rgb = _stretch(rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _build_from_centers(
    source: Path,
    out_dir: Path,
    centers: Sequence[tuple[float, float]],
    *,
    gsd: float,
    tile_px: int,
    stride: float,
    prefix: str | None,
    resampling: Resampling,
    max_tiles: int | None,
) -> dict[str, Any]:
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as dataset:
        if not dataset.crs:
            raise ValueError(f"{source} has no CRS")
        center_lat, center_lon = _source_center(dataset)
        crs = local_crs(center_lat, center_lon)
        native_col, native_row = _ground_resolution(dataset)
        to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        ordered = list(centers)
        if max_tiles is not None:
            ordered = ordered[: max(1, int(max_tiles))]
        dst_transform, dst_width, dst_height = calculate_default_transform(
            dataset.crs,
            crs,
            dataset.width,
            dataset.height,
            *dataset.bounds,
            resolution=(gsd, gsd),
        )
        tiles: list[dict[str, Any]] = []
        with WarpedVRT(
            dataset,
            crs=crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=resampling,
        ) as vrt:
            left = vrt.bounds.left
            top = vrt.bounds.top
            for index, (east, north) in enumerate(ordered):
                col_off = int(round((east - left) / gsd - tile_px / 2.0))
                row_off = int(round((top - north) / gsd - tile_px / 2.0))
                image = _read_tile(vrt, col_off, row_off, tile_px)
                tile_id = f"{prefix or source.stem}_{index:05d}"
                image_path = tiles_dir / f"{tile_id}.png"
                cv2.imwrite(str(image_path), image)
                min_east = left + col_off * gsd
                max_east = min_east + tile_px * gsd
                max_north = top - row_off * gsd
                min_north = max_north - tile_px * gsd
                tiles.append(
                    {
                        "id": tile_id,
                        "image": str(image_path.relative_to(out_dir)),
                        "east": float(east),
                        "north": float(north),
                        "bounds": {
                            "min_east": min_east,
                            "min_north": min_north,
                            "max_east": max_east,
                            "max_north": max_north,
                        },
                        "lonlat": _lonlat_bounds(
                            to_wgs, min_east, min_north, max_east, max_north
                        ),
                        "gsd": gsd,
                        "tile_px": tile_px,
                        "col": col_off,
                        "row": row_off,
                        "source": str(source),
                        "kind": "satellite",
                        "variant": "",
                    }
                )

    manifest = {
        "name": out_dir.name,
        "source": str(source),
        "crs": crs,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "native_gsd_m": [native_col, native_row],
        "gsd": gsd,
        "tile_px": tile_px,
        "tile_ground_m": gsd * tile_px,
        "stride_m": stride,
        "tiles": tiles,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def build_tiles_along_route(
    source: str | Path,
    out_dir: str | Path,
    route_lonlat: Sequence[tuple[float, float]],
    *,
    gsd: float | None = None,
    tile_px: int = 768,
    tile_ground_m: float | None = None,
    stride_m: float | None = None,
    max_tiles: int | None = None,
    prefix: str | None = None,
    resampling: Resampling = Resampling.average,
    offset_east_m: float = 0.0,
    offset_north_m: float = 0.0,
) -> dict[str, Any]:
    source = Path(source)
    out_dir = Path(out_dir)
    if not route_lonlat:
        raise ValueError("route_lonlat must contain at least one point")
    with rasterio.open(source) as dataset:
        if not dataset.crs:
            raise ValueError(f"{source} has no CRS")
        center_lat, center_lon = _source_center(dataset)
        crs = local_crs(center_lat, center_lon)
        native_col, native_row = _ground_resolution(dataset)
    if tile_ground_m is not None:
        gsd = tile_ground_m / float(tile_px)
    if gsd is None:
        gsd = (native_col + native_row) / 2.0
    gsd = float(gsd)
    if gsd <= 0:
        raise ValueError("gsd must be positive")
    stride = float(stride_m or gsd * tile_px * 0.5)
    if stride <= 0:
        raise ValueError("stride_m must be positive")
    to_local = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    centers: dict[tuple[int, int], tuple[float, float]] = {}
    for lon, lat in route_lonlat:
        east, north = to_local.transform(lon, lat)
        east += offset_east_m
        north += offset_north_m
        key = (int(round(east / stride)), int(round(north / stride)))
        centers.setdefault(key, (key[0] * stride, key[1] * stride))
    ordered = [
        point
        for _, point in sorted(centers.items(), key=lambda item: (item[0][1], item[0][0]))
    ]
    manifest = _build_from_centers(
        source,
        out_dir,
        ordered,
        gsd=gsd,
        tile_px=tile_px,
        stride=stride,
        prefix=prefix,
        resampling=resampling,
        max_tiles=max_tiles,
    )
    manifest["offset_east_m"] = offset_east_m
    manifest["offset_north_m"] = offset_north_m
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def build_tiles_in_bbox(
    source: str | Path,
    out_dir: str | Path,
    bbox: dict[str, float],
    *,
    gsd: float | None = None,
    tile_px: int = 768,
    tile_ground_m: float | None = None,
    stride_m: float | None = None,
    max_tiles: int | None = None,
    prefix: str | None = None,
    resampling: Resampling = Resampling.average,
) -> dict[str, Any]:
    source = Path(source)
    out_dir = Path(out_dir)
    with rasterio.open(source) as dataset:
        if not dataset.crs:
            raise ValueError(f"{source} has no CRS")
        center_lat, center_lon = _source_center(dataset)
        crs = local_crs(center_lat, center_lon)
        native_col, native_row = _ground_resolution(dataset)
    if tile_ground_m is not None:
        gsd = tile_ground_m / float(tile_px)
    if gsd is None:
        gsd = (native_col + native_row) / 2.0
    gsd = float(gsd)
    if gsd <= 0:
        raise ValueError("gsd must be positive")
    stride = float(stride_m or gsd * tile_px * 0.5)
    if stride <= 0:
        raise ValueError("stride_m must be positive")
    to_local = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners = [
        to_local.transform(lon, lat)
        for lon, lat in (
            (bbox["min_lon"], bbox["min_lat"]),
            (bbox["max_lon"], bbox["min_lat"]),
            (bbox["max_lon"], bbox["max_lat"]),
            (bbox["min_lon"], bbox["max_lat"]),
        )
    ]
    min_east = min(point[0] for point in corners) - stride / 2.0
    max_east = max(point[0] for point in corners) + stride / 2.0
    min_north = min(point[1] for point in corners) - stride / 2.0
    max_north = max(point[1] for point in corners) + stride / 2.0
    centers: list[tuple[float, float]] = []
    east = min_east + stride / 2.0
    while east <= max_east:
        north = min_north + stride / 2.0
        while north <= max_north:
            centers.append((east, north))
            north += stride
        east += stride
    centers.sort(key=lambda point: (point[1], point[0]))
    return _build_from_centers(
        source,
        out_dir,
        centers,
        gsd=gsd,
        tile_px=tile_px,
        stride=stride,
        prefix=prefix,
        resampling=resampling,
        max_tiles=max_tiles,
    )


class TileStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        payload = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.manifest: dict[str, Any] = payload
        self.tiles: list[dict[str, Any]] = list(payload.get("tiles", []))
        self.crs = str(payload["crs"])
        self.gsd = float(payload["gsd"])
        self.tile_ground_m = float(payload.get("tile_ground_m", 0.0))
        self._east = np.array([float(tile["east"]) for tile in self.tiles], np.float64)
        self._north = np.array([float(tile["north"]) for tile in self.tiles], np.float64)
        self._to_local = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._to_wgs = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    @property
    def size(self) -> int:
        return len(self.tiles)

    def to_local(self, lon: float, lat: float) -> tuple[float, float]:
        east, north = self._to_local.transform(lon, lat)
        return float(east), float(north)

    def to_wgs(self, east: float, north: float) -> tuple[float, float]:
        lon, lat = self._to_wgs.transform(east, north)
        return float(lon), float(lat)

    def path_for(self, tile: dict[str, Any]) -> Path:
        return self.root / str(tile["image"])

    def candidates(
        self,
        east: float,
        north: float,
        radius: float,
        limit: int | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.tiles:
            return []
        distances = (self._east - east) ** 2 + (self._north - north) ** 2
        order = np.argsort(distances)
        selected: list[dict[str, Any]] = []
        for index in order:
            if distances[index] > radius**2:
                break
            tile = self.tiles[int(index)]
            if kind is not None and tile.get("kind") != kind:
                continue
            selected.append(tile)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    def all_tiles(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind is None:
            return list(self.tiles)
        return [tile for tile in self.tiles if tile.get("kind") == kind]

    def with_references(self, references: str | Path) -> "ReferenceStore":
        return ReferenceStore(self, references)


class ReferenceStore(TileStore):
    def __init__(self, base: TileStore, references: str | Path):
        self.root = base.root
        self.manifest = dict(base.manifest)
        self.crs = base.crs
        self.gsd = base.gsd
        self.tile_ground_m = base.tile_ground_m
        self._to_local = base._to_local
        self._to_wgs = base._to_wgs
        payload = json.loads(Path(references).read_text(encoding="utf-8"))
        self.tiles = list(payload.get("tiles", []))
        self._east = np.array([float(tile["east"]) for tile in self.tiles], np.float64)
        self._north = np.array([float(tile["north"]) for tile in self.tiles], np.float64)
        self.reference_manifest = payload

    def path_for(self, tile: dict[str, Any]) -> Path:
        path = Path(str(tile["image"]))
        if path.is_absolute():
            return path
        return self.root / path
