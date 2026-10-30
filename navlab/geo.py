from __future__ import annotations

import math
from typing import Any

from pyproj import Geod, Transformer
from shapely.geometry import LineString, Point, Polygon, box

GEOD = Geod(ellps="WGS84")


def route_points(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    step_m: float = 500.0,
) -> tuple[list[tuple[float, float]], float, float, float]:
    azimuth, _, distance = GEOD.inv(start_lon, start_lat, end_lon, end_lat)
    count = max(2, int(distance / max(step_m, 1.0)) + 1)
    points = GEOD.npts(start_lon, start_lat, end_lon, end_lat, count - 2)
    coords = [(start_lon, start_lat), *points, (end_lon, end_lat)]
    return coords, distance, azimuth, count


def _aeqd(lat: float, lon: float) -> str:
    return (
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 "
        "+units=m +no_defs +type=crs"
    )


def corridor_geometry(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    width_km: float,
) -> dict[str, Any]:
    coords, distance_m, azimuth, _ = route_points(
        start_lat, start_lon, end_lat, end_lon, step_m=250.0
    )
    mid_lat = (start_lat + end_lat) / 2.0
    mid_lon = (start_lon + end_lon) / 2.0
    crs = _aeqd(mid_lat, mid_lon)
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    local = [forward.transform(lon, lat) for lon, lat in coords]
    line = LineString(local)
    corridor_local = line.buffer(max(width_km * 1000.0 / 2.0, 10.0), cap_style=2)
    minx, miny, maxx, maxy = corridor_local.bounds
    corners = [
        inverse.transform(minx, miny),
        inverse.transform(maxx, miny),
        inverse.transform(maxx, maxy),
        inverse.transform(minx, maxy),
    ]
    lats = [lat for lon, lat in corners]
    lons = [lon for lon, lat in corners]
    return {
        "length_m": distance_m,
        "area_m2": corridor_local.area,
        "azimuth_deg": azimuth,
        "bbox": {
            "min_lat": min(lats),
            "min_lon": min(lons),
            "max_lat": max(lats),
            "max_lon": max(lons),
        },
        "center": {"lat": mid_lat, "lon": mid_lon},
        "crs": crs,
    }


def tile_bounds_along_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    tile_ground_m: float,
    limit: int = 8,
    step_m: float | None = None,
    offset_m: float = 0.0,
) -> list[dict[str, Any]]:
    step = step_m or max(tile_ground_m * 0.75, 50.0)
    coords, distance_m, azimuth, _ = route_points(
        start_lat, start_lon, end_lat, end_lon, step_m=step
    )
    tiles: list[dict[str, Any]] = []
    for index, (lon, lat) in enumerate(coords):
        if index % max(1, int(round(tile_ground_m / step))) != 0 and index not in {0, len(coords) - 1}:
            continue
        crs = _aeqd(lat, lon)
        forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x, y = forward.transform(lon, lat)
        half = tile_ground_m / 2.0
        corner_lonlat = [
            inverse.transform(x - half, y - half),
            inverse.transform(x + half, y - half),
            inverse.transform(x + half, y + half),
            inverse.transform(x - half, y + half),
        ]
        lats = [value[1] for value in corner_lonlat]
        lons = [value[0] for value in corner_lonlat]
        tiles.append(
            {
                "index": len(tiles),
                "center": {"lat": lat, "lon": lon},
                "bounds": {
                    "min_lat": min(lats),
                    "min_lon": min(lons),
                    "max_lat": max(lats),
                    "max_lon": max(lons),
                },
            }
        )
        if len(tiles) >= limit:
            break
    return tiles


def route_summary(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    width_km: float,
    gsd_m: float | None,
    tile_px: int = 1024,
) -> dict[str, Any]:
    corridor = corridor_geometry(start_lat, start_lon, end_lat, end_lon, width_km)
    tile_ground_m = (gsd_m or 0.5) * tile_px
    tile_area_m2 = tile_ground_m**2
    corridor["tile_ground_m"] = tile_ground_m
    corridor["tile_count_estimate"] = int(math.ceil(corridor["area_m2"] / tile_area_m2))
    corridor["length_km"] = corridor["length_m"] / 1000.0
    corridor["area_km2"] = corridor["area_m2"] / 1_000_000.0
    return corridor


def local_ned(
    home_lat: float,
    home_lon: float,
    lat: float,
    lon: float,
) -> tuple[float, float]:
    crs = _aeqd(home_lat, home_lon)
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    hx, hy = forward.transform(home_lon, home_lat)
    x, y = forward.transform(lon, lat)
    return y - hy, x - hx


def route_points_multi(
    points: list[tuple[float, float]],
    step_m: float = 500.0,
) -> tuple[list[tuple[float, float]], float]:
    coords: list[tuple[float, float]] = []
    total = 0.0
    for index in range(len(points) - 1):
        lat1, lon1 = points[index]
        lat2, lon2 = points[index + 1]
        _, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
        count = max(2, int(distance / max(step_m, 1.0)) + 1)
        intermediate = (
            GEOD.npts(lon1, lat1, lon2, lat2, count - 2) if count > 2 else []
        )
        segment = [(lon1, lat1), *intermediate, (lon2, lat2)]
        if coords:
            segment = segment[1:]
        coords.extend(segment)
        total += distance
    if not coords and points:
        coords = [(points[0][1], points[0][0])]
    return coords, total


def corridor_geometry_multi(
    points: list[tuple[float, float]],
    width_km: float,
) -> dict[str, Any]:
    if len(points) < 2:
        raise ValueError("at least two route points are required")
    coords, distance_m = route_points_multi(points, step_m=250.0)
    center_lat = sum(point[0] for point in points) / len(points)
    center_lon = sum(point[1] for point in points) / len(points)
    crs = _aeqd(center_lat, center_lon)
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    local = [forward.transform(lon, lat) for lon, lat in coords]
    line = LineString(local)
    corridor_local = line.buffer(max(width_km * 1000.0 / 2.0, 10.0), cap_style=2)
    minx, miny, maxx, maxy = corridor_local.bounds
    corners = [
        inverse.transform(minx, miny),
        inverse.transform(maxx, miny),
        inverse.transform(maxx, maxy),
        inverse.transform(minx, maxy),
    ]
    lats = [lat for lon, lat in corners]
    lons = [lon for lon, lat in corners]
    polygon = [
        [lat, lon]
        for lon, lat in (
            inverse.transform(x, y) for x, y in corridor_local.exterior.coords
        )
    ]
    return {
        "length_m": distance_m,
        "length_km": distance_m / 1000.0,
        "area_m2": corridor_local.area,
        "area_km2": corridor_local.area / 1_000_000.0,
        "bbox": {
            "min_lat": min(lats),
            "min_lon": min(lons),
            "max_lat": max(lats),
            "max_lon": max(lons),
        },
        "center": {"lat": center_lat, "lon": center_lon},
        "polygon": polygon,
        "crs": crs,
    }


def route_summary_multi(
    points: list[tuple[float, float]],
    width_km: float,
    gsd_m: float | None,
    tile_px: int = 1024,
) -> dict[str, Any]:
    corridor = corridor_geometry_multi(points, width_km)
    tile_ground_m = (gsd_m or 0.5) * tile_px
    corridor["tile_ground_m"] = tile_ground_m
    corridor["tile_count_estimate"] = int(
        math.ceil(corridor["area_m2"] / (tile_ground_m**2))
    )
    corridor["points"] = [[lat, lon] for lat, lon in points]
    return corridor


def tile_bounds_multi(
    points: list[tuple[float, float]],
    tile_ground_m: float,
    limit: int = 8,
    step_m: float | None = None,
) -> list[dict[str, Any]]:
    step = step_m or max(tile_ground_m * 0.75, 50.0)
    coords, _ = route_points_multi(points, step_m=step)
    stride = max(1, int(round(tile_ground_m / step)))
    tiles: list[dict[str, Any]] = []
    for index, (lon, lat) in enumerate(coords):
        if index % stride != 0 and index != len(coords) - 1:
            continue
        crs = _aeqd(lat, lon)
        forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x, y = forward.transform(lon, lat)
        half = tile_ground_m / 2.0
        corners = [
            inverse.transform(x - half, y - half),
            inverse.transform(x + half, y - half),
            inverse.transform(x + half, y + half),
            inverse.transform(x - half, y + half),
        ]
        lats = [value[1] for value in corners]
        lons = [value[0] for value in corners]
        tiles.append(
            {
                "index": len(tiles),
                "center": {"lat": lat, "lon": lon},
                "bounds": {
                    "min_lat": min(lats),
                    "min_lon": min(lons),
                    "max_lat": max(lats),
                    "max_lon": max(lons),
                },
            }
        )
        if len(tiles) >= limit:
            break
    return tiles


def distance_to_polyline_m(
    points: list[tuple[float, float]],
    lat: float,
    lon: float,
) -> float:
    if len(points) < 2:
        return 0.0
    center_lat = sum(point[0] for point in points) / len(points)
    center_lon = sum(point[1] for point in points) / len(points)
    crs = _aeqd(center_lat, center_lon)
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    line = LineString([forward.transform(p[1], p[0]) for p in points])
    x, y = forward.transform(lon, lat)
    return float(line.distance(Point(x, y)))


def mission_route_points(mission: Any) -> list[tuple[float, float]]:
    meta = mission.meta or {}
    raw = meta.get("waypoints") or []
    points: list[tuple[float, float]] = [
        (float(mission.start_lat), float(mission.start_lon))
    ]
    for item in raw:
        if isinstance(item, dict):
            points.append((float(item["lat"]), float(item["lon"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((float(item[0]), float(item[1])))
    points.append((float(mission.end_lat), float(mission.end_lon)))
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or abs(point[0] - deduped[-1][0]) > 1e-9 or abs(point[1] - deduped[-1][1]) > 1e-9:
            deduped.append(point)
    return deduped
