#!/usr/bin/env python3
"""Build a georeferenced Gazebo world from a real orthophoto (+ OSM buildings).

Why this exists
---------------
A simulator world is only useful for GPS-denied navigation if its downward
camera shows the *same map* the localizer matches against. AirSim cities and
Gazebo ``baylands`` have no orthophoto, so every match score there measures the
sim2real gap. This script tiles a real GeoTIFF onto the ground plane, extrudes
building footprints on top of it, and pins the world to a WGS84 origin, so:

* world ``x`` = East, ``y`` = North, ``z`` = Up (ENU, same as ``orthofly``),
* ``(0, 0)`` in the world is ``--center-lat/--center-lon``,
* a frame from the downward camera is matchable against ``--ortho``.

Output layout
-------------
::

    <out>/models/geo_<name>/model.sdf, model.config,
                            meshes/tile.obj, textures/tile_0_0.png ...
    <out>/models/geo_<name>_buildings/...        (only with buildings)
    <out>/worlds/<name>.sdf
    <out>/env.sh            # GZ_SIM_RESOURCE_PATH for this world
    <out>/meta.json         # origin, extent, tile grid, GSD, building count

Usage
-----
::

    python3 scripts/make_geo_world.py --ortho site.tif --name site1 \\
        --extent-m 800 --tile-m 200 --tile-px 1024 \\
        --center-lat 47.397 --center-lon 8.546 \\
        --buildings data/site1_buildings.geojson

    source outputs/geo/site1/env.sh
    gz sim -r outputs/geo/site1/worlds/site1.sdf

Approximations (documented on purpose)
--------------------------------------
* Ground is flat: a GeoTIFF is 2.5D. Terrain relief needs a DEM and a heightmap
  mesh; not implemented here.
* Building footprints are reduced to their convex hull and extruded to a
  single height. Enough for parallax/occlusion; not for facade matching.
* Textures are sampled at ``--tile-px`` regardless of the source GSD, so the
  effective ground GSD is ``tile_m / tile_px``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from navlab.geo import local_crs  # noqa: E402
from sat2aerial.orthofly import GroundMap  # noqa: E402

DEFAULT_HEIGHT_M = 8.0
LEVEL_HEIGHT_M = 3.0


# --------------------------------------------------------------------------- #
# ground tiles pointed at the GeoTIFF
# --------------------------------------------------------------------------- #


def _enu_to_pixel_matrix(ground: GroundMap, x0: float, y0: float, step: float = 20.0) -> np.ndarray:
    """Local affine ``ENU metres -> map pixels`` around ``(x0, y0)``."""
    cx, cy = ground.local_to_pixel(x0, y0)
    ex, ey = ground.local_to_pixel(x0 + step, y0)
    nx, ny = ground.local_to_pixel(x0, y0 + step)
    a = (ex - cx) / step
    b = (nx - cx) / step
    c = (ey - cy) / step
    d = (ny - cy) / step
    # Re-base the constant term: the fitted point is (x0, y0), not ENU (0, 0).
    return np.array(
        [
            [a, b, cx - a * x0 - b * y0],
            [c, d, cy - c * x0 - d * y0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


TILE_PAD_PX = 2  # overlap between neighbouring quads; hides the filtering seam


def extract_tile(ground: GroundMap, x0: float, y0: float, tile_m: float, tile_px: int, flip_v: bool) -> np.ndarray:
    """Render one ground tile, row 0 = north, column 0 = west.

    ``TILE_PAD_PX`` pixels of real context are included on every side so
    adjacent quads overlap slightly instead of showing a hairline at the seam.
    """
    gsd = tile_m / tile_px
    size_px = tile_px + 2 * TILE_PAD_PX
    # tile pixel (col, row) -> ENU; row grows southwards
    tile_to_enu = np.array(
        [
            [gsd, 0.0, x0 - TILE_PAD_PX * gsd + gsd / 2.0],
            [0.0, -gsd, y0 + tile_m + TILE_PAD_PX * gsd - gsd / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix = (_enu_to_pixel_matrix(ground, x0 + tile_m / 2.0, y0 + tile_m / 2.0) @ tile_to_enu)[:2]
    tile = cv2.warpAffine(
        ground.image,
        matrix,
        (size_px, size_px),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if flip_v:
        tile = cv2.flip(tile, 0)
    return tile


def coverage_fraction(ground: GroundMap, extent_m: float) -> float:
    """How much of the requested square actually falls inside the raster."""
    half = extent_m / 2.0
    height, width = ground.image.shape[:2]
    inside = 0
    total = 0
    for x in np.linspace(-half, half, 9):
        for y in np.linspace(-half, half, 9):
            col, row = ground.local_to_pixel(x, y)
            total += 1
            if 0 <= col < width and 0 <= row < height:
                inside += 1
    return inside / total


# --------------------------------------------------------------------------- #
# OBJ helpers
# --------------------------------------------------------------------------- #


def _tile_obj(tile_m: float) -> str:
    half = tile_m / 2.0
    return "\n".join(
        [
            "# ground tile quad, UV (0,0) = south-west, v grows north",
            f"v {-half:.4f} {-half:.4f} 0.0",
            f"v {half:.4f} {-half:.4f} 0.0",
            f"v {half:.4f} {half:.4f} 0.0",
            f"v {-half:.4f} {half:.4f} 0.0",
            "vt 0 0",
            "vt 1 0",
            "vt 1 1",
            "vt 0 1",
            "vn 0 0 1",
            "f 1/1/1 2/2/1 3/3/1",
            "f 1/1/1 3/3/1 4/4/1",
            "",
        ]
    )


def _triangulate_convex(points: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Fan triangulation; valid for convex rings (we hull the footprint first)."""
    return [(0, index, index + 1) for index in range(1, len(points) - 1)]


def _prism_vertices(ring: list[tuple[float, float]], height: float) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[float, float, float]]]:
    """Return (vertices, faces, face_normals) for one extruded footprint."""
    count = len(ring)
    bottom = [(x, y, 0.0) for x, y in ring]
    top = [(x, y, height) for x, y in ring]
    vertices = np.array(bottom + top, dtype=np.float64)

    faces: list[tuple[int, int, int]] = []
    normals: list[tuple[float, float, float]] = []

    for index in range(count):
        a = index
        b = (index + 1) % count
        faces.append((a, b, a + count))
        faces.append((b, b + count, a + count))
        dx = ring[b][0] - ring[a][0]
        dy = ring[b][1] - ring[a][1]
        length = math.hypot(dx, dy) or 1.0
        normals.append((dy / length, -dx / length, 0.0))
        normals.append((dy / length, -dx / length, 0.0))

    for triangle in _triangulate_convex(ring):
        faces.append((triangle[0] + count, triangle[1] + count, triangle[2] + count))
        normals.append((0.0, 0.0, 1.0))
    return vertices, faces, normals


def _write_obj(path: Path, groups: list[tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[float, float, float]]]]) -> None:
    lines = ["# extruded building prisms, ENU metres, z up"]
    vertex_offset = 1  # OBJ indices are 1-based
    normal_offset = 1
    for vertices, faces, normals in groups:
        for x, y, z in vertices:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        for nx, ny, nz in normals:
            lines.append(f"vn {nx:.4f} {ny:.4f} {nz:.4f}")
        for index, face in enumerate(faces):
            a, b, c = (value + vertex_offset for value in face)
            normal_index = normal_offset + min(index, len(normals) - 1)
            lines.append(f"f {a}//{normal_index} {b}//{normal_index} {c}//{normal_index}")
        vertex_offset += len(vertices)
        normal_offset += len(normals)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# buildings
# --------------------------------------------------------------------------- #


def _load_geojson(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") == "FeatureCollection":
        return [feature for feature in data.get("features", []) if feature.get("geometry")]
    if data.get("type") == "Feature":
        return [data]
    raise ValueError(f"{path}: expected a GeoJSON FeatureCollection")


def _feature_rings(geometry: dict) -> list[list[tuple[float, float]]]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if kind == "Polygon":
        return [[(float(p[0]), float(p[1])) for p in ring] for ring in coordinates[:1]]
    if kind == "MultiPolygon":
        return [
            [(float(p[0]), float(p[1])) for p in polygon[0]]
            for polygon in coordinates
            if polygon
        ]
    return []


def _feature_height(properties: dict) -> float:
    for key in ("height", "building:height", "building_height"):
        value = properties.get(key)
        if value is not None:
            try:
                return max(2.0, float(str(value).replace("m", "").strip()))
            except ValueError:
                pass
    levels = properties.get("building:levels") or properties.get("levels")
    if levels is not None:
        try:
            return max(2.0, float(levels) * LEVEL_HEIGHT_M)
        except ValueError:
            pass
    return DEFAULT_HEIGHT_M


def build_buildings(
    features: list[dict],
    origin: tuple[float, float],
    extent_m: float,
    max_buildings: int,
) -> tuple[list[tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[float, float, float]]]], dict]:
    from pyproj import Transformer
    from shapely.geometry import Polygon

    lat, lon = origin
    to_enu = Transformer.from_crs("EPSG:4326", local_crs(lat, lon), always_xy=True)
    half = extent_m / 2.0

    groups = []
    stats = {"features": len(features), "used": 0, "skipped_area": 0, "skipped_outside": 0, "heights": []}

    for feature in features:
        properties = feature.get("properties") or {}
        height = _feature_height(properties)
        for ring in _feature_rings(feature.get("geometry") or {}):
            if len(ring) < 3:
                continue
            points = [to_enu.transform(lon_value, lat_value) for lon_value, lat_value in ring]
            try:
                polygon = Polygon(points)
            except Exception:
                continue
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area < 4.0:
                stats["skipped_area"] += 1
                continue
            hull = polygon.convex_hull
            min_x, min_y, max_x, max_y = hull.bounds
            if max_x < -half or min_x > half or max_y < -half or min_y > half:
                stats["skipped_outside"] += 1
                continue
            # clip to the world so the mesh does not explode
            ring_points = list(hull.exterior.coords)[:-1]
            ring_points = [(min(max(x, -half), half), min(max(y, -half), half)) for x, y in ring_points]
            deduped: list[tuple[float, float]] = []
            for point in ring_points:
                if not deduped or math.dist(point, deduped[-1]) > 0.05:
                    deduped.append(point)
            if len(deduped) < 3:
                continue
            vertices, faces, normals = _prism_vertices(deduped, height)
            groups.append((vertices, faces, normals))
            stats["used"] += 1
            stats["heights"].append(height)
            if len(groups) >= max_buildings:
                stats["truncated"] = True
                return groups, stats
    return groups, stats


def overpass_buildings(center: tuple[float, float], extent_m: float, timeout: float = 60.0) -> dict:
    """Fetch building footprints around ``center`` from Overpass and return GeoJSON."""
    import urllib.parse
    import urllib.request

    lat, lon = center
    dlat = extent_m / 2.0 / 111_320.0 * 1.3
    dlon = extent_m / 2.0 / (111_320.0 * max(0.1, math.cos(math.radians(lat)))) * 1.3
    query = (
        "[out:json][timeout:{timeout}];"
        "way[building]({south},{west},{north},{east});"
        "out geom;"
    ).format(
        timeout=int(timeout),
        south=lat - dlat,
        west=lon - dlon,
        north=lat + dlat,
        east=lon + dlon,
    )
    request = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "drone-vla-make-geo-world/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout + 15) as response:
        payload = json.loads(response.read().decode())

    features = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry") or []
        if len(geometry) < 3:
            continue
        ring = [[node["lon"], node["lat"]] for node in geometry]
        ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "properties": element.get("tags") or {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


# --------------------------------------------------------------------------- #
# SDF
# --------------------------------------------------------------------------- #


def ground_model(name: str, extent_m: float, tiles: list[tuple[int, int]], tile_m: float) -> str:
    visual_blocks = []
    for index_x, index_y in tiles:
        x = (index_x + 0.5) * tile_m - extent_m / 2.0
        y = (index_y + 0.5) * tile_m - extent_m / 2.0
        visual_blocks.append(
            f"""      <visual name="tile_{index_x}_{index_y}">
        <pose>{x:.4f} {y:.4f} 0 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh><uri>model://geo_{name}/meshes/tile.obj</uri></mesh>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
          <pbr>
            <metal>
              <albedo_map>model://geo_{name}/textures/tile_{index_x}_{index_y}.png</albedo_map>
              <roughness>0.95</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>"""
        )
    return f"""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="geo_{name}">
    <static>true</static>
    <link name="link">
      <collision name="ground">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>{extent_m:.3f} {extent_m:.3f}</size>
          </plane>
        </geometry>
      </collision>
{chr(10).join(visual_blocks)}
    </link>
  </model>
</sdf>
"""


def buildings_model(name: str) -> str:
    return f"""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="geo_{name}_buildings">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <mesh><uri>model://geo_{name}_buildings/meshes/buildings.obj</uri></mesh>
        </geometry>
      </collision>
      <visual name="visual">
        <cast_shadows>true</cast_shadows>
        <geometry>
          <mesh><uri>model://geo_{name}_buildings/meshes/buildings.obj</uri></mesh>
        </geometry>
        <material>
          <ambient>0.62 0.62 0.64 1</ambient>
          <diffuse>0.62 0.62 0.64 1</diffuse>
          <specular>0.08 0.08 0.08 1</specular>
          <pbr>
            <metal>
              <albedo>0.62 0.62 0.64 1</albedo>
              <roughness>0.8</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


MODEL_CONFIG = """\
<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>drone-vla</name></author>
  <description>{description}</description>
</model>
"""


def world_sdf(
    name: str,
    origin: tuple[float, float],
    vehicle_uri: str,
    spawn: tuple[float, float, float],
    spawn_heading_deg: float,
    has_buildings: bool,
    spawn_name: str = "iris",
) -> str:
    yaw = math.radians(90.0 - spawn_heading_deg)  # ENU heading -> SDF yaw (CCW from +x)
    buildings = (
        f"""
    <include>
      <uri>model://geo_{name}_buildings</uri>
      <name>buildings</name>
      <pose>0 0 0 0 0 0</pose>
    </include>"""
        if has_buildings
        else ""
    )
    return f"""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="geo_{name}">
    <!-- Free-running physics: lockstep freezes this world (see STATUS_2026-11-06 §8). -->
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer">
      <use_units_gauss>false</use_units_gauss>
    </plugin>

    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.55 0.7 0.9 1</background>
      <grid>false</grid>
      <sky><clouds><speed>12</speed></clouds></sky>
      <shadows>true</shadows>
    </scene>
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 500 0 0 0</pose>
      <intensity>1</intensity>
      <direction>0.3 0.2 -1</direction>
      <diffuse>0.95 0.95 0.95 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation><range>2000</range><constant>1</constant><linear>0</linear><quadratic>0</quadratic></attenuation>
    </light>

    <include>
      <uri>model://geo_{name}</uri>
      <name>ground</name>
      <pose>0 0 0 0 0 0</pose>
    </include>{buildings}

    <include>
      <uri>{vehicle_uri}</uri>
      <name>{spawn_name}</name>
      <pose>{spawn[0]:.3f} {spawn[1]:.3f} {spawn[2]:.3f} 0 0 {yaw:.6f}</pose>
    </include>

    <!-- (0, 0) in the world is this coordinate; x = East, y = North. -->
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{origin[0]:.10f}</latitude_deg>
      <longitude_deg>{origin[1]:.10f}</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
  </world>
</sdf>
"""


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def build_world(args) -> dict:
    name = args.name
    out = Path(args.out) / name
    models = out / "models"
    worlds = out / "worlds"
    if out.exists() and args.clean:
        shutil.rmtree(out)
    models.mkdir(parents=True, exist_ok=True)
    worlds.mkdir(parents=True, exist_ok=True)

    origin = None
    if args.center_lat is not None and args.center_lon is not None:
        origin = (args.center_lat, args.center_lon)
    ground = GroundMap.from_geotiff(args.ortho, origin=origin, max_pixels=args.max_pixels)
    origin = (ground.origin_lat, ground.origin_lon)

    tile_m = float(args.tile_m)
    count = max(1, int(math.ceil(args.extent_m / tile_m)))
    extent = count * tile_m
    if abs(extent - args.extent_m) > 1e-6:
        print(f"extent rounded up to {extent:.0f} m ({count}x{count} tiles of {tile_m:.0f} m)")

    coverage = coverage_fraction(ground, extent)
    if coverage < 0.999:
        print(f"WARNING: only {coverage * 100:.1f}% of the {extent:.0f} m square is inside {args.ortho}")

    model_dir = models / f"geo_{name}"
    (model_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (model_dir / "textures").mkdir(parents=True, exist_ok=True)
    quad_m = tile_m + 2 * TILE_PAD_PX * (tile_m / float(args.tile_px))
    (model_dir / "meshes" / "tile.obj").write_text(_tile_obj(quad_m), encoding="utf-8")

    tiles = []
    for index_x in range(count):
        for index_y in range(count):
            x0 = index_x * tile_m - extent / 2.0
            y0 = index_y * tile_m - extent / 2.0
            tile = extract_tile(ground, x0, y0, tile_m, args.tile_px, args.flip_v)
            cv2.imwrite(str(model_dir / "textures" / f"tile_{index_x}_{index_y}.png"), tile)
            tiles.append((index_x, index_y))

    (model_dir / "model.sdf").write_text(ground_model(name, extent, tiles, tile_m), encoding="utf-8")
    (model_dir / "model.config").write_text(
        MODEL_CONFIG.format(name=f"geo_{name}", description=f"Orthophoto ground, {extent:.0f} m, {tile_m:.0f} m tiles."),
        encoding="utf-8",
    )

    building_stats = {"used": 0}
    has_buildings = False
    if args.buildings or args.osm:
        if args.osm:
            print("fetching buildings from Overpass ...")
            features = overpass_buildings(origin, extent).get("features", [])
        else:
            features = _load_geojson(Path(args.buildings))
        groups, building_stats = build_buildings(features, origin, extent, args.max_buildings)
        if groups:
            has_buildings = True
            building_dir = models / f"geo_{name}_buildings"
            (building_dir / "meshes").mkdir(parents=True, exist_ok=True)
            _write_obj(building_dir / "meshes" / "buildings.obj", groups)
            (building_dir / "model.sdf").write_text(buildings_model(name), encoding="utf-8")
            (building_dir / "model.config").write_text(
                MODEL_CONFIG.format(
                    name=f"geo_{name}_buildings",
                    description=f"{len(groups)} extruded OSM footprints.",
                ),
                encoding="utf-8",
            )

    world_path = worlds / f"{name}.sdf"
    spawn_name = args.spawn_name or args.vehicle.rstrip("/").rsplit("/", 1)[-1]
    world_path.write_text(
        world_sdf(
            name,
            origin,
            args.vehicle,
            (args.spawn_x, args.spawn_y, args.spawn_z),
            args.spawn_heading,
            has_buildings,
            spawn_name=spawn_name,
        ),
        encoding="utf-8",
    )

    # `iris_down_cam` includes model://iris_with_standoffs, which ships with
    # ardupilot_gazebo, so that tree has to be on the path as well.
    search_paths = [models.resolve(), REPO / "sim/ardupilot/models"]
    search_paths += [Path(item) for item in args.model_path]
    home_models = Path.home() / "ardupilot_gazebo/models"
    if home_models.is_dir():
        search_paths.append(home_models)
    joined = ":".join(str(path) for path in search_paths)
    (out / "env.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# source this before `gz sim` so model:// URIs resolve\n"
        f"export GZ_SIM_RESOURCE_PATH={joined}"
        ':${GZ_SIM_RESOURCE_PATH:-}\n',
        encoding="utf-8",
    )

    meta = {
        "name": name,
        "world": str(world_path),
        "ortho": str(Path(args.ortho).resolve()),
        "origin": {"lat": origin[0], "lon": origin[1], "crs": local_crs(*origin)},
        "extent_m": extent,
        "tile_m": tile_m,
        "tile_px": args.tile_px,
        "ground_gsd_m": tile_m / float(args.tile_px),
        "source_gsd_m": ground.gsd_m,
        "coverage": round(coverage, 4),
        "flip_v": bool(args.flip_v),
        "buildings": {"enabled": has_buildings, **building_stats},
        "spawn": {
            "name": spawn_name,
            "model": args.vehicle,
            "x_m": args.spawn_x,
            "y_m": args.spawn_y,
            "z_m": args.spawn_z,
            "heading_deg": args.spawn_heading,
        },
        "world_convention": "ENU: x=East, y=North, z=Up; (0,0) = origin lat/lon",
    }
    meta["buildings"].pop("heights", None)
    meta["buildings"]["count"] = meta["buildings"].pop("used", 0)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ortho", required=True, help="georeferenced GeoTIFF")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default="outputs/geo")
    parser.add_argument("--extent-m", type=float, default=600.0)
    parser.add_argument("--tile-m", type=float, default=200.0)
    parser.add_argument("--tile-px", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=40_000_000)
    parser.add_argument("--center-lat", type=float, default=None)
    parser.add_argument("--center-lon", type=float, default=None)
    parser.add_argument("--buildings", default=None, help="GeoJSON footprints (lon/lat polygons)")
    parser.add_argument("--osm", action="store_true", help="fetch footprints from Overpass instead")
    parser.add_argument("--max-buildings", type=int, default=4000)
    parser.add_argument("--vehicle", default="model://iris_down_cam")
    parser.add_argument(
        "--spawn-name",
        default=None,
        help="entity name inside the world (default: the model URI basename); "
        "the set_pose bridge needs this exact name",
    )
    parser.add_argument("--model-path", action="append", default=[], help="extra GZ model dirs for env.sh")
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=1.5)
    parser.add_argument("--spawn-heading", type=float, default=90.0, help="degrees clockwise from north")
    parser.add_argument("--flip-v", action="store_true", help="flip tile rows if textures appear upside-down")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args(argv)

    meta = build_world(args)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print()
    print(f"source {Path(meta['world']).parent.parent / 'env.sh'}")
    print(f"gz sim -r {meta['world']}")
    print()
    print("verify the world is localizable before flying it:")
    print(f"  python3 scripts/check_geo_world.py --run {Path(args.out) / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
