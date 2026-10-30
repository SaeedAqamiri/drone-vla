from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class SizingInput:
    distance_km: float = 5000.0
    corridor_km: float = 2.0
    altitude_m: float = 200.0
    fov_deg: float = 73.0
    aspect: float = 1.5
    speed_kmh: float = 180.0
    camera_fps: float = 10.0
    keyframe_hz: float = 1.0
    match_every_m: float = 500.0
    satellite_gsd_m: float = 0.5
    drone_pixels_wide: int = 4000
    tile_px: int = 1024
    gen_seconds: float = 40.0
    api_seconds: float = 1.0
    api_cost_per_image: float = 0.009
    gpu_count: int = 1
    match_seconds: float = 0.3
    frame_mb: float = 1.0
    generated_tile_mb: float = 1.5
    compression_ratio: float = 4.0


def compute(inp: SizingInput) -> dict:
    footprint_w = 2.0 * inp.altitude_m * math.tan(math.radians(inp.fov_deg) / 2.0)
    footprint_h = footprint_w / inp.aspect
    footprint_area_m2 = footprint_w * footprint_h
    drone_gsd_m = footprint_w / float(inp.drone_pixels_wide)

    corridor_area_km2 = inp.distance_km * inp.corridor_km
    corridor_area_m2 = corridor_area_km2 * 1_000_000.0

    tile_ground_m = inp.tile_px * inp.satellite_gsd_m
    tile_area_km2 = (tile_ground_m / 1000.0) ** 2
    tiles = int(math.ceil(corridor_area_km2 / tile_area_km2))

    satellite_px = corridor_area_m2 / (inp.satellite_gsd_m**2)
    satellite_raw_gb = satellite_px * 3.0 / 1e9
    satellite_stored_gb = satellite_raw_gb / max(inp.compression_ratio, 1.0)

    mission_hours = inp.distance_km / inp.speed_kmh
    mission_seconds = mission_hours * 3600.0
    speed_mps = inp.speed_kmh / 3.6
    spacing_m = speed_mps / max(inp.camera_fps, 1e-9)
    overlap = 1.0 - spacing_m / footprint_w
    frames = int(round(mission_seconds * inp.camera_fps))
    keyframes = int(round(mission_seconds * inp.keyframe_hz))
    keyframe_storage_gb = keyframes * inp.frame_mb / 1000.0

    match_queries = int(math.ceil(inp.distance_km * 1000.0 / inp.match_every_m))
    match_hours = match_queries * inp.match_seconds / 3600.0

    gen_hours_one = tiles * inp.gen_seconds / 3600.0
    gen_hours_fleet = gen_hours_one / max(inp.gpu_count, 1)
    api_hours = tiles * inp.api_seconds / 3600.0
    api_cost = tiles * inp.api_cost_per_image
    generated_storage_gb = tiles * inp.generated_tile_mb / 1000.0

    route_footprints = inp.distance_km * 1000.0 / footprint_w

    return {
        "input": asdict(inp),
        "drone": {
            "footprint_m": [round(footprint_w, 1), round(footprint_h, 1)],
            "footprint_area_m2": round(footprint_area_m2, 1),
            "footprint_gsd_cm": round(drone_gsd_m * 100.0, 2),
            "speed_mps": round(speed_mps, 2),
            "frame_spacing_m": round(spacing_m, 2),
            "frame_overlap": round(overlap, 4),
            "route_footprints": int(round(route_footprints)),
        },
        "mission": {
            "distance_km": inp.distance_km,
            "hours": round(mission_hours, 2),
            "frames_at_camera_fps": frames,
            "keyframes_at_keyframe_hz": keyframes,
            "keyframe_storage_gb": round(keyframe_storage_gb, 2),
        },
        "satellite": {
            "corridor_area_km2": round(corridor_area_km2, 1),
            "gsd_m": inp.satellite_gsd_m,
            "tile_ground_m": round(tile_ground_m, 1),
            "tiles": tiles,
            "raw_gb": round(satellite_raw_gb, 1),
            "stored_gb": round(satellite_stored_gb, 1),
        },
        "generation": {
            "tiles": tiles,
            "one_gpu_hours": round(gen_hours_one, 1),
            "one_gpu_days": round(gen_hours_one / 24.0, 1),
            "fleet_hours": round(gen_hours_fleet, 1),
            "api_hours": round(api_hours, 2),
            "api_cost_usd": round(api_cost, 2),
            "generated_storage_gb": round(generated_storage_gb, 1),
        },
        "matching": {
            "queries": match_queries,
            "hours_at_assumed_cost": round(match_hours, 2),
            "share_of_mission": round(match_hours / max(mission_hours, 1e-9), 4),
        },
    }


def format_report(result: dict) -> str:
    drone = result["drone"]
    mission = result["mission"]
    satellite = result["satellite"]
    generation = result["generation"]
    matching = result["matching"]
    lines = [
        f"footprint: {drone['footprint_m'][0]} x {drone['footprint_m'][1]} m "
        f"({drone['footprint_area_m2']} m^2), drone GSD {drone['footprint_gsd_cm']} cm/px",
        f"mission: {mission['distance_km']} km in {mission['hours']} h, "
        f"{mission['frames_at_camera_fps']} frames at camera fps, "
        f"{mission['keyframes_at_keyframe_hz']} keyframes "
        f"({mission['keyframe_storage_gb']} GB at {result['input']['frame_mb']} MB/frame)",
        f"satellite corridor: {satellite['corridor_area_km2']} km^2, "
        f"{satellite['tiles']} tiles of {satellite['tile_ground_m']} m at {satellite['gsd_m']} m GSD, "
        f"~{satellite['stored_gb']} GB stored",
        f"generation: {generation['one_gpu_hours']} h on one GPU "
        f"({generation['one_gpu_days']} days), {generation['fleet_hours']} h on "
        f"{result['input']['gpu_count']} GPUs, {generation['api_hours']} h via API, "
        f"${generation['api_cost_usd']}, {generation['generated_storage_gb']} GB",
        f"matching: {matching['queries']} queries, {matching['hours_at_assumed_cost']} h total, "
        f"{matching['share_of_mission'] * 100:.2f}% of mission time",
    ]
    return "\n".join(lines)
