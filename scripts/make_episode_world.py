#!/usr/bin/env python3
"""Generate a light, textured episode world for ArduPilot SITL.

The baylands photogrammetry map runs at ~0.2x real time under llvmpipe (47 s of
sim time in 216 s of wall clock), which makes 300-step episodes meaningless.
This builds a 400x400 m ground plane with a tiled procedural grass texture
instead: one quad, ~1 MB of texture, and a real-time factor near 1.

Outputs (all under sim/ardupilot):
    models/textured_ground/{model.sdf,model.config,meshes/ground.obj,textures/ground.png}
    worlds/field_apm_cam.sdf

Usage:
    python3 scripts/make_episode_world.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SIM = REPO / "sim/ardupilot"
GROUND = SIM / "models/textured_ground"
TEXTURE_PX = 1024
TILE_M = 12.0          # metres per texture repeat
GROUND_M = 400.0       # world extent
SEED = 7


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal RGB PNG writer (avoids a Pillow dependency in this script)."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def make_texture(rng: np.random.Generator) -> np.ndarray:
    """Grass-like: low-frequency blotches + fine noise + a few dry patches."""
    coarse = rng.normal(size=(32, 32))
    coarse = np.repeat(np.repeat(coarse, TEXTURE_PX // 32, axis=0), TEXTURE_PX // 32, axis=1)

    mid = rng.normal(size=(128, 128))
    mid = np.repeat(np.repeat(mid, TEXTURE_PX // 128, axis=0), TEXTURE_PX // 128, axis=1)

    fine = rng.normal(size=(TEXTURE_PX, TEXTURE_PX))

    green = 96 + 18 * coarse + 8 * mid + 6 * fine
    red = 62 + 12 * coarse + 6 * mid + 5 * fine
    blue = 44 + 8 * coarse + 4 * mid + 4 * fine

    # scatter darker tufts and lighter dry patches
    tufts = rng.random((TEXTURE_PX, TEXTURE_PX)) < 0.02
    dry = rng.random((TEXTURE_PX, TEXTURE_PX)) < 0.008
    green = np.where(tufts, green * 0.55, green)
    red = np.where(tufts, red * 0.55, red)
    blue = np.where(tufts, blue * 0.55, blue)
    green = np.where(dry, green * 1.15, green)
    red = np.where(dry, red * 1.55, red)

    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_obj(path: Path, size_m: float, tiles: float) -> None:
    half = size_m / 2.0
    lines = [
        "# flat ground quad, UVs tile the texture",
        f"v {-half:.3f} {-half:.3f} 0.000",
        f"v {half:.3f} {-half:.3f} 0.000",
        f"v {half:.3f} {half:.3f} 0.000",
        f"v {-half:.3f} {half:.3f} 0.000",
        f"vt 0.000 0.000",
        f"vt {tiles:.3f} 0.000",
        f"vt {tiles:.3f} {tiles:.3f}",
        f"vt 0.000 {tiles:.3f}",
        "vn 0 0 1",
        "usemtl ground",
        "f 1/1/1 2/2/1 3/3/1",
        "f 1/1/1 3/3/1 4/4/1",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


MODEL_SDF = """\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="textured_ground">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>{size} {size}</size>
          </plane>
        </geometry>
      </collision>
      <visual name="visual">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh>
            <uri>model://textured_ground/meshes/ground.obj</uri>
          </mesh>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
          <pbr>
            <metal>
              <albedo_map>model://textured_ground/textures/ground.png</albedo_map>
              <roughness>0.95</roughness>
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
  <name>textured_ground</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>drone-vla</name></author>
  <description>Tiled procedural grass ground for SITL episodes.</description>
</model>
"""


def landmarks(n: int = 26, max_radius: float = 70.0) -> str:
    """Scatter simple trees so the camera has landmarks and optic flow.

    A flat, uniformly textured plane gives a navigation dataset almost nothing
    to look at: every frame is the same grass, and a forward camera sees no
    parallax. Trunks and canopies are cheap primitives (cylinder + sphere), so
    they cost little under llvmpipe while making the scene readable.
    """
    rng = np.random.default_rng(SEED + 1)
    out = []
    for i in range(n):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        dist = rng.uniform(6.0, max_radius)
        x, y = dist * np.cos(ang), dist * np.sin(ang)
        height = rng.uniform(2.0, 5.5)
        trunk_r = rng.uniform(0.15, 0.30)
        canopy_r = rng.uniform(0.9, 1.8)
        green = rng.uniform(0.20, 0.42)
        out.append(
            f"""    <model name="tree_{i}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} 0 0 0 0</pose>
      <link name="trunk">
        <pose>0 0 {height / 2:.3f} 0 0 0</pose>
        <visual name="bark">
          <geometry><cylinder><radius>{trunk_r:.3f}</radius><length>{height:.3f}</length></cylinder></geometry>
          <material>
            <ambient>0.28 0.18 0.10 1</ambient>
            <diffuse>0.36 0.23 0.13 1</diffuse>
          </material>
        </visual>
        <visual name="canopy">
          <pose>0 0 {height / 2 + canopy_r * 0.6:.3f} 0 0 0</pose>
          <geometry><sphere><radius>{canopy_r:.3f}</radius></sphere></geometry>
          <material>
            <ambient>{green * 0.7:.3f} {green:.3f} {green * 0.5:.3f} 1</ambient>
            <diffuse>{green * 0.8:.3f} {green + 0.12:.3f} {green * 0.55:.3f} 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""
        )
    return "".join(out)


def world(
    vehicle_uri: str = "model://iris_down_cam",
    spawn_z: float = 1.5,
    vehicle_name: str = "iris",
) -> str:
    return f"""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="field">
    <!-- Lock step: ``type="ignore"`` hands the stepping clock to the ArduPilot
         plugin, which advances the world once per SITL servo packet. The
         free-running alternative (``type="ode"`` + lock_step 0) leaves the two
         simulators unsynchronised and the rate loop cannot hold attitude. This
         is the same configuration ardupilot_gazebo's own worlds use. -->
    <physics name="1ms" type="ignore">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
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
      <use_earth_frame_ned>false</use_earth_frame_ned>
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
      <attenuation>
        <range>2000</range>
        <constant>1</constant>
        <linear>0</linear>
        <quadratic>0</quadratic>
      </attenuation>
    </light>

    <include>
      <uri>model://textured_ground</uri>
      <name>ground</name>
      <pose>0 0 0 0 0 0</pose>
    </include>

{landmarks()}
    <include>
      <uri>{vehicle_uri}</uri>
      <name>{vehicle_name}</name>
      <pose>0 0 {spawn_z} 0 0 0</pose>
    </include>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
  </world>
</sdf>
"""


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build the episode ground + worlds.")
    p.add_argument("--vehicle-uri", default=None,
                   help="extra world with this vehicle, e.g. model://x500_apm")
    p.add_argument("--vehicle-name", default="x500")
    p.add_argument("--spawn-z", type=float, default=0.5)
    p.add_argument("--out", default=None, help="output .sdf name for that world")
    args = p.parse_args(argv)

    rng = np.random.default_rng(SEED)
    (GROUND / "meshes").mkdir(parents=True, exist_ok=True)
    (GROUND / "textures").mkdir(parents=True, exist_ok=True)

    write_png(GROUND / "textures/ground.png", make_texture(rng))
    make_obj(GROUND / "meshes/ground.obj", GROUND_M, GROUND_M / TILE_M)
    (GROUND / "model.sdf").write_text(MODEL_SDF.format(size=GROUND_M), encoding="utf-8")
    (GROUND / "model.config").write_text(MODEL_CONFIG, encoding="utf-8")

    worlds = SIM / "worlds"
    worlds.mkdir(parents=True, exist_ok=True)
    (worlds / "field_apm_cam.sdf").write_text(world(), encoding="utf-8")
    print(f"world    {worlds / 'field_apm_cam.sdf'}")

    if args.vehicle_uri:
        name = args.out or f"field_{args.vehicle_name}.sdf"
        text = world(
            vehicle_uri=args.vehicle_uri,
            spawn_z=args.spawn_z,
            vehicle_name=args.vehicle_name,
        )
        (worlds / name).write_text(text, encoding="utf-8")
        print(f"world    {worlds / name}  ({args.vehicle_uri})")

    print(f"texture  {GROUND / 'textures/ground.png'}")
    print(f"mesh     {GROUND / 'meshes/ground.obj'}")
    print(f"tile     {TILE_M} m/repeat over {GROUND_M} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
