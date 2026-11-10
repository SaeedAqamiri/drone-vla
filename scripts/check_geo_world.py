#!/usr/bin/env python3
"""Prove that a generated Gazebo world is actually localizable.

`make_geo_world.py` can produce a world that looks fine but is georeferenced
wrong (flipped tiles, swapped axes, bad CRS fit). Those bugs are invisible in a
preview and fatal for map matching. This script closes the loop:

1. writes a static downward ``probe_cam`` model and a probe copy of the world,
2. runs it headless in Gazebo and captures one frame,
3. renders the *expected* frame from the same GeoTIFF at the same pose,
4. matches the two and checks scale, rotation, coverage and position offset.

A pass means: the simulator's downward camera shows the same map, at the same
orientation and georeference, as the localizer's reference imagery.

    python3 scripts/check_geo_world.py --run outputs/geo/site1 --alt 250
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from sat2aerial.matching import match_images  # noqa: E402
from sat2aerial.orthofly import Camera, GroundMap, Pose  # noqa: E402

PW, PH, HFOV = 512, 384, 60.0

PROBE_SDF = """\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="probe_cam">
    <static>true</static>
    <link name="link">
      <sensor name="camera" type="camera">
        <!-- same convention as sim/ardupilot/models/iris_down_cam -->
        <pose>0 0 0 0 1.5708 0</pose>
        <camera>
          <horizontal_fov>{hfov_rad:.6f}</horizontal_fov>
          <image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>
          <clip><near>0.05</near><far>2000</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <topic>probe/image</topic>
        <visualize>false</visualize>
      </sensor>
    </link>
  </model>
</sdf>
"""

PROBE_CONFIG = """\
<?xml version="1.0"?>
<model>
  <name>probe_cam</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
</model>
"""


def _write_probe(run: Path, name: str, origin: tuple[float, float], pose: tuple[float, float, float], heading: float, has_buildings: bool) -> Path:
    import make_geo_world as geo

    model_dir = run / "models" / "probe_cam"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.sdf").write_text(
        PROBE_SDF.format(hfov_rad=math.radians(HFOV), width=PW, height=PH), encoding="utf-8"
    )
    (model_dir / "model.config").write_text(PROBE_CONFIG, encoding="utf-8")

    world = geo.world_sdf(
        name, origin, "model://probe_cam", pose, heading, has_buildings, spawn_name="probe_cam"
    )
    world_path = run / "worlds" / f"{name}_probe.sdf"
    world_path.write_text(world, encoding="utf-8")
    return world_path


def _env(run: Path, partition: str) -> dict:
    env = dict(os.environ)
    paths = [str((run / "models").resolve()), str((REPO / "sim/ardupilot/models"))]
    home_models = Path.home() / "ardupilot_gazebo/models"
    if home_models.is_dir():
        paths.append(str(home_models))
    existing = env.get("GZ_SIM_RESOURCE_PATH")
    if existing:
        paths.append(existing)
    env["GZ_SIM_RESOURCE_PATH"] = ":".join(paths)
    env["GZ_PARTITION"] = partition
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    return env


def _capture(world_path: Path, out_dir: Path, env: dict, timeout: float, topic: str, python: str) -> np.ndarray:
    npy = out_dir / "capture.npy"
    if npy.exists():
        npy.unlink()

    log = (out_dir / "gz.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["gz", "sim", "-s", "-r", "--headless-rendering", str(world_path)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        time.sleep(6.0)  # world load + first render
        helper = Path(__file__).resolve().parent / "gz_capture_once.py"
        last_error = ""
        for attempt in range(2):
            result = subprocess.run(
                [python, str(helper), "--topic", topic, "--out", str(npy), "--timeout", str(timeout)],
                capture_output=True,
                text=True,
                env=env,  # must share GZ_PARTITION with the server
            )
            if result.returncode == 0 and npy.exists():
                return np.load(npy)
            last_error = f"{result.stdout.strip()} {result.stderr.strip()}".strip()
            time.sleep(3.0)
        tail = (out_dir / "gz.log").read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        raise RuntimeError(f"capture failed: {last_error}\n" + "\n".join(tail))
    finally:
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        log.close()


def check(run: Path, x: float, y: float, alt: float, heading: float, timeout: float, python: str) -> int:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    name = meta["name"]
    origin = (meta["origin"]["lat"], meta["origin"]["lon"])

    out_dir = run / "check"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    world_path = _write_probe(run, name, origin, (x, y, alt), heading, bool(meta["buildings"]["enabled"]))
    env = _env(run, partition=f"chk_{name}_{os.getpid()}")
    print(f"world  {world_path}")
    print(f"pose   x={x} y={y} alt={alt} heading={heading}")

    frame = _capture(world_path, out_dir, env, timeout, "/probe/image", python)
    capture_path = out_dir / "capture.png"
    cv2.imwrite(str(capture_path), frame)

    ground = GroundMap.from_geotiff(meta["ortho"], origin=origin)
    camera = Camera(width=PW, height=PH, hfov_deg=HFOV)
    expected = ground.render(Pose(x=x, y=y, alt=alt, yaw_deg=heading), camera)
    expected_path = out_dir / "expected.png"
    cv2.imwrite(str(expected_path), expected)

    result, homography = match_images(expected_path, capture_path, mode="gray", detector="sift", max_dim=1024)
    offset = None
    if homography is not None:
        try:
            inverse = np.linalg.inv(homography)
            centre = inverse @ np.array([PW / 2.0, PH / 2.0, 1.0])
            centre = centre[:2] / centre[2]
            gsd = camera.gsd_m(alt)
            east = float(centre[0] - PW / 2.0) * gsd
            north = float(-(centre[1] - PH / 2.0)) * gsd
            offset = (east, north, math.hypot(east, north))
        except np.linalg.LinAlgError:
            offset = None

    verdict = {
        "inliers": result.inliers,
        "inlier_ratio": round(result.inlier_ratio, 4),
        "coverage": round(result.coverage, 4),
        "median_reproj_px": round(result.median_reproj, 3) if result.median_reproj is not None else None,
        "scale": round(result.scale, 5) if result.scale is not None else None,
        "rotation_deg": round(result.rotation_deg, 3) if result.rotation_deg is not None else None,
        "offset_m": None if offset is None else {"east": round(offset[0], 2), "north": round(offset[1], 2), "total": round(offset[2], 2)},
    }
    passed = (
        result.inliers >= 50
        and result.coverage >= 0.10
        and result.scale is not None
        and abs(result.scale - 1.0) <= 0.05
        and result.rotation_deg is not None
        and abs(result.rotation_deg) <= 3.0
        and (offset is None or offset[2] <= 3.0)
    )
    verdict["passed"] = bool(passed)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(json.dumps(verdict, indent=2))
    if passed:
        print("PASS — the world camera and the reference map agree.")
    else:
        print("FAIL — the world does not match the reference map.")
        if result.scale is not None and abs(result.scale - 1.0) > 0.2:
            print("  scale is far from 1: check --tile-px / --tile-m and the camera altitude.")
        if result.rotation_deg is not None and abs(result.rotation_deg) > 45:
            print("  rotation is far from 0: check --spawn-heading, or try make_geo_world --flip-v.")
        if result.inliers < 20:
            print("  few inliers: try --flip-v (upside-down ground texture) or a less uniform area.")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="output directory of make_geo_world")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--alt", type=float, default=None, help="defaults to half the world extent")
    parser.add_argument("--heading", type=float, default=0.0, help="degrees clockwise from north")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--gz-python", default="/usr/bin/python3", help="interpreter with gz.transport13")
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="only regenerate the probe world/model (used by the live loop), do not capture",
    )
    args = parser.parse_args(argv)

    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    alt = args.alt if args.alt is not None else max(60.0, meta["extent_m"] * 0.6)

    if args.write_only:
        world = _write_probe(
            run,
            meta["name"],
            (meta["origin"]["lat"], meta["origin"]["lon"]),
            (args.x, args.y, alt),
            args.heading,
            bool(meta["buildings"]["enabled"]),
        )
        print(world)
        return 0

    return check(run, args.x, args.y, alt, args.heading, args.timeout, args.gz_python)


if __name__ == "__main__":
    raise SystemExit(main())
