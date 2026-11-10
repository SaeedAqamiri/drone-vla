#!/usr/bin/env python3
"""Derive a downward-looking camera variant of the ArduPilot ``iris`` model.

``ardupilot_gazebo`` ships ``iris_with_ardupilot`` with an IMU only, so the
SITL flight produces no images. This script copies that model into the repo and
appends a fixed camera link that points straight down -- the same viewpoint the
PX4 ``x500_mono_cam_down`` episodes were recorded from.

Usage:
    python3 scripts/make_apm_cam_model.py            # writes sim/ardupilot/models/iris_down_cam
    python3 scripts/make_apm_cam_model.py --src /path/to/iris_with_ardupilot/model.sdf
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _default_src() -> Path:
    """Locate the installed ``iris_with_ardupilot`` model."""
    candidates = [
        Path.home() / "ardupilot_gazebo/models/iris_with_ardupilot",
        Path("/home/drone/ardupilot_gazebo/models/iris_with_ardupilot"),
        Path("/usr/local/share/ardupilot_gazebo/models/iris_with_ardupilot"),
    ]
    for candidate in candidates:
        if (candidate / "model.sdf").is_file():
            return candidate
    return candidates[0]


CAM_BLOCK = """\
    <!-- drone-vla: downward camera for VLA episodes -->
    <link name="cam_link">
      <!-- iris landing gear tips sit at z=-0.11, so the camera hangs below them
           and the frame never intrudes on the downward view -->
      <pose>0 0 -0.14 0 0 0</pose>
      <inertial>
        <mass>0.02</mass>
        <inertia>
          <ixx>1e-5</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>1e-5</iyy><iyz>0</iyz>
          <izz>1e-5</izz>
        </inertia>
      </inertial>
      <sensor name="camera" type="camera">
        <!-- camera looks along +X of its own frame; pitch +90 deg aims it down -->
        <pose>0 0 0 0 1.5708 0</pose>
        <camera>
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>512</width>
            <height>384</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.05</near>
            <far>120</far>
          </clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>15</update_rate>
        <topic>camera/image</topic>
        <visualize>false</visualize>
      </sensor>
    </link>

    <joint name="cam_joint" type="fixed">
      <parent>iris_with_standoffs::base_link</parent>
      <child>cam_link</child>
    </joint>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=None, help="source model directory")
    p.add_argument("--dest", type=Path, default=REPO / "sim/ardupilot/models/iris_down_cam")
    p.add_argument("--name", default="iris_down_cam")
    args = p.parse_args()

    source_sdf = (args.src or _default_src()) / "model.sdf"
    if not source_sdf.is_file():
        raise SystemExit(f"missing {source_sdf} (is ardupilot_gazebo installed?)")

    text = source_sdf.read_text(encoding="utf-8")
    if "<model name=" not in text:
        raise SystemExit(f"{source_sdf} is not an SDF model")

    text = re.sub(r"<model name=['\"][^'\"]+['\"]>", f'<model name="{args.name}">', text, count=1)
    text = text.replace("</model>", CAM_BLOCK + "  </model>", 1)

    # ArduPilotPlugin's lockstep stops the world whenever SITL pauses, which on
    # this box leaves the vehicle frozen mid-air after the initial drop. Let the
    # physics run free; the plugin still exchanges FDM packets each cycle.
    text, replaced = re.subn(
        r"<lock_step>\s*1\s*</lock_step>", "<lock_step>0</lock_step>", text
    )
    if not replaced:
        print("warning: no <lock_step>1</lock_step> found; left as is")

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "model.sdf").write_text(text, encoding="utf-8")

    config = (args.src or _default_src()) / "model.config"
    if config.is_file():
        shutil.copyfile(config, args.dest / "model.config")

    print(f"wrote {args.dest / 'model.sdf'}")
    print("sensor topic: /world/<world>/model/<model>/link/cam_link/sensor/camera/image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
