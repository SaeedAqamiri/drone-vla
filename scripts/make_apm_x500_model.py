#!/usr/bin/env python3
"""Build an ArduPilot-driven variant of the PX4 ``x500`` airframe.

Why this exists
---------------
The stock ArduPilot airframe (``iris``) gets its lift from a blade-element
aerodynamic model that produces no usable thrust in Gazebo Harmonic 8.15 with
the current ``ardupilot_gazebo`` plugin, so the vehicle never leaves the ground.
The PX4 ``x500`` airframe instead uses ``gz-sim-multicopter-motor-model-system``,
which turns a rotor speed into thrust with ``T = k * omega^2`` and is known to
fly on this exact setup.

This script composes a model that keeps x500's motor model and downward camera,
and adds ``ArduPilotPlugin`` **only for the sensor/FDM side** (no ``<control>``
blocks). Motor commands travel the other way, through
``scripts/apm_motor_bridge.py``:

    ArduPilot SERVO_OUTPUT_RAW  ->  <model>/command/motor_speed  ->  thrust

Usage:
    python3 scripts/make_apm_x500_model.py
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

APM_PLUGIN = """\
    <!-- Sensors + JSON FDM only. Motor commands are injected by
         scripts/apm_motor_bridge.py, so there are deliberately no <control>
         blocks here: the x500 MotorModel owns thrust. -->
    <plugin name="ArduPilotPlugin"
      filename="ArduPilotPlugin">
      <fdm_addr>127.0.0.1</fdm_addr>
      <fdm_port_in>9002</fdm_port_in>
      <connectionTimeoutMaxCount>5</connectionTimeoutMaxCount>
      <!-- Lock step: the world advances once per ArduPilot servo packet, which
           keeps sensor/actuator timing deterministic. Free-running (0) makes
           the rate loop unstable on this airframe. -->
      <lock_step>1</lock_step>
      <no_time_sync>1</no_time_sync>
      <have_32_channels>0</have_32_channels>
      <modelXYZToAirplaneXForwardZDown degrees="true">0 0 0 180 0 0</modelXYZToAirplaneXForwardZDown>
      <gazeboXYZToNED degrees="true">0 0 0 180 0 90</gazeboXYZToNED>
      <imuName>ap_imu_link::ap_imu_sensor</imuName>
    </plugin>

"""

# ArduPilot assumes IMU samples are in its *body* frame:
#     x-forward, y-right, z-down   (FRD)
# Gazebo reports each sensor's data in that sensor's own frame, and x500's
# imu_sensor sits directly on base_link with no offset, i.e. FLU
# (y-left, z-up). Feeding FLU samples to a plugin that assumes FRD inverts roll
# and pitch feedback: the vehicle levels itself upside down (roll -> 180 deg
# about a second after take-off) instead of holding attitude.
#
# ardupilot_gazebo's working iris model solves this by giving its IMU sensor a
# 180 deg roll relative to the body. We do the same with a dedicated link rather
# than editing PX4's model: the plugin only reads the sensor named in <imuName>.
IMU_LINK = """\
    <!-- ArduPilot-facing IMU: 180 deg roll converts FLU (Gazebo) to FRD
         (ArduPilot). Without this the vehicle flips on take-off. -->
    <link name='ap_imu_link'>
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>0.01</mass>
        <inertia>
          <ixx>1e-6</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1e-6</iyy>
          <iyz>0</iyz>
          <izz>1e-6</izz>
        </inertia>
      </inertial>
      <sensor name="ap_imu_sensor" type="imu">
        <gz_frame_id>ap_imu_link</gz_frame_id>
        <pose degrees="true">0 0 0 180 0 0</pose>
        <always_on>1</always_on>
        <update_rate>250</update_rate>
      </sensor>
    </link>
    <joint name='ap_imu_joint' type='fixed'>
      <parent>base_link</parent>
      <child>ap_imu_link</child>
    </joint>

"""

CAMERA = """\
    <!-- camera: nose-mounted, 30 deg below the horizon.
         Two things matter here:
         * a nadir camera sees only ground texture, which is useless for a
           navigation/VLA dataset, so it is pitched 30 deg down to keep the
           horizon and the terrain ahead in view like a real FPV cam;
         * it sits at x={cam_x} m, i.e. beyond the front propeller disc (rotor
           tips reach x ~ 0.32 m). Mounted at x=0 the 99.7 deg horizontal FOV
           puts the drone's own propeller in the top of every frame. -->
    <include merge='true'>
      <uri>model://mono_cam</uri>
      <pose>{cam_x} 0 0 0 {cam_pitch} 0</pose>
      <name>mono_cam</name>
    </include>
    <joint name="CameraJoint" type="fixed">
      <parent>base_link</parent>
      <child>camera_link</child>
      <pose relative_to="base_link">{cam_x} 0 0 0 {cam_pitch} 0</pose>
    </joint>
"""

MODEL_CONFIG = """\
<?xml version="1.0"?>
<model>
  <name>x500_apm</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>drone-vla</name></author>
  <description>PX4 x500 airframe + ArduPilot sensor bridge + downward camera.</description>
</model>
"""


def find_models_dir() -> Path:
    candidates = [
        Path("/home/drone/PX4-Autopilot/Tools/simulation/gz/models"),
        Path.home() / "PX4-Autopilot/Tools/simulation/gz/models",
    ]
    for candidate in candidates:
        if (candidate / "x500/model.sdf").is_file():
            return candidate
    raise SystemExit(f"cannot find PX4 gz models in {[str(c) for c in candidates]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--dest", type=Path, default=REPO / "sim/ardupilot/models/x500_apm")
    p.add_argument("--name", default="x500_apm")
    p.add_argument("--skip-camera", action="store_true")
    p.add_argument(
        "--cam-depression",
        type=float,
        default=30.0,
        help="degrees below the horizon (90 = straight down)",
    )
    p.add_argument(
        "--cam-x",
        type=float,
        default=0.35,
        help="camera x offset in metres; must clear the front propeller disc (~0.32)",
    )
    args = p.parse_args()

    models = args.models_dir or find_models_dir()
    src = models / "x500/model.sdf"
    if not src.is_file():
        raise SystemExit(f"missing {src}")

    text = src.read_text(encoding="utf-8")
    text = re.sub(r"<model name=['\"][^'\"]+['\"]>", f'<model name="{args.name}">', text, count=1)

    block = APM_PLUGIN + IMU_LINK
    if not args.skip_camera:
        # SDF pitch is measured from the horizon: 0 = level, pi/2 = straight
        # down. (Subtracting from pi/2 here is a classic sign error that turned
        # a "20 deg" camera into a 70 deg near-nadir one.)
        cam_pitch = math.radians(args.cam_depression)
        block += CAMERA.format(
            cam_pitch=f"{cam_pitch:.6f}", cam_x=f"{args.cam_x:.3f}"
        )
    if "</model>" not in text:
        raise SystemExit(f"{src} is not an SDF model")
    text = text.replace("</model>", block + "  </model>", 1)

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "model.sdf").write_text(text, encoding="utf-8")

    config = models / "x500/model.config"
    if config.is_file():
        # keep the PX4 metadata file around for reference/debugging
        shutil.copyfile(config, args.dest / "model.config.px4")
    (args.dest / "model.config").write_text(MODEL_CONFIG, encoding="utf-8")

    print(f"wrote {args.dest / 'model.sdf'}")
    print("sensor FDM : ArduPilotPlugin -> udp 9002, imu ap_imu_link::ap_imu_sensor (FRD)")
    print(f"motor topic: /{args.name}/command/motor_speed (gz.msgs.Actuators)")
    if not args.skip_camera:
        print(f"camera     : model://mono_cam, {args.cam_depression:.0f} deg below horizon")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
