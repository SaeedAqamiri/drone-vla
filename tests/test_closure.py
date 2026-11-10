from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sat2aerial.closure import (
    ControllerConfig,
    LoopConfig,
    MapLocalizer,
    MavlinkControl,
    MavlinkSink,
    VISION_PARAMS,
    WaypointController,
    point_segment_distance,
    run_simulation,
)
from sat2aerial.orthofly import Camera, Effects, Pose, procedural_map

CLEAN = Effects(blur_sigma=0.0, noise_sigma=0.0, gamma_jitter=0.0, exposure_jitter=0.0, vignette=0.0, exposure_ms=0.0)


class LocalizerTests(unittest.TestCase):
    def test_recovers_position_from_offset_hint(self) -> None:
        ground = procedural_map(size=1024, gsd_m=0.5, seed=5)
        camera = Camera(width=512, height=512, hfov_deg=60.0)
        localizer = MapLocalizer(ground, camera, base_search_m=80.0)
        frame = ground.render(Pose(x=50.0, y=-30.0, alt=60.0, yaw_deg=15.0), camera)

        fix, homography = localizer.locate(frame, (75.0, -8.0), altitude=60.0)
        self.assertIsNotNone(homography)
        self.assertTrue(fix.accepted, fix.reason)
        self.assertLess(math.hypot(fix.x - 50.0, fix.y - (-30.0)), 3.0)
        self.assertGreater(fix.inliers, 20)

    def test_rejects_a_jump_beyond_the_search_radius(self) -> None:
        ground = procedural_map(size=1024, gsd_m=0.5, seed=6)
        camera = Camera(width=512, height=512, hfov_deg=60.0)
        localizer = MapLocalizer(ground, camera, base_search_m=15.0, max_search_m=15.0, growth_mps=0.0)
        frame = ground.render(Pose(x=200.0, y=150.0, alt=60.0), camera)

        # The frame is 250 m away from the prior, so the crop around the prior
        # cannot contain it; a wildly wrong homography must be rejected.
        fix, _ = localizer.locate(frame, (0.0, 0.0), altitude=60.0)
        self.assertFalse(fix.accepted)
        self.assertFalse(math.isfinite(fix.x) and abs(fix.x) > 500)

    def test_search_radius_grows_with_age(self) -> None:
        ground = procedural_map(size=512, gsd_m=0.5, seed=7)
        localizer = MapLocalizer(ground, Camera(320, 320, 60.0), base_search_m=30.0, max_search_m=120.0, growth_mps=10.0)
        self.assertAlmostEqual(localizer.search_radius(0.0), 30.0)
        self.assertAlmostEqual(localizer.search_radius(5.0), 80.0)
        self.assertAlmostEqual(localizer.search_radius(60.0), 120.0)


class ControllerTests(unittest.TestCase):
    def test_advances_waypoints_and_stops(self) -> None:
        controller = WaypointController([(10.0, 0.0), (20.0, 0.0)], ControllerConfig(reach_m=3.0, speed_mps=5.0))
        vx, vy = controller.command((0.0, 0.0))
        self.assertAlmostEqual(math.hypot(vx, vy), 5.0, places=6)
        self.assertAlmostEqual(vx, 5.0, places=6)

        controller.command((9.0, 0.0))  # inside the first waypoint
        self.assertEqual(controller.index, 1)
        controller.command((19.0, 0.0))
        self.assertTrue(controller.done)
        self.assertEqual(controller.command((19.0, 0.0)), (0.0, 0.0))


class MavlinkSinkTests(unittest.TestCase):
    def test_vision_message_is_encoded_in_ned(self) -> None:
        from sat2aerial.closure import MavlinkSink
        from pymavlink import mavutil

        sink = MavlinkSink(url=None)
        sink.send_vision(x_enu=10.0, y_enu=20.0, z_up=5.0, yaw_deg=90.0)
        self.assertEqual(sink.vision_sent, 1)
        x_n, y_e, z_d, yaw = sink.last_vision
        self.assertAlmostEqual(x_n, 20.0, places=6)   # north = ENU y
        self.assertAlmostEqual(y_e, 10.0, places=6)   # east  = ENU x
        self.assertAlmostEqual(z_d, -5.0, places=6)   # down  = -up
        self.assertAlmostEqual(yaw, math.pi / 2, places=6)

        payload = sink.buffer.getvalue()
        self.assertGreater(len(payload), 0)
        parsed = mavutil.mavlink.MAVLink(None)
        message = parsed.decode(bytearray(payload))
        self.assertEqual(message.get_type(), "VISION_POSITION_ESTIMATE")
        self.assertAlmostEqual(message.x, 20.0, places=4)
        self.assertAlmostEqual(message.y, 10.0, places=4)
        self.assertAlmostEqual(message.z, -5.0, places=4)

    def test_position_setpoint_uses_local_ned(self) -> None:
        from sat2aerial.closure import MavlinkSink
        from pymavlink import mavutil

        sink = MavlinkSink(url=None)
        sink.send_position(x_enu=30.0, y_enu=-40.0, z_up=60.0)
        payload = sink.buffer.getvalue()
        message = mavutil.mavlink.MAVLink(None).decode(bytearray(payload))
        self.assertEqual(message.get_type(), "SET_POSITION_TARGET_LOCAL_NED")
        self.assertAlmostEqual(message.x, -40.0, places=4)
        self.assertAlmostEqual(message.y, 30.0, places=4)
        self.assertAlmostEqual(message.z, -60.0, places=4)
        self.assertEqual(message.coordinate_frame, mavutil.mavlink.MAV_FRAME_LOCAL_NED)


class FlyThroughTests(unittest.TestCase):
    def test_segment_distance_catches_a_skipped_waypoint(self) -> None:
        # a 2 Hz localizer moves ~10 m per update at 5 m/s, so a 5 m reach
        # circle can be stepped clean over
        self.assertLess(point_segment_distance((0.0, 0.0), (-30.0, 1.0), (30.0, -1.0)), 2.0)
        self.assertAlmostEqual(point_segment_distance((0.0, 10.0), (-30.0, 0.0), (30.0, 0.0)), 10.0, places=6)
        # behind the start of the segment: distance to the endpoint
        self.assertAlmostEqual(point_segment_distance((-10.0, 0.0), (0.0, 0.0), (10.0, 0.0)), 10.0, places=6)
        # degenerate segment
        self.assertAlmostEqual(point_segment_distance((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)), 5.0, places=6)


class _FakeMaster:
    """Minimal stand-in for a mavutil connection, so the control sequence can
    be encoded and inspected without a vehicle."""

    def __init__(self, mav) -> None:
        self.mav = mav
        self.target_system = 1
        self.target_component = 1

    def recv_match(self, type=None, blocking=True, timeout=0):
        return None

    def motors_armed(self) -> bool:
        return False

    @property
    def flightmode(self) -> str:
        return "GUIDED"


class MavlinkControlTests(unittest.TestCase):
    def _control(self) -> tuple[MavlinkControl, MavlinkSink]:
        sink = MavlinkSink(url=None)
        sink.master = _FakeMaster(sink.mav)
        return MavlinkControl(sink), sink

    def _decode(self, payload: bytes) -> list[str]:
        from pymavlink import mavutil

        parser = mavutil.mavlink.MAVLink(None)
        parser.robust_parsing = True
        messages = []
        for byte in payload:
            message = parser.parse_char(bytes([byte]))
            if message is not None:
                messages.append(message)
        return messages

    def test_parses_back_into_the_commands_ardupilot_expects(self) -> None:
        from pymavlink import mavutil

        control, sink = self._control()
        control.set_params({"EK3_SRC1_POSXY": 6.0, "GPS1_TYPE": 0.0}, settle=0.0)
        control.send_origin(40.0155, -105.269, 0.0)
        control.set_guided()
        control.takeoff(100.0)

        messages = self._decode(sink.buffer.getvalue())
        kinds = [m.get_type() for m in messages]
        self.assertEqual(kinds.count("PARAM_SET"), 2)
        self.assertEqual(kinds.count("COMMAND_LONG"), 2)
        self.assertIn("SET_GPS_GLOBAL_ORIGIN", kinds)

        params = {m.param_id: m.param_value for m in messages if m.get_type() == "PARAM_SET"}
        self.assertEqual(params["EK3_SRC1_POSXY"], 6.0)
        self.assertEqual(params["GPS1_TYPE"], 0.0)
        # VISO_TYPE=1 is what makes ArduPilot accept VISION_POSITION_ESTIMATE at
        # all; without it the EKF has no position and arming fails with
        # "AHRS: waiting for home"
        self.assertEqual(VISION_PARAMS["VISO_TYPE"], 1.0)
        self.assertEqual(VISION_PARAMS["EK3_SRC1_POSXY"], 6.0)

        origin = next(m for m in messages if m.get_type() == "SET_GPS_GLOBAL_ORIGIN")
        self.assertEqual(origin.latitude, round(40.0155 * 1e7))
        self.assertEqual(origin.longitude, round(-105.269 * 1e7))

        commands = [m for m in messages if m.get_type() == "COMMAND_LONG"]
        self.assertEqual(commands[0].command, mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        self.assertEqual(commands[0].param1, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED)
        self.assertEqual(commands[0].param2, 4)  # ArduCopter GUIDED
        self.assertEqual(commands[1].command, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
        self.assertAlmostEqual(commands[1].param7, 100.0, places=4)

    def test_arm_sends_the_arm_bit(self) -> None:
        from pymavlink import mavutil

        control, sink = self._control()
        self.assertFalse(control.arm(timeout=0.1))
        messages = self._decode(sink.buffer.getvalue())
        arm = next(m for m in messages if m.get_type() == "COMMAND_LONG")
        self.assertEqual(arm.command, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        self.assertEqual(arm.param1, 1.0)

    def test_control_refuses_a_dry_run_sink(self) -> None:
        with self.assertRaises(RuntimeError):
            MavlinkControl(MavlinkSink(url=None))


class ClosedLoopTests(unittest.TestCase):
    def test_reaches_the_waypoint_using_only_map_matching(self) -> None:
        ground = procedural_map(size=1536, gsd_m=0.5, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_simulation(
                ground,
                LoopConfig(
                    out=Path(tmp),
                    camera=Camera(384, 384, 60.0),
                    altitude_m=60.0,
                    waypoints=[(80.0, 40.0)],
                    control_hz=20.0,
                    localize_hz=2.0,
                    duration_s=30.0,
                    effects=CLEAN,
                    keep_every=0,
                ),
            )
            self.assertTrue(summary["completed"], summary)
            self.assertLess(summary["final_distance_m"], 8.0)
            self.assertGreater(summary["fix_rate"], 0.8)
            self.assertIsNotNone(summary["error_m"]["median"])
            self.assertLess(summary["error_m"]["median"], 3.0)
            self.assertTrue((Path(tmp) / "trajectory.png").is_file())
            rows = (Path(tmp) / "trajectory.csv").read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(rows), 10)

    def test_survives_camera_dropout(self) -> None:
        ground = procedural_map(size=1536, gsd_m=0.5, seed=10)
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_simulation(
                ground,
                LoopConfig(
                    out=Path(tmp),
                    camera=Camera(384, 384, 60.0),
                    altitude_m=60.0,
                    waypoints=[(60.0, 0.0)],
                    duration_s=40.0,
                    dropout=0.35,
                    effects=CLEAN,
                    seed=3,
                ),
            )
            # dead reckoning keeps it going; complete is the point, not accuracy
            self.assertTrue(summary["completed"], summary)
            self.assertLess(summary["fix_rate"], 0.95)
            self.assertGreater(summary["match_rate"], 0.8)
            self.assertLess(summary["final_distance_m"], 12.0)

    def test_summary_is_json_serialisable(self) -> None:
        ground = procedural_map(size=768, gsd_m=0.5, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_simulation(
                ground,
                LoopConfig(out=Path(tmp), camera=Camera(256, 256, 60.0), altitude_m=60.0, waypoints=[(30.0, 0.0)], duration_s=15.0, effects=CLEAN),
            )
            text = json.dumps(summary)
            self.assertIn("fix_rate", text)


if __name__ == "__main__":
    unittest.main()
