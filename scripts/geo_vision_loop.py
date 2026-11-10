#!/usr/bin/env python3
"""Live (or replayed) closed loop: map matching -> MAVLink -> waypoints.

The offline twin of this script is `sat2aerial closure` (see `sat2aerial/closure.py`):
same localizer, same controller, same MAVLink encoding. The difference is where
frames come from and where the estimate goes.

Sources
-------
``--source replay:<dir>``
    An ``orthofly`` run (``frames/`` + ``poses.csv``). Frames are fed in order;
    the poses are used **only for the error report**, never by the controller.
    This is how the script itself is verified without a vehicle.

``--source file:<path>``
    A single image file that ``scripts/gz_frame_pipe.py`` keeps overwriting.
    The loop treats a changed mtime as a new frame.

Sink
----
``--mavlink tcp:127.0.0.1:5760`` connects to ArduPilot SITL and sends
``VISION_POSITION_ESTIMATE`` plus position setpoints. ``--dry-run`` encodes the
same messages into memory and prints a summary.

ArduPilot setup for GPS-denied flight (do this once, before takeoff)::

    EK3_SRC1_POSXY = 6      # ExternalNav
    EK3_SRC1_VELXY = 6
    EK3_SRC1_POSZ  = 1      # baro
    EK3_SRC1_YAW   = 1      # compass
    GPS_TYPE       = 0      # disable GPS
    ARMING_CHECK   = 0      # optional, for bench runs

The vision estimate must be in the EKF's local NED frame. This script assumes
the map origin (``--origin-lat/--origin-lon``) *is* the EKF origin, i.e. the
same lat/lon you set on the vehicle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sat2aerial.closure import (  # noqa: E402
    VISION_PARAMS,
    ControllerConfig,
    MapLocalizer,
    MavlinkControl,
    MavlinkSink,
    WaypointController,
    point_segment_distance,
)
from sat2aerial.orthofly import Camera, GroundMap  # noqa: E402

STOP = {"value": False}


def _handle_signal(*_args) -> None:
    STOP["value"] = True


def meters_per_degree(latitude_deg: float) -> tuple[float, float]:
    """WGS84 metres per degree of latitude and longitude at this latitude."""
    phi = math.radians(latitude_deg)
    lat = (
        111132.92
        - 559.82 * math.cos(2 * phi)
        + 1.175 * math.cos(4 * phi)
        - 0.0023 * math.cos(6 * phi)
    )
    lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi) + 0.118 * math.cos(5 * phi)
    return lat, lon


class Telemetry:
    """Track what the autopilot reports, without a second MAVLink link.

    ``LOCAL_POSITION_NED`` gives the local height the camera actually sees
    (the bridge teleports the Gazebo camera to exactly this), so the localizer
    scales its search with the *real* altitude instead of a constant. ``ATTITUDE``
    gives the yaw that the vision fix should carry.

    ``STATUSTEXT``, ``EKF_STATUS_REPORT`` and ``POSITION_TARGET_LOCAL_NED`` are
    kept because they are the only way to see *why* the autopilot is not doing
    what it was told — the answer is usually one status line or one rejected
    target, not a wrong number.

    ``SIMSTATE`` is the simulator's ground truth. It is used **only** to fill the
    ``error_m`` column, never by the controller: the live run is supposed to fly
    on map matching, and this is how that claim gets a number.
    """

    #: EK3_STATUS_REPORT health flags
    EKF_FLAGS = {
        0x01: "attitude",
        0x02: "vel_horiz",
        0x04: "vel_vert",
        0x08: "pos_horiz_rel",
        0x10: "pos_horiz_abs",
        0x20: "pos_vert_abs",
        0x40: "pos_vert_agl",
        0x80: "const_pos_mode",
        0x100: "pred_pos_horiz_rel",
        0x200: "gps_glitch",
        0x400: "accel_error",
    }

    def __init__(self, origin: tuple[float, float] | None = None) -> None:
        self.origin = origin
        self.m_per_deg = meters_per_degree(origin[0]) if origin else (111320.0, 111320.0)
        self.z_up: float | None = None
        self.x_east: float | None = None
        self.y_north: float | None = None
        self.vx_east: float | None = None
        self.vy_north: float | None = None
        self.truth_east: float | None = None
        self.truth_north: float | None = None
        self.truth_messages = 0
        self.yaw_deg: float = 0.0
        self.messages = 0
        self.texts: list[str] = []
        self.ekf_flags = 0
        self.ekf_settled: bool = False
        self.target: tuple[float, float, float] | None = None
        self.target_frame: int | None = None

    def ekf_report(self) -> str:
        names = [name for bit, name in self.EKF_FLAGS.items() if self.ekf_flags & bit]
        return " ".join(names) if names else "none"

    def __call__(self, message) -> None:
        kind = message.get_type()
        if kind == "LOCAL_POSITION_NED":
            self.z_up = -float(message.z)
            # NED (north, east) -> ENU (east, north), the map's frame
            self.x_east = float(message.y)
            self.y_north = float(message.x)
            self.vx_east = float(message.vy)
            self.vy_north = float(message.vx)
            self.messages += 1
        elif kind == "ATTITUDE":
            self.yaw_deg = math.degrees(float(message.yaw)) % 360.0
        elif kind == "EKF_STATUS_REPORT":
            self.ekf_flags = int(message.flags)
            self.ekf_settled = True
        elif kind == "POSITION_TARGET_LOCAL_NED":
            # what ArduPilot thinks it is flying to; if this stays at the current
            # position our setpoints are being rejected (e.g. by the fence)
            self.target = (float(message.x), float(message.y), float(message.z))
            self.target_frame = int(message.coordinate_frame)
        elif kind == "SIMSTATE" and self.origin is not None:
            self.truth_east = (message.lng * 1e-7 - self.origin[1]) * self.m_per_deg[1]
            self.truth_north = (message.lat * 1e-7 - self.origin[0]) * self.m_per_deg[0]
            self.truth_messages += 1
        elif kind == "STATUSTEXT":
            text = message.text.strip()
            self.texts.append(text)
            print(f"[autopilot] {text}", flush=True)


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #


class ReplaySource:
    def __init__(self, directory: Path, camera: Camera) -> None:
        self.directory = directory
        self.frames = sorted((directory / "frames").glob("*.png"))
        if not self.frames:
            raise SystemExit(f"no frames under {directory / 'frames'}")
        self.pose_rows: list[dict] = []
        poses = directory / "poses.csv"
        if poses.is_file():
            with poses.open(encoding="utf-8") as handle:
                self.pose_rows = list(csv.DictReader(handle))
        self.camera = camera

    def __len__(self) -> int:
        return len(self.frames)

    def frame(self, index: int) -> tuple[np.ndarray | None, dict | None]:
        image = cv2.imread(str(self.frames[index]), cv2.IMREAD_COLOR)
        truth = None
        if index < len(self.pose_rows):
            row = self.pose_rows[index]
            truth = {"x": float(row["x_m"]), "y": float(row["y_m"]), "alt": float(row["alt_m"]), "yaw": float(row["yaw_deg"])}
        return image, truth


class FileSource:
    def __init__(self, path: Path, timeout: float = 10.0) -> None:
        self.path = path
        self.timeout = timeout
        self._mtime = 0.0

    def wait_frame(self) -> np.ndarray | None:
        started = time.time()
        while time.time() - started < self.timeout:
            try:
                mtime = self.path.stat().st_mtime
            except FileNotFoundError:
                time.sleep(0.02)
                continue
            if mtime > self._mtime:
                image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
                if image is not None:
                    self._mtime = mtime
                    return image
            time.sleep(0.01)
        return None


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #


def run(args) -> int:
    if args.map:
        origin = None
        if args.origin_lat is not None and args.origin_lon is not None:
            origin = (args.origin_lat, args.origin_lon)
        ground = GroundMap.from_geotiff(args.map, origin=origin)
    elif args.procedural:
        from sat2aerial.orthofly import procedural_map

        ground = procedural_map(args.procedural_size, args.procedural_gsd, args.seed)
    else:
        raise SystemExit("pass --map PATH or --procedural")

    camera = Camera(width=args.width, height=args.height, hfov_deg=args.hfov)
    localizer = MapLocalizer(ground, camera, base_search_m=args.search_m, max_search_m=args.max_search_m)
    controller = WaypointController([tuple(p) for p in args.waypoints], ControllerConfig(speed_mps=args.speed, reach_m=args.reach))
    telemetry = Telemetry(origin=(args.origin_lat, args.origin_lon) if args.origin_lat is not None else None)
    sink = MavlinkSink(
        url=None if args.dry_run else args.mavlink,
        heartbeat_timeout=args.connect_timeout,
    )

    def on_message(message) -> None:
        telemetry(message)
        sink.observe_time(message)

    control = None if sink.master is None else MavlinkControl(sink, on_message=on_message)
    if control is None and (args.arm or args.takeoff_alt > 0 or args.setup_params or args.send_origin):
        raise SystemExit("--arm/--takeoff-alt/--setup-params/--send-origin need a live MAVLink link (no --dry-run)")

    est = (args.start[0], args.start[1])
    previous_est = est
    last_fix = time.time()
    fixes = attempts = 0
    errors: list[float] = []
    rows: list[dict] = []
    log = Path(args.log) if args.log else None
    log_handle = None
    log_writer = None
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        # streamed, not buffered: a live run can be killed at any moment and the
        # per-frame record is the only evidence of what happened
        log_handle = log.open("w", newline="", encoding="utf-8")

    def write_row(row: dict) -> None:
        nonlocal log_writer
        rows.append(row)
        if log_handle is None:
            return
        if log_writer is None:
            log_writer = csv.DictWriter(log_handle, fieldnames=list(row))
            log_writer.writeheader()
        log_writer.writerow(row)
        log_handle.flush()

    def feed_vision(alt: float | None = None, yaw: float | None = None) -> None:
        """Publish the current estimate and a GCS heartbeat.

        Called both inside the loop and while waiting for arm/climb, because the
        EKF needs a continuous external-nav stream to keep a position at all.
        """
        height = alt if alt is not None else (args.alt if telemetry.z_up is None else max(telemetry.z_up, 0.0))
        heading = args.yaw if yaw is None else yaw
        sink.send_vision(est[0], est[1], height, heading)
        sink.send_heartbeat()
        if control is not None:
            # keep reading: STATUSTEXT and EKF_STATUS_REPORT are the only window
            # into why the EKF will not accept the fixes
            control.drain()

    preflight: dict | None = None
    if control is not None:
        preflight = {"params": 0, "origin": False, "armed": False, "takeoff_m": 0.0}
        # without this the EKF has no position to publish, so LOCAL_POSITION_NED
        # never arrives and there is nothing to teleport the camera to
        for message_id, hz in ((2, 1.0), (30, 10.0), (32, 20.0), (33, 2.0), (85, 2.0), (164, 2.0), (193, 1.0)):
            control.set_message_interval(message_id, hz)
        # prompts SYSTEM_TIME so vision stamps can be in ArduPilot's boot clock
        control.sink.request_system_time()
        if control.wait_for(lambda: sink.time_synced, timeout=3.0, label="SYSTEM_TIME"):
            print("clock synced to the autopilot's boot time", flush=True)
        else:
            print(
                "warning: no SYSTEM_TIME; vision stamps fall back to Unix epoch "
                "(the EKF's delay compensation may reject them)",
                flush=True,
            )
        if args.setup_params:
            print("setting GPS-denied parameters ...", flush=True)
            control.set_params(VISION_PARAMS)
            mismatches = control.verify_params(VISION_PARAMS, timeout=6.0)
            preflight["params"] = len(VISION_PARAMS) - len(mismatches)
            for name, (wanted, actual) in mismatches.items():
                shown = "no reply" if actual is None else f"{actual:g}"
                print(f"  WARNING {name}: wanted {wanted:g}, read back {shown}", flush=True)
            if not mismatches:
                print(f"  all {len(VISION_PARAMS)} parameters verified", flush=True)
        if args.send_origin:
            if args.origin_lat is None or args.origin_lon is None:
                raise SystemExit("--send-origin needs --origin-lat/--origin-lon")
            control.send_origin(args.origin_lat, args.origin_lon, 0.0)
            control.set_home(args.origin_lat, args.origin_lon, 0.0)
            preflight["origin"] = True
            print(f"EKF origin + home -> {args.origin_lat:.7f}, {args.origin_lon:.7f}", flush=True)
        # the EKF needs a position before the arming checks will pass, and it can
        # only get one from the vision stream — so feed it until it reports a
        # horizontal position, not for a fixed guess at how long that takes
        print(f"publishing the start fix until the EKF has a position (max {args.ekf_timeout:.0f}s) ...", flush=True)
        started_bootstrap = time.time()
        while time.time() - started_bootstrap < max(args.ekf_timeout, args.bootstrap_s):
            feed_vision(alt=0.0)
            if telemetry.ekf_flags & 0x10:  # pos_horiz_abs
                break
            time.sleep(0.1)
        preflight["bootstrap_s"] = round(time.time() - started_bootstrap, 1)
        preflight["ekf"] = telemetry.ekf_report()
        preflight["armable"] = bool(telemetry.ekf_flags & 0x10 and telemetry.ekf_flags & 0x01)
        print(
            f"EKF health: {telemetry.ekf_report()}"
            f" | local position messages: {telemetry.messages}",
            flush=True,
        )
        control.set_guided()
        if args.arm:
            print("arming ...", flush=True)
            preflight["armed"] = control.arm(
                timeout=args.arm_timeout, keepalive=lambda: feed_vision(alt=0.0)
            )
            print(f"armed={preflight['armed']} mode={control.mode_name}", flush=True)
            if not preflight["armed"]:
                print(f"arm failed; EKF: {telemetry.ekf_report()}", flush=True)
                for text in telemetry.texts[-6:]:
                    print(f"  autopilot said: {text}", flush=True)
                raise SystemExit(
                    "refusing to continue: the vehicle is not armed. "
                    "Common causes: EKF has no external-nav position, or a pre-arm check that "
                    "ARMING_CHECK=0 did not clear."
                )
        if args.takeoff_alt > 0:
            print(f"takeoff -> {args.takeoff_alt:.1f} m ...", flush=True)
            control.takeoff(args.takeoff_alt)

            def climbing() -> bool:
                return telemetry.z_up is None or telemetry.z_up >= args.takeoff_alt - args.takeoff_tolerance

            climbed = control.wait_for(
                climbing,
                timeout=args.takeoff_timeout,
                keepalive=lambda: feed_vision(),
                label="climb",
            )
            preflight["takeoff_m"] = None if telemetry.z_up is None else round(telemetry.z_up, 2)
            print(
                f"climb finished={climbed} altitude={preflight['takeoff_m']} m",
                flush=True,
            )

    started = time.time()
    if args.source.startswith("replay:"):
        source = ReplaySource(Path(args.source.split(":", 1)[1]), camera)
        iterator = range(len(source))
    elif args.source.startswith("file:"):
        source = FileSource(Path(args.source.split(":", 1)[1]))
        iterator = range(args.max_frames)
    else:
        raise SystemExit("--source must be replay:<dir> or file:<path>")

    try:
        for index in iterator:
            if STOP["value"]:
                break
            tick = time.time()
            if isinstance(source, ReplaySource):
                frame, truth = source.frame(index)
                if frame is None:
                    continue
                altitude = truth["alt"] if truth else args.alt
            else:
                frame = source.wait_frame()
                if frame is None:
                    print("no new frame; stopping", flush=True)
                    break
                truth = None
                # the camera height is whatever the vehicle reports; the map
                # match must be scaled with the real altitude, not a constant
                altitude = telemetry.z_up if (args.alt_from_telemetry and telemetry.z_up) else args.alt

            attempts += 1
            age = time.time() - last_fix
            fix, _ = localizer.locate(frame, est, altitude, age_s=age, seed=index)
            if fix.accepted:
                est = (fix.x, fix.y)
                last_fix = time.time()
                fixes += 1
                if truth is not None:
                    errors.append(math.hypot(est[0] - truth["x"], est[1] - truth["y"]))

            # publish the fix, then the next setpoint
            vehicle_yaw = telemetry.yaw_deg if (control is not None and telemetry.messages) else args.yaw
            feed_vision(alt=altitude, yaw=vehicle_yaw)
            if not controller.done:
                # advance while the waypoint is inside the reach circle, or was
                # flown through between two (possibly far apart) fixes
                while not controller.done:
                    target = controller.waypoints[controller.index]
                    inside = math.hypot(target[0] - est[0], target[1] - est[1]) <= args.reach
                    flown_through = point_segment_distance(target, previous_est, est) <= args.reach
                    if not (inside or flown_through):
                        break
                    before = controller.index
                    controller.command(est)
                    if controller.index == before:
                        break
                previous_est = est
                if not controller.done:
                    target = controller.waypoints[controller.index]
                    sink.send_position(target[0], target[1], altitude)

            row = {
                "i": index,
                "t": round(time.time() - started, 3),
                "est_x": round(est[0], 3),
                "est_y": round(est[1], 3),
                "alt": round(altitude, 2),
                "fix": int(fix.accepted),
                "inliers": fix.inliers,
                "coverage": round(fix.coverage, 4),
                "reason": fix.reason,
                "wp": controller.index,
            }
            if telemetry.messages:
                row["tele_alt"] = None if telemetry.z_up is None else round(telemetry.z_up, 2)
                row["tele_yaw"] = round(telemetry.yaw_deg, 1)
                row["tele_x"] = None if telemetry.x_east is None else round(telemetry.x_east, 2)
                row["tele_y"] = None if telemetry.y_north is None else round(telemetry.y_north, 2)
                row["tele_vx"] = None if telemetry.vx_east is None else round(telemetry.vx_east, 2)
                row["tele_vy"] = None if telemetry.vy_north is None else round(telemetry.vy_north, 2)
                row["ekf"] = telemetry.ekf_report()
            if truth is not None:
                row["truth_x"] = truth["x"]
                row["truth_y"] = truth["y"]
                row["error_m"] = round(math.hypot(est[0] - truth["x"], est[1] - truth["y"]), 3)
                errors.append(row["error_m"])
            elif telemetry.truth_messages:
                # live: the simulator's truth exists only to score the fix
                row["truth_x"] = round(telemetry.truth_east, 2)
                row["truth_y"] = round(telemetry.truth_north, 2)
                row["error_m"] = round(
                    math.hypot(est[0] - telemetry.truth_east, est[1] - telemetry.truth_north), 3
                )
                errors.append(row["error_m"])
            write_row(row)

            if args.progress and attempts % args.progress == 0:
                print(
                    f"  t={row['t']:7.1f}s frame={index:5d} est=({est[0]:7.1f},{est[1]:7.1f}) "
                    f"ekf=({telemetry.x_east if telemetry.x_east is None else round(telemetry.x_east, 1)},"
                    f"{telemetry.y_north if telemetry.y_north is None else round(telemetry.y_north, 1)}) "
                    f"v=({telemetry.vx_east if telemetry.vx_east is None else round(telemetry.vx_east, 1)},"
                    f"{telemetry.vy_north if telemetry.vy_north is None else round(telemetry.vy_north, 1)}) "
                    f"alt={altitude:6.1f} fix={int(fix.accepted)} inl={fix.inliers:4d} "
                    f"cov={fix.coverage:.2f} wp={controller.index}"
                    + (
                        ""
                        if telemetry.target is None
                        else f" tgt=({telemetry.target[0]:7.1f},{telemetry.target[1]:7.1f},"
                        f"{telemetry.target[2]:6.1f}) frame={telemetry.target_frame}"
                    ),
                    flush=True,
                )

            if isinstance(source, ReplaySource):
                # replay as fast as the matcher allows; no wall-clock pacing
                pass
            else:
                period = 1.0 / max(args.rate, 0.2)
                sleep = period - (time.time() - tick)
                if sleep > 0:
                    time.sleep(sleep)

            if controller.done:
                break
            if args.max_duration > 0 and time.time() - started > args.max_duration:
                print(
                    f"stopping after {args.max_duration:.0f}s (reached "
                    f"{controller.wps_reached}/{len(controller.waypoints)} waypoints)",
                    flush=True,
                )
                break

        if controller.done and args.hold_s > 0 and control is not None:
            # keep the estimate and the last setpoint flowing so ArduPilot holds
            # position instead of drifting the moment we disconnect
            print(f"holding for {args.hold_s:.1f}s ...", flush=True)
            held_until = time.time() + args.hold_s
            final = controller.waypoints[-1]
            while time.time() < held_until and not STOP["value"]:
                feed_vision(yaw=telemetry.yaw_deg)
                sink.send_position(final[0], final[1], telemetry.z_up or args.alt)
                time.sleep(0.2)
        if control is not None and args.land_at_end:
            print("sending LAND ...", flush=True)
            control.set_mode_custom(9)  # ArduCopter LAND
    finally:
        if log_handle is not None:
            log_handle.close()
        if errors and (args.plot or args.log):
            # a picture of what happened: the same overlay the offline sim writes
            try:
                from sat2aerial.closure import plot_trajectory

                truth = [(r["truth_x"], r["truth_y"]) for r in rows if r.get("truth_x") is not None]
                estimate = [(r["est_x"], r["est_y"]) for r in rows if r.get("truth_x") is not None]
                if len(truth) > 1:
                    out_png = Path(args.plot) if args.plot else log.with_name("trajectory.png")
                    plot_trajectory(ground, truth, estimate, list(controller.waypoints), out_png)
                    print(f"trajectory -> {out_png}", flush=True)
            except Exception as exception:  # plotting must never fail a flight report
                print(f"(could not plot the trajectory: {exception})", flush=True)

    summary = {
        "source": args.source,
        "mavlink": "dry-run" if args.dry_run else args.mavlink,
        "attempts": attempts,
        "fixes": fixes,
        "fix_rate": round(fixes / max(attempts, 1), 4),
        "vision_messages": sink.vision_sent,
        "heartbeats": sink.heartbeats_sent,
        "clock_synced": sink.time_synced,
        "setpoints": sink.setpoints_sent,
        "waypoints_reached": controller.wps_reached,
        "total_waypoints": len(controller.waypoints),
        "final_estimate": [round(est[0], 3), round(est[1], 3)],
        "final_target": list(controller.waypoints[-1]),
        "final_distance_m": round(math.hypot(est[0] - controller.waypoints[-1][0], est[1] - controller.waypoints[-1][1]), 3),
        "vehicle": None if control is None else {
            "preflight": preflight,
            "mode": control.mode_name,
            "armed": control.is_armed,
            "altitude_m": None if telemetry.z_up is None else round(telemetry.z_up, 2),
            "yaw_deg": round(telemetry.yaw_deg, 1),
            "telemetry_messages": telemetry.messages,
        },
        "error_m": {
            "median": round(float(np.median(errors)), 3) if errors else None,
            "p90": round(float(np.percentile(errors, 90)), 3) if errors else None,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if (controller.done or STOP["value"]) else 1


def parse_waypoints(raw: str) -> list[tuple[float, float]]:
    points = []
    for chunk in raw.replace(";", "|").split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        points.append((float(x), float(y)))
    if not points:
        raise SystemExit("--waypoints is empty")
    return points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", default=None, help="georeferenced GeoTIFF used as the reference")
    parser.add_argument("--procedural", action="store_true", help="use the procedural city (replay testing only)")
    parser.add_argument("--procedural-size", type=int, default=2048)
    parser.add_argument("--procedural-gsd", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--origin-lat", type=float, default=None)
    parser.add_argument("--origin-lon", type=float, default=None)
    parser.add_argument("--source", required=True, help="replay:<orthofly run> or file:<image>")
    parser.add_argument("--mavlink", default="tcp:127.0.0.1:5760")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--waypoints", default="100,0")
    parser.add_argument("--start", default="0,0", help="initial estimate x,y metres")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--hfov", type=float, default=60.0)
    parser.add_argument("--alt", type=float, default=60.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--reach", type=float, default=5.0)
    parser.add_argument("--search-m", type=float, default=40.0)
    parser.add_argument("--max-search-m", type=float, default=200.0)
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=100000)
    parser.add_argument("--log", default=None)
    parser.add_argument("--plot", default=None, help="write a truth-vs-estimate trajectory PNG here")
    # -- flight controller (live only) ----------------------------------------
    parser.add_argument("--connect-timeout", type=float, default=30.0, help="seconds to wait for a heartbeat")
    parser.add_argument("--setup-params", action="store_true", help="send the GPS-denied ArduPilot parameter set")
    parser.add_argument("--send-origin", action="store_true", help="send SET_GPS_GLOBAL_ORIGIN for --origin-lat/--origin-lon")
    parser.add_argument("--arm", action="store_true", help="switch to GUIDED and arm before flying")
    parser.add_argument("--arm-timeout", type=float, default=30.0)
    parser.add_argument("--takeoff-alt", type=float, default=0.0, help="0 disables the automatic takeoff")
    parser.add_argument("--takeoff-timeout", type=float, default=90.0)
    parser.add_argument("--takeoff-tolerance", type=float, default=2.0, help="metres below the target that still counts as arrived")
    parser.add_argument("--bootstrap-s", type=float, default=3.0, help="minimum seconds of vision to publish before arming")
    parser.add_argument("--ekf-timeout", type=float, default=20.0, help="max seconds to wait for the EKF to report a position")
    parser.add_argument("--alt-from-telemetry", dest="alt_from_telemetry", action="store_true", default=True,
                        help="scale the map match with the reported altitude (default)")
    parser.add_argument("--no-alt-from-telemetry", dest="alt_from_telemetry", action="store_false")
    parser.add_argument("--hold-s", type=float, default=3.0, help="keep streaming the final setpoint after reaching the last waypoint")
    parser.add_argument("--land-at-end", action="store_true")
    parser.add_argument("--max-duration", type=float, default=600.0, help="wall-clock cap for the flight, seconds (0 disables)")
    parser.add_argument("--progress", type=int, default=10, help="print a status line every N frames (0 disables)")
    args = parser.parse_args(argv)

    start = args.start.split(",")
    args.start = (float(start[0]), float(start[1]))
    args.waypoints = parse_waypoints(args.waypoints)

    signal.signal(signal.SIGINT, _handle_signal)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
