"""Closed-loop vision-only navigation over a georeferenced map.

This is the piece that turns `orthofly` + `localize` from a benchmark into a
navigation loop: the *only* position source is map matching, and the controller
consumes that estimate. If the estimate drifts or dies, the vehicle drifts.

Two backends share one controller/localizer:

* ``simulate`` (here, fully offline): a point-mass model integrates the
  commanded velocity; the camera image is rendered from the *true* pose with
  sensor effects, and the localizer only ever sees that image.
* live (``scripts/geo_vision_loop.py``): the same objects, but frames come from
  Gazebo and the estimate goes to ArduPilot as ``VISION_POSITION_ESTIMATE``.

Design notes that matter for the real vehicle
---------------------------------------------
* **Search window grows with fix age.** A tight crop matches fast and cannot
  jump far away; when fixes stop, the window widens instead of lying.
* **Dead reckoning between fixes.** The estimate integrates the commanded
  velocity (what an EKF does with a body-velocity input), so a short dropout
  degrades gracefully instead of freezing the estimate in place.
* **The match must agree with the prior.** A fix that lands further than the
  search radius is rejected, because that is the "confident but wrong" case.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .matching import match_arrays
from .orthofly import Camera, Effects, GroundMap, Pose


# --------------------------------------------------------------------------- #
# localizer
# --------------------------------------------------------------------------- #


@dataclass
class Fix:
    x: float
    y: float
    inliers: int
    coverage: float
    ratio: float
    ref_gsd_m: float
    latency_ms: float
    accepted: bool
    reason: str = ""


class MapLocalizer:
    """Match a downward frame against a crop of the map around a prior."""

    def __init__(
        self,
        ground: GroundMap,
        camera: Camera,
        base_search_m: float = 40.0,
        max_search_m: float = 200.0,
        growth_mps: float = 12.0,
        ref_max_px: int = 768,
        ref_min_px: int = 320,
        mode: str = "gray",
        detector: str = "sift",
        max_dim: int = 1024,
    ) -> None:
        self.ground = ground
        self.camera = camera
        self.base_search_m = base_search_m
        self.max_search_m = max_search_m
        self.growth_mps = growth_mps
        self.ref_max_px = ref_max_px
        self.ref_min_px = ref_min_px
        self.mode = mode
        self.detector = detector
        self.max_dim = max_dim

    def search_radius(self, age_s: float) -> float:
        return min(self.max_search_m, self.base_search_m + self.growth_mps * max(0.0, age_s))

    def locate(self, frame: np.ndarray, hint: tuple[float, float], altitude: float, age_s: float = 0.0, seed: int = 0) -> tuple[Fix, np.ndarray | None]:
        radius = self.search_radius(age_s)
        footprint = self.camera.ground_footprint_m(max(altitude, 1.0))
        crop_m = footprint + 2.0 * radius
        frame_gsd = max(footprint / self.camera.width, 1e-6)
        ref_px = int(round(crop_m / (frame_gsd * 1.05)))
        ref_px = max(self.ref_min_px, min(self.ref_max_px, ref_px))
        ref_gsd = crop_m / ref_px

        size_map_px = max(8, int(round(crop_m / self.ground.gsd_m)))
        reference = self.ground.crop(hint[0], hint[1], size_map_px)
        if reference.shape[0] != ref_px:
            reference = cv2.resize(reference, (ref_px, ref_px), interpolation=cv2.INTER_AREA)

        started = time.perf_counter()
        result, homography = match_arrays(
            reference,
            frame,
            mode=self.mode,
            max_dim=self.max_dim,
            detector=self.detector,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        if homography is None or result.inliers < 4:
            return Fix(hint[0], hint[1], result.inliers, result.coverage, result.inlier_ratio, ref_gsd, latency_ms, False, "no_homography"), None

        query_w = frame.shape[1]
        query_h = frame.shape[0]
        scale_query = min(1.0, self.max_dim / float(max(query_w, query_h)))
        scale_ref = min(1.0, self.max_dim / float(max(reference.shape[1], reference.shape[0])))
        centre = np.array([query_w * scale_query / 2.0, query_h * scale_query / 2.0, 1.0])

        try:
            inverse = np.linalg.inv(homography)
            mapped = inverse @ centre
            mapped = mapped[:2] / mapped[2]
        except np.linalg.LinAlgError:
            return Fix(hint[0], hint[1], result.inliers, result.coverage, result.inlier_ratio, ref_gsd, latency_ms, False, "singular"), None

        delta = (mapped - np.array([reference.shape[1] * scale_ref / 2.0, reference.shape[0] * scale_ref / 2.0])) / max(scale_ref, 1e-9)
        east = float(delta[0]) * ref_gsd
        north = float(-delta[1]) * ref_gsd
        x = hint[0] + east
        y = hint[1] + north

        shift = math.hypot(x - hint[0], y - hint[1])
        if shift > radius + footprint * 0.5:
            return Fix(x, y, result.inliers, result.coverage, result.inlier_ratio, ref_gsd, latency_ms, False, "jump_rejected"), homography

        accepted = result.success
        return (
            Fix(x, y, result.inliers, result.coverage, result.inlier_ratio, ref_gsd, latency_ms, accepted, "" if accepted else "gate"),
            homography,
        )


# --------------------------------------------------------------------------- #
# controller
# --------------------------------------------------------------------------- #


@dataclass
class ControllerConfig:
    speed_mps: float = 5.0
    reach_m: float = 4.0
    gain: float = 0.8
    max_accel: float = 4.0


def point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    """Shortest distance from ``point`` to the segment ``start``-``end``.

    A localizer that runs at 1-2 Hz moves the estimate tens of metres per update,
    so a waypoint radius can be stepped clean over. Checking the segment flown
    between two estimates catches that fly-through.
    """
    ax, ay = start
    bx, by = end
    px, py = point
    dx, dy = bx - ax, by - ay
    span = dx * dx + dy * dy
    if span <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class WaypointController:
    def __init__(self, waypoints: list[tuple[float, float]], config: ControllerConfig | None = None) -> None:
        if not waypoints:
            raise ValueError("at least one waypoint is required")
        self.waypoints = list(waypoints)
        self.config = config or ControllerConfig()
        self.index = 0
        self.wps_reached = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def command(self, estimate: tuple[float, float]) -> tuple[float, float]:
        if self.done:
            return 0.0, 0.0
        target = self.waypoints[self.index]
        dx = target[0] - estimate[0]
        dy = target[1] - estimate[1]
        distance = math.hypot(dx, dy)
        if distance <= self.config.reach_m:
            self.wps_reached += 1
            self.index += 1
            return self.command(estimate)
        speed = min(self.config.speed_mps, self.config.gain * distance)
        return dx / distance * speed, dy / distance * speed


# --------------------------------------------------------------------------- #
# MAVLink output
# --------------------------------------------------------------------------- #


class MavlinkSink:
    """Publish the vision fix (and optional setpoints) to a flight controller.

    ``url=None`` keeps everything in memory (an encoder-only ``MAVLink`` writing
    to a ``BytesIO``), which is how the message layout is unit-tested without a
    vehicle. With a URL it connects over MAVLink (``tcp:127.0.0.1:5760`` for
    ArduPilot SITL).

    Frames: the localizer works in ENU metres; MAVLink external navigation wants
    NED metres in the EKF's local frame, so ``x_n = y_enu``, ``y_e = x_enu`` and
    ``z_d = -z_up``. Yaw is clockwise from north in both, so it passes through.
    """

    def __init__(self, url: str | None = None, source_system: int = 255, heartbeat_timeout: float = 30.0) -> None:
        import io

        from pymavlink import mavutil

        self._mavutil = mavutil
        self.master = None
        self.buffer = io.BytesIO()
        if url:
            self.master = mavutil.mavlink_connection(url, source_system=source_system)
            self.mav = self.master.mav
            if heartbeat_timeout:
                # learn target_system/target_component from the autopilot instead
                # of blasting setpoints at sysid 0
                self.master.wait_heartbeat(timeout=heartbeat_timeout)
        else:
            self.mav = mavutil.mavlink.MAVLink(self.buffer)
        self.vision_sent = 0
        self.setpoints_sent = 0
        self.heartbeats_sent = 0
        self._boot_us_at_sync: float | None = None
        self._mono_at_sync: float = 0.0
        self.time_synced = False
        self.last_vision: tuple[float, float, float, float] | None = None

    def send_heartbeat(self) -> None:
        """Advertise a GCS so the autopilot keeps using this link."""
        mavlink = self._mavutil.mavlink
        self.mav.heartbeat_send(mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        self.heartbeats_sent += 1

    # -- clock ------------------------------------------------------------- #

    def request_system_time(self) -> None:
        """Ask the autopilot for SYSTEM_TIME so vision stamps can use its clock.

        Without a GPS ArduPilot does not send SYSTEM_TIME on its own; this is the
        same prompt its own CI uses before flying on external nav.
        """
        import time

        self.mav.system_time_send(int(time.time() * 1e6), 0)

    def observe_time(self, message) -> bool:
        """Anchor our clock to the autopilot's boot time (first SYSTEM_TIME wins)."""
        import time

        if message.get_type() != "SYSTEM_TIME" or not message.time_boot_ms:
            return False
        if self._boot_us_at_sync is not None:
            return True
        self._boot_us_at_sync = float(message.time_boot_ms) * 1000.0
        self._mono_at_sync = time.monotonic()
        self.time_synced = True
        return True

    def timestamp_us(self) -> int:
        """Stamp for VISION_POSITION_ESTIMATE.

        The EKF subtracts this from its own clock to work out how old the fix is.
        A Unix-epoch stamp works only if the lag estimator has converged; a
        boot-time stamp is what the EKF expects and what SITL's own vision feed
        sends. Fall back to epoch when no SYSTEM_TIME arrived (spec-legal, and
        harmless before the first fix).
        """
        import time

        if self._boot_us_at_sync is None:
            return int(time.time() * 1e6)
        return int(self._boot_us_at_sync + (time.monotonic() - self._mono_at_sync) * 1e6)

    # -- encoding ---------------------------------------------------------- #

    def send_vision(self, x_enu: float, y_enu: float, z_up: float, yaw_deg: float, roll_rad: float = 0.0, pitch_rad: float = 0.0, covariance: list[float] | None = None) -> None:
        x_n = float(y_enu)
        y_e = float(x_enu)
        z_d = -float(z_up)
        yaw_rad = math.radians(yaw_deg) % (2.0 * math.pi)
        message = self._mavutil.mavlink.MAVLink_vision_position_estimate_message(
            self.timestamp_us(), x_n, y_e, z_d, roll_rad, pitch_rad, yaw_rad
        )
        if covariance is not None and "covariance" in message.get_fieldnames():
            # MAVLink 2 only: MAVLink 1 has no covariance array, and ArduPilot
            # then falls back to EK3_POSNE_M_NSE / VISO noise parameters.
            message.covariance = list(covariance)
        self.mav.send(message)
        self.last_vision = (x_n, y_e, z_d, yaw_rad)
        self.vision_sent += 1

    def send_position(self, x_enu: float, y_enu: float, z_up: float, yaw_deg: float = 0.0) -> None:
        if self.master is not None:
            target_system = self.master.target_system
            target_component = self.master.target_component
        else:
            target_system = 1
            target_component = 1
        # positions only: ignore velocity/accel/yaw-rate bits
        type_mask = 0b0000111111111000
        self.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            target_system,
            target_component,
            self._mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            float(y_enu),          # x: north
            float(x_enu),          # y: east
            -float(z_up),          # z: down
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            math.radians(yaw_deg) % (2.0 * math.pi),
            0.0,
        )
        self.setpoints_sent += 1


# --------------------------------------------------------------------------- #
# MAVLink control: params, origin, arm, takeoff
# --------------------------------------------------------------------------- #


#: ArduPilot parameters for flying on map matching alone (no GPS).
#:
#: ``VISO_TYPE = 1`` (MAVLink) is the one that actually matters: it defaults to
#: ``0`` (None), and without it ArduPilot parses ``VISION_POSITION_ESTIMATE`` and
#: throws it away — the EKF then has no horizontal position, so there is no home,
#: so arming fails with "AHRS: waiting for home". It is also ``@RebootRequired``,
#: so the SITL defaults file must set it (a live ``PARAM_SET`` is too late).
#:
#: ``EK3_SRC1_POSXY = 6`` is ExternalNav, ``POSZ = 1`` keeps the barometer for
#: height, ``YAW = 1`` keeps the compass, and ``GPS1_TYPE = 0`` stops the EKF
#: waiting for a GPS that will never appear.
#:
#: Names are for ArduPilot 4.7: ``GPS_TYPE`` became ``GPS1_TYPE`` and
#: ``ARMING_CHECK`` became ``ARMING_SKIPCHK`` (a skip mask, ``-1`` = skip all).
VISION_PARAMS: dict[str, float] = {
    "VISO_TYPE": 1.0,
    "AHRS_EKF_TYPE": 3.0,
    "EK3_ENABLE": 1.0,
    "EK3_SRC1_POSXY": 6.0,
    "EK3_SRC1_VELXY": 6.0,
    "EK3_SRC1_POSZ": 1.0,
    "EK3_SRC1_VELZ": 0.0,
    "EK3_SRC1_YAW": 1.0,
    # The pre-arm check inspects *every* source set, including every VELZ, not
    # just the active one. A GPS reference left in SRC2/SRC3 (VELZ defaults to
    # GPS) fails with "EK3 sources require GPS" once GPS1_TYPE=0. Clear them.
    "EK3_SRC2_POSXY": 0.0,
    "EK3_SRC2_VELXY": 0.0,
    "EK3_SRC2_POSZ": 0.0,
    "EK3_SRC2_VELZ": 0.0,
    "EK3_SRC2_YAW": 0.0,
    "EK3_SRC3_POSXY": 0.0,
    "EK3_SRC3_VELXY": 0.0,
    "EK3_SRC3_POSZ": 0.0,
    "EK3_SRC3_VELZ": 0.0,
    "EK3_SRC3_YAW": 0.0,
    "GPS1_TYPE": 0.0,
    "ARMING_SKIPCHK": -1.0,  # bench runs: no GPS, no RC, so the checks would block arming
    "FENCE_ENABLE": 0.0,     # out-of-fence targets are rejected silently
    "FENCE_TYPE": 0.0,
}

GUIDED_MODE = 4  # ArduCopter custom mode


class MavlinkControl:
    """Arm, take off, set parameters and seed the EKF origin.

    Only ever used against a real autopilot (SITL or hardware). Everything is a
    ``COMMAND_LONG`` / ``PARAM_SET`` / ``SET_GPS_GLOBAL_ORIGIN``, and every
    blocking wait can run a ``keepalive`` callback so the vision stream (and the
    GCS heartbeat) keeps flowing while the vehicle climbs — otherwise the EKF
    loses position exactly when it needs it.

    ``on_message`` may be supplied to observe everything received; the live loop
    uses it to track altitude and yaw without opening a second link.
    """

    def __init__(self, sink: "MavlinkSink", on_message=None) -> None:
        if sink.master is None:
            raise RuntimeError("MavlinkControl needs a real connection (not --dry-run)")
        self.sink = sink
        self.master = sink.master
        self.mav = sink.mav
        self.mavutil = sink._mavutil
        self.on_message = on_message

    # -- helpers ----------------------------------------------------------- #

    def _targets(self) -> tuple[int, int]:
        return self.master.target_system or 1, self.master.target_component or 1

    def _drain(self, types: list[str], timeout: float) -> None:
        """Consume pending messages of ``types`` for ``timeout`` seconds."""
        deadline = time.time() + max(0.0, timeout)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            message = self.master.recv_match(type=types, blocking=True, timeout=min(remaining, 0.5))
            if message is not None and self.on_message is not None:
                self.on_message(message)

    def drain(self, limit: int = 200) -> int:
        """Read whatever the autopilot has already sent, without blocking."""
        count = 0
        for _ in range(limit):
            message = self.master.recv_match(blocking=False)
            if message is None:
                break
            count += 1
            if self.on_message is not None:
                self.on_message(message)
        return count

    def wait_for(self, predicate, timeout: float, keepalive=None, label: str = "condition") -> bool:
        """Poll ``predicate()`` while draining telemetry; returns whether it held."""
        deadline = time.time() + timeout
        next_keepalive = 0.0
        while time.time() < deadline:
            if predicate():
                return True
            message = self.master.recv_match(blocking=True, timeout=0.2)
            if message is not None and self.on_message is not None:
                self.on_message(message)
            if keepalive is not None and time.time() >= next_keepalive:
                keepalive()
                next_keepalive = time.time() + 0.1
        if predicate():
            return True
        print(f"warning: timed out waiting for {label}", flush=True)
        return False

    # -- requests ---------------------------------------------------------- #

    def set_param(self, name: str, value: float) -> None:
        target_system, target_component = self._targets()
        self.mav.param_set_send(
            target_system,
            target_component,
            name.encode("ascii"),
            float(value),
            self.mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )

    def request_param(self, name: str) -> None:
        """Ask for a parameter's current value; the reply is a PARAM_VALUE."""
        target_system, target_component = self._targets()
        self.mav.param_request_read_send(
            target_system, target_component, name.encode("ascii"), -1
        )

    def verify_params(self, params: dict[str, float], timeout: float = 5.0) -> dict[str, tuple[float, float | None]]:
        """Read the parameters back and report ``name -> (wanted, actual)`` mismatches.

        PARAM_SET is fire-and-forget, so on a fresh SITL a write that never
        landed looks exactly like an EKF that ignores external nav. Read-back
        removes that ambiguity. ``actual`` is ``None`` when the autopilot never
        answered for that name.
        """
        wanted = dict(params)
        seen: dict[str, float] = {}
        for name in wanted:
            self.request_param(name)

        deadline = time.time() + timeout
        while time.time() < deadline and len(seen) < len(wanted):
            message = self.master.recv_match(type=["PARAM_VALUE"], blocking=True, timeout=0.5)
            if message is None:
                continue
            name = message.param_id
            if isinstance(name, bytes):
                name = name.decode()
            name = name.strip("\x00")
            if name in wanted:
                seen[name] = float(message.param_value)

        mismatches: dict[str, tuple[float, float | None]] = {}
        for name, value in wanted.items():
            if name not in seen:
                mismatches[name] = (value, None)
            elif abs(seen[name] - value) > 1e-3 * max(1.0, abs(value)):
                mismatches[name] = (value, seen[name])
        return mismatches

    def set_message_interval(self, message_id: int, hz: float) -> None:
        """Ask for a telemetry message at a given rate on this link.

        ArduPilot's ``SRx_*`` stream rates are per-MAVLink-channel and default to
        ~1 Hz for position, which is not enough to teleport a camera. This is the
        per-connection override and does not touch stored parameters.
        """
        target_system, target_component = self._targets()
        interval_us = 0 if hz <= 0 else int(round(1e6 / hz))
        self.mav.command_long_send(
            target_system,
            target_component,
            self.mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            float(message_id),
            float(interval_us),
            0, 0, 0, 0,
            0.0,  # param7: 0 = no ack requested
        )

    def set_params(self, params: dict[str, float], settle: float = 0.6) -> None:
        """Send each parameter, spacing them out so none are dropped."""
        for name, value in params.items():
            self.set_param(name, value)
            print(f"  PARAM_SET {name} = {value:g}", flush=True)
            self._drain(["PARAM_VALUE"], settle)

    def send_origin(self, lat: float, lon: float, alt_m: float = 0.0) -> None:
        """Tell the EKF where local (0, 0) is; it only accepts it while unset."""
        target_system, _ = self._targets()
        self.mav.set_gps_global_origin_send(
            target_system,
            int(round(lat * 1e7)),
            int(round(lon * 1e7)),
            int(round(alt_m * 1000)),
        )

    def set_home(self, lat: float, lon: float, alt_m: float = 0.0) -> None:
        """Set HOME explicitly.

        Without a GPS nothing ever populates HOME, and ArduPilot refuses to arm
        with "AHRS: waiting for home" even when every other check is skipped.
        """
        target_system, target_component = self._targets()
        self.mav.command_long_send(
            target_system,
            target_component,
            self.mavutil.mavlink.MAV_CMD_DO_SET_HOME,
            0,
            0.0, 0.0, 0.0, 0.0,  # param1=0: use the coordinates below
            float(lat), float(lon), float(alt_m),
        )

    def set_guided(self) -> None:
        self.set_mode_custom(GUIDED_MODE)

    def set_mode_custom(self, mode: int) -> None:
        target_system, target_component = self._targets()
        self.mav.command_long_send(
            target_system,
            target_component,
            self.mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            self.mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode,
            0, 0, 0, 0, 0,
        )

    def arm(self, timeout: float = 30.0, keepalive=None) -> bool:
        target_system, target_component = self._targets()
        self.mav.command_long_send(
            target_system,
            target_component,
            self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1.0, 0, 0, 0, 0, 0, 0,
        )
        return self.wait_for(lambda: self.is_armed, timeout, keepalive=keepalive, label="arming")

    def takeoff(self, altitude_m: float) -> None:
        target_system, target_component = self._targets()
        self.mav.command_long_send(
            target_system,
            target_component,
            self.mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0,
            float(altitude_m),
        )

    # -- observed state ---------------------------------------------------- #

    @property
    def is_armed(self) -> bool:
        # mavutil reads this from the last HEARTBEAT's base_mode
        return bool(self.master.motors_armed())

    @property
    def mode_name(self) -> str:
        try:
            return self.master.flightmode
        except Exception:  # pragma: no cover - pymavlink quirks
            return "?"


# --------------------------------------------------------------------------- #
# offline closed loop
# --------------------------------------------------------------------------- #


@dataclass
class LoopConfig:
    out: Path
    camera: Camera = field(default_factory=Camera)
    altitude_m: float = 60.0
    start: tuple[float, float] = (0.0, 0.0)
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    control_hz: float = 20.0
    localize_hz: float = 2.0
    duration_s: float = 120.0
    tau_s: float = 0.7
    dropout: float = 0.0
    seed: int = 0
    effects: Effects = field(default_factory=Effects)
    keep_every: int = 0
    yaw_mode: str = "track"  # track the commanded velocity, like a gimbal-stabilised nadir camera


def plot_trajectory(ground: GroundMap, truth: list[tuple[float, float]], estimate: list[tuple[float, float]], waypoints: list[tuple[float, float]], out: Path) -> None:
    """Draw truth (green) and estimate (magenta) over the map, plus the waypoints."""
    xs = [p[0] for p in truth] + [p[0] for p in waypoints]
    ys = [p[1] for p in truth] + [p[1] for p in waypoints]
    margin = 120.0
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    centre = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    span = max(max_x - min_x, max_y - min_y)
    size = 1024
    gsd = span / size
    crop_px = int(round(span / ground.gsd_m))
    image = ground.crop(centre[0], centre[1], crop_px)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    def to_px(point: tuple[float, float]) -> tuple[int, int]:
        col = (point[0] - (centre[0] - span / 2.0)) / gsd
        row = ((centre[1] + span / 2.0) - point[1]) / gsd
        return int(round(col)), int(round(row))

    for index in range(len(truth) - 1):
        cv2.line(image, to_px(truth[index]), to_px(truth[index + 1]), (60, 220, 60), 2)
    for index in range(len(estimate) - 1):
        cv2.line(image, to_px(estimate[index]), to_px(estimate[index + 1]), (255, 80, 255), 1)
    for index, waypoint in enumerate(waypoints):
        cv2.circle(image, to_px(waypoint), 8, (0, 200, 255), 2)
        cv2.putText(image, str(index), (to_px(waypoint)[0] + 10, to_px(waypoint)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.circle(image, to_px(truth[0]), 6, (255, 120, 0), -1)
    cv2.putText(image, "truth", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60), 2)
    cv2.putText(image, "estimate", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image)


def run_simulation(ground: GroundMap, cfg: LoopConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.keep_every:
        (out / "frames").mkdir(exist_ok=True)

    localizer = MapLocalizer(ground, cfg.camera)
    controller = WaypointController(cfg.waypoints)
    rng = np.random.default_rng(cfg.seed)

    truth_x, truth_y = cfg.start
    est_x, est_y = cfg.start
    vx = vy = 0.0
    vcmd_x = vcmd_y = 0.0
    yaw = 0.0
    last_fix_t = 0.0
    next_localize = 0.0
    dt = 1.0 / cfg.control_hz

    rows: list[dict[str, Any]] = []
    truth_path = [(truth_x, truth_y)]
    est_path = [(est_x, est_y)]
    errors: list[float] = []
    fixes = 0
    attempts = 0
    opportunities = 0
    steps = int(cfg.duration_s * cfg.control_hz)
    stopped_at = cfg.duration_s

    for step in range(steps):
        t = step * dt

        # -- perception (only ever sees the rendered image) ------------------ #
        while next_localize <= t + 1e-9:
            next_localize += 1.0 / cfg.localize_hz
            opportunities += 1
            if cfg.dropout > 0 and rng.random() < cfg.dropout:
                continue
            pose = Pose(x=truth_x, y=truth_y, alt=cfg.altitude_m, yaw_deg=yaw)
            frame = ground.render(pose, cfg.camera)
            frame = cfg.effects.apply(frame, rng, math.hypot(vx, vy), cfg.altitude_m, cfg.camera.focal_px)
            age = t - last_fix_t
            fix, _ = localizer.locate(frame, (est_x, est_y), cfg.altitude_m, age_s=age, seed=cfg.seed + attempts)
            attempts += 1
            if fix.accepted:
                est_x, est_y = fix.x, fix.y
                last_fix_t = t
                fixes += 1
                errors.append(math.hypot(est_x - truth_x, est_y - truth_y))
            if cfg.keep_every and attempts % cfg.keep_every == 0:
                cv2.imwrite(str(out / "frames" / f"f_{attempts:05d}.png"), frame)

        # -- control on the estimate ---------------------------------------- #
        vcmd_x, vcmd_y = controller.command((est_x, est_y))
        if cfg.yaw_mode == "track" and math.hypot(vcmd_x, vcmd_y) > 0.3:
            yaw = math.degrees(math.atan2(vcmd_x, vcmd_y)) % 360.0

        # -- truth integrates the command (first-order velocity lag) -------- #
        alpha = min(1.0, dt / max(cfg.tau_s, 1e-3))
        vx += (vcmd_x - vx) * alpha
        vy += (vcmd_y - vy) * alpha
        truth_x += vx * dt
        truth_y += vy * dt

        # -- dead reckoning between fixes (what an EKF does with body velocity)
        est_x += vcmd_x * dt
        est_y += vcmd_y * dt

        truth_path.append((truth_x, truth_y))
        est_path.append((est_x, est_y))
        rows.append(
            {
                "t": round(t, 3),
                "truth_x": round(truth_x, 3),
                "truth_y": round(truth_y, 3),
                "est_x": round(est_x, 3),
                "est_y": round(est_y, 3),
                "error_m": round(math.hypot(est_x - truth_x, est_y - truth_y), 3),
                "yaw_deg": round(yaw, 2),
                "vx": round(vx, 3),
                "vy": round(vy, 3),
                "cmd_x": round(vcmd_x, 3),
                "cmd_y": round(vcmd_y, 3),
                "wp": controller.index,
                "fix_age_s": round(t - last_fix_t, 3),
            }
        )

        if controller.done:
            stopped_at = t
            break

    final = cfg.waypoints[-1]
    final_distance = math.hypot(truth_x - final[0], truth_y - final[1])
    with (out / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_trajectory(ground, truth_path[::2], est_path[::2], cfg.waypoints, out / "trajectory.png")

    summary = {
        "waypoints": cfg.waypoints,
        "reached": controller.wps_reached,
        "total_waypoints": len(cfg.waypoints),
        "completed": controller.done and final_distance <= max(8.0, controller.config.reach_m * 2.0),
        "final_distance_m": round(final_distance, 3),
        "time_s": round(stopped_at, 2),
        # fix_rate is end-to-end availability (dropped frames count against it);
        # match_rate is conditional on an image actually arriving.
        "fix_rate": round(fixes / max(opportunities, 1), 4),
        "match_rate": round(fixes / max(attempts, 1), 4),
        "opportunities": opportunities,
        "attempts": attempts,
        "fixes": fixes,
        "error_m": {
            "median": round(float(np.median(errors)), 3) if errors else None,
            "p90": round(float(np.percentile(errors, 90)), 3) if errors else None,
            "max": round(float(np.max(errors)), 3) if errors else None,
        },
        "config": {
            "altitude_m": cfg.altitude_m,
            "speed_mps": controller.config.speed_mps,
            "localize_hz": cfg.localize_hz,
            "control_hz": cfg.control_hz,
            "dropout": cfg.dropout,
            "camera": asdict(cfg.camera),
            "effects": asdict(cfg.effects),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def add_arguments(parser: Any) -> None:
    parser.add_argument("--ortho", default=None)
    parser.add_argument("--procedural", action="store_true")
    parser.add_argument("--procedural-size", type=int, default=2048)
    parser.add_argument("--procedural-gsd", type=float, default=0.5)
    parser.add_argument("--center-lat", type=float, default=None)
    parser.add_argument("--center-lon", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--waypoints", default="", help="x,y;x,y;... in ENU metres (default: derive from --path)")
    parser.add_argument("--path", default="line", choices=["line", "snake"])
    parser.add_argument("--length", type=float, default=200.0)
    parser.add_argument("--legs", type=int, default=2)
    parser.add_argument("--leg-spacing", type=float, default=60.0)
    parser.add_argument("--heading", type=float, default=90.0)
    parser.add_argument("--alt", type=float, default=60.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--hfov", type=float, default=60.0)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--localize-hz", type=float, default=2.0)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--blur", type=float, default=0.6)
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument("--jpeg", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-every", type=int, default=0)


def _route(cfg: Any) -> list[tuple[float, float]]:
    from .orthofly import line_path, snake_path

    if cfg.path == "snake":
        points = snake_path(cfg.length, cfg.heading, cfg.legs, cfg.leg_spacing, (0.0, 0.0))
    else:
        points = line_path(cfg.length, cfg.heading, (0.0, 0.0))
    return [point for point in points[1:]]


def run_from_args(args: Any) -> int:
    from .orthofly import Camera, Effects, GroundMap, procedural_map

    if args.ortho:
        origin = None
        if args.center_lat is not None and args.center_lon is not None:
            origin = (args.center_lat, args.center_lon)
        ground = GroundMap.from_geotiff(args.ortho, origin=origin)
    elif args.procedural:
        ground = procedural_map(args.procedural_size, args.procedural_gsd, args.seed)
    else:
        raise SystemExit("closure: pass --ortho PATH or --procedural")

    if args.waypoints:
        waypoints = []
        for chunk in args.waypoints.replace(";", "|").split("|"):
            chunk = chunk.strip()
            if not chunk:
                continue
            x, y = chunk.split(",")
            waypoints.append((float(x), float(y)))
    else:
        waypoints = _route(args)

    summary = run_simulation(
        ground,
        LoopConfig(
            out=Path(args.out),
            camera=Camera(width=args.size, height=args.size, hfov_deg=args.hfov),
            altitude_m=args.alt,
            waypoints=waypoints,
            control_hz=args.control_hz,
            localize_hz=args.localize_hz,
            duration_s=args.duration,
            dropout=args.dropout,
            seed=args.seed,
            keep_every=args.keep_every,
            effects=Effects(
                blur_sigma=args.blur,
                noise_sigma=args.noise,
                gamma_jitter=0.08,
                exposure_jitter=0.06,
                jpeg_quality=args.jpeg,
                vignette=0.15,
                exposure_ms=2.0,
            ),
        ),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="sat2aerial.closure", description=__doc__.splitlines()[0])
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
