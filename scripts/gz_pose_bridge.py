#!/usr/bin/env python3
"""Follow the *simulated vehicle's true pose* with a Gazebo camera.

ArduPilot HITL is archived and Plane-only, so the autopilot here is SITL: real
firmware (EKF3, GUIDED, arming checks, MAVLink), numerically simulated sensors.

Two things are simulated, and this file exists because of the second one:

* the **physics** — SITL's built-in ``--model quad``;
* the **camera** — a static downward camera in Gazebo, teleported to the
  vehicle's pose every tick instead of being rigidly attached. (The ``x500_apm``
  model *can* fly inside Gazebo with a motor bridge — STATUS_2026-11-06 §8.1 —
  but that adds a second physics loop and a motor path to tune; see
  ``sat2aerial/README.md`` §3b for the trade-off.)

Why truth and not the EKF estimate
----------------------------------
The camera has to sit where the vehicle physically is. Driving it from
``LOCAL_POSITION_NED`` instead makes the image a picture of whatever the EKF
already believes, so map matching can only ever confirm that belief: the vision
channel carries no information and the estimate is free to drift. (Measured:
the loop then flies away from its waypoint at 12 m/s while every fix "succeeds".)
SITL's ``SIMSTATE`` message carries the simulated truth as int32 lat/lng, so the
camera gets the real pose and the EKF has something to be corrected *by*.

Frames: ``SIMSTATE`` is (lat, lon) and yaw in radians NED. Gazebo with
``world_frame_orientation=ENU`` is (east, north, up), so ENU offsets are computed
from the map origin and ``sdf_yaw = pi/2 - ned_yaw``. Height comes from
``LOCAL_POSITION_NED`` (baro-accurate, and only affects the render scale).

Runs under the **system python3** (gz bindings + pymavlink)::

    /usr/bin/python3 scripts/gz_pose_bridge.py \\
        --world geo_boulder --model probe_cam --mavlink tcp:127.0.0.1:5760 \\
        --origin-lat 40.0155 --origin-lon -105.269
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from gz.msgs10 import boolean_pb2
from gz.msgs10 import pose_pb2
import gz.transport13 as gztransport
from pymavlink import mavutil

# Node.request(service, request, request_type, response_type, timeout_ms) -> (ok, reply)
REQUEST_TYPE = pose_pb2.Pose
RESPONSE_TYPE = boolean_pb2.Boolean
SERVICE_TIMEOUT_MS = 500

MSG_ATTITUDE = 30
MSG_LOCAL_POSITION_NED = 32
MSG_SIMSTATE = 164


def meters_per_degree(latitude_deg: float) -> tuple[float, float]:
    """WGS84 metres per degree of latitude and longitude at this latitude."""
    phi = math.radians(latitude_deg)
    lat = (
        111132.92
        - 559.82 * math.cos(2 * phi)
        + 1.175 * math.cos(4 * phi)
        - 0.0023 * math.cos(6 * phi)
    )
    lon = (
        111412.84 * math.cos(phi)
        - 93.5 * math.cos(3 * phi)
        + 0.118 * math.cos(5 * phi)
    )
    return lat, lon


def _call_set_pose(
    node: "gztransport.Node", service: str, request: "pose_pb2.Pose"
) -> bool:
    """Call the world's set_pose service.

    ``Node.request`` needs both protobuf types and returns a ``(result, reply)``
    tuple; the reply is ``None`` when the service did not answer. Different gz
    builds are lenient about the tuple, so accept a bare reply as well.
    """
    reply = node.request(service, request, REQUEST_TYPE, RESPONSE_TYPE, SERVICE_TIMEOUT_MS)
    if isinstance(reply, tuple):
        result, reply = reply
        if not result:
            return False
    return bool(getattr(reply, "data", True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True, help="Gazebo world name (for the service path)")
    parser.add_argument("--model", default="probe_cam")
    parser.add_argument("--mavlink", default="tcp:127.0.0.1:5760")
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--timeout", type=float, default=60.0, help="give up if no heartbeat")
    parser.add_argument("--origin-lat", type=float, required=True, help="map origin = world (0, 0)")
    parser.add_argument("--origin-lon", type=float, required=True)
    parser.add_argument(
        "--pose-source",
        choices=["simstate", "ekf"],
        default="simstate",
        help="simstate = the vehicle's true pose (correct); ekf = mirror the "
        "autopilot's own estimate, which makes the vision channel an echo and "
        "leaves the EKF nothing to correct with",
    )
    parser.add_argument("--truth-timeout", type=float, default=10.0, help="fail if no SIMSTATE arrives")
    parser.add_argument(
        "--default-z",
        type=float,
        default=100.0,
        help="camera height until the first LOCAL_POSITION_NED arrives (only affects the render scale)",
    )
    args = parser.parse_args()

    m_lat, m_lon = meters_per_degree(args.origin_lat)

    node = gztransport.Node()
    service = f"/world/{args.world}/set_pose"

    print(f"connecting to {args.mavlink} ...", flush=True)
    master = mavutil.mavlink_connection(args.mavlink)
    master.wait_heartbeat(timeout=args.timeout)
    print(f"heartbeat from system {master.target_system}", flush=True)

    # the default stream rates are ~1 Hz, which would make the camera crawl
    requested = [MSG_ATTITUDE, MSG_LOCAL_POSITION_NED]
    if args.pose_source == "simstate":
        requested.append(MSG_SIMSTATE)
    for message_id in requested:
        master.mav.command_long_send(
            master.target_system or 1,
            master.target_component or 1,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            float(message_id),
            float(int(1e6 / max(args.rate, 1.0))),
            0, 0, 0, 0, 0,
        )
    print(
        f"requested {'+'.join(str(i) for i in requested)} at {args.rate:g} Hz; "
        f"pose source = {args.pose_source}",
        flush=True,
    )

    period = 1.0 / max(args.rate, 1.0)
    state = {
        "north": 0.0,
        "east": 0.0,
        "z_up": None,
        "yaw": None,
        "ekf_north": 0.0,
        "ekf_east": 0.0,
        "seen_truth": False,
        "seen_ekf": False,
    }
    counts = {"sim": 0, "pose": 0, "att": 0, "sent": 0, "fail": 0}
    # recv_match filters by message *name*; passing ids silently matches nothing
    watch = ["ATTITUDE", "LOCAL_POSITION_NED"]
    if args.pose_source == "simstate":
        watch.append("SIMSTATE")

    started = time.time()
    try:
        while True:
            message = master.recv_match(type=watch, blocking=True, timeout=1.0)
            while message is not None:
                kind = message.get_type()
                if kind == "SIMSTATE":
                    # simulator truth: int32 deg*1e7, so no float32 loss
                    lat = message.lat * 1e-7
                    lon = message.lng * 1e-7
                    state["east"] = (lon - args.origin_lon) * m_lon
                    state["north"] = (lat - args.origin_lat) * m_lat
                    state["yaw"] = message.yaw
                    state["seen_truth"] = True
                    counts["sim"] += 1
                elif kind == "LOCAL_POSITION_NED":
                    state["z_up"] = -message.z
                    state["ekf_north"] = message.x
                    state["ekf_east"] = message.y
                    state["seen_ekf"] = True
                    counts["pose"] += 1
                else:  # ATTITUDE
                    if state["yaw"] is None:
                        state["yaw"] = message.yaw
                    counts["att"] += 1
                message = master.recv_match(type=watch, blocking=False)

            elapsed = time.time() - started
            if args.pose_source == "simstate" and not state["seen_truth"]:
                if elapsed > args.truth_timeout:
                    print(
                        f"ERROR: no SIMSTATE after {elapsed:.0f}s — cannot place the camera at the "
                        "vehicle's true pose. Refusing to fall back to the EKF estimate: that "
                        "would make the vision channel an echo of the autopilot's own belief.",
                        flush=True,
                    )
                    return 1
                if int(elapsed) % 5 == 0:
                    print(f"waiting for SIMSTATE ({elapsed:.0f}s) ...", flush=True)
                time.sleep(period)
                continue

            if state["yaw"] is None:
                time.sleep(period)
                continue

            if args.pose_source == "ekf":
                state["east"], state["north"] = state["ekf_east"], state["ekf_north"]
                if not state["seen_ekf"]:
                    time.sleep(period)
                    continue

            request = pose_pb2.Pose()
            request.name = args.model
            request.position.x = state["east"]
            request.position.y = state["north"]
            # only the render scale depends on height; LOCAL_POSITION_NED.z is
            # baro-accurate and 1 m of error at 100 m is 1% of the footprint
            request.position.z = state["z_up"] if state["z_up"] is not None else args.default_z
            sdf_yaw = math.pi / 2.0 - state["yaw"]
            request.orientation.z = math.sin(sdf_yaw / 2.0)
            request.orientation.w = math.cos(sdf_yaw / 2.0)

            ok = _call_set_pose(node, service, request)
            if not ok:
                counts["fail"] += 1
            else:
                counts["sent"] += 1

            if counts["sent"] and counts["sent"] % 100 == 0:
                print(
                    f"t={elapsed:6.1f}s sent={counts['sent']} fail={counts['fail']} sim={counts['sim']} "
                    f"enu=({state['east']:7.1f},{state['north']:7.1f},{request.position.z:6.1f}) "
                    f"yaw={math.degrees(state['yaw']):6.1f}",
                    flush=True,
                )
            if counts["fail"] and counts["fail"] % 100 == 0 and counts["sent"] == 0:
                print(
                    f"WARNING: {counts['fail']} set_pose calls failed — is {service} advertised? "
                    "check `gz service -l`",
                    flush=True,
                )
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        print(
            f"bridge stopped: sent={counts['sent']} fail={counts['fail']} "
            f"simstate={counts['sim']} local_pos={counts['pose']} attitude={counts['att']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
