#!/usr/bin/env python3
"""Bridge ArduPilot motor outputs to a PX4-style Gazebo multicopter model.

ArduPilot's ``ArduPilotPlugin`` only drives *joints* of the stock ``iris``
airframe, which relies on a blade-element aerodynamic model that yields no lift
in this Gazebo build. The PX4 ``x500`` airframe instead uses
``gz-sim-multicopter-motor-model-system``: it listens on
``/<model>/command/motor_speed`` (``gz.msgs.Actuators``) and turns each rotor
speed into thrust ``T = motorConstant * omega^2``.

This process closes that loop without touching C++:

    ArduPilot SERVO_OUTPUT_RAW (MAVLink)  ->  gz Actuators (motor speeds)

The x500 rotor layout is identical to ArduPilot's QuadX, so servo 1..4 map to
rotor_0..3 with no permutation:

    rotor_0 (+x,-y) front-right  ccw   <- servo 1
    rotor_1 (-x,+y) rear-left    ccw   <- servo 2
    rotor_2 (+x,+y) front-left   cw    <- servo 3
    rotor_3 (-x,-y) rear-right   cw    <- servo 4

Usage (SITL on 5760, bridge on the second MAVLink port):
    python3 apm_motor_bridge.py --connection tcp:127.0.0.1:5762 --model x500
"""

from __future__ import annotations

import argparse
import time

import gz.transport13 as gz_transport
from gz.msgs10 import actuators_pb2
from pymavlink import mavutil

# Thrust per rotor is motorConstant * omega^2 with motorConstant = 8.54858e-06
# and maxRotVelocity = 1000 rad/s (see x500_base/model.sdf). The model clamps
# at 1000, so the whole PWM range must map onto 0..1000 rad/s; otherwise the top
# third of the throttle range is dead and the controller loses authority.
DEFAULT_MAX_RAD_S = 1000.0


def pwm_to_speed(pwm: int, pwm_min: int, pwm_max: int, max_rad_s: float) -> float:
    if pwm <= 0:
        return 0.0
    span = max(1, pwm_max - pwm_min)
    u = (pwm - pwm_min) / span
    u = min(1.0, max(0.0, u))
    return u * max_rad_s


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--connection", default="tcp:127.0.0.1:5762")
    p.add_argument("--model", default="x500", help="Gazebo model name in the world")
    p.add_argument("--rate", type=float, default=50.0, help="publish rate in Hz")
    p.add_argument("--pwm-min", type=int, default=1000)
    p.add_argument("--pwm-max", type=int, default=2000)
    p.add_argument("--max-rad-s", type=float, default=DEFAULT_MAX_RAD_S)
    p.add_argument("--timeout", type=float, default=0.0,
                   help="stop after N seconds (0 = run until killed)")
    args = p.parse_args()

    topic = f"/{args.model}/command/motor_speed"
    node = gz_transport.Node()
    pub = node.advertise(topic, actuators_pb2.Actuators)
    if not pub.valid():
        print(f"ERROR: cannot advertise {topic}", flush=True)
        return 1
    print(f"publishing motor speeds on {topic}", flush=True)

    print(f"connecting MAVLink {args.connection} ...", flush=True)
    link = mavutil.mavlink_connection(args.connection, dialect="ardupilotmega")
    if link.wait_heartbeat(timeout=60) is None:
        print("ERROR: no heartbeat", flush=True)
        return 1
    print("connected", flush=True)

    link.mav.request_data_stream_send(
        link.target_system,
        link.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        50,
        1,
    )

    period = 1.0 / max(1.0, args.rate)
    speeds = [0.0, 0.0, 0.0, 0.0]
    started = time.time()
    last_log = started
    frames = 0

    while True:
        if args.timeout and time.time() - started > args.timeout:
            break

        msg = link.recv_match(type="SERVO_OUTPUT_RAW", blocking=False)
        if msg is not None:
            pwms = [msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw]
            speeds = [
                pwm_to_speed(pwm, args.pwm_min, args.pwm_max, args.max_rad_s)
                for pwm in pwms
            ]

        out = actuators_pb2.Actuators()
        out.velocity.extend(speeds)
        pub.publish(out)
        frames += 1

        now = time.time()
        if now - last_log > 2.0:
            print(
                "speeds(rad/s)=" + " ".join(f"{s:6.1f}" for s in speeds)
                + f"  pub_fps={frames / (now - last_log):5.1f}",
                flush=True,
            )
            frames = 0
            last_log = now

        time.sleep(period)

    print("bridge stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
