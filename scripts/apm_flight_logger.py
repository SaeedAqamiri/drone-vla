"""Fly an ArduPilot SITL quad in GUIDED mode with normalized velocity commands.

Same action contract as ``gz_flight_logger.py`` ([fwd, lat, up, yaw_rate] in
[-1, 1] at 15 Hz, logged to the same CSV schema), but the transport is raw
MAVLink instead of MAVSDK, because ArduPilot has no ``offboard`` mode:

    mode GUIDED  +  SET_POSITION_TARGET_LOCAL_NED (velocity + yaw)

ArduPilot accepts that message only in GUIDED, so unlike PX4 there is no
``offboard.start()`` step -- the setpoint stream itself engages control.

Usage (SITL already running and forwarding to udp 14600):
    python3 apm_flight_logger.py --steps 300 --out actions.csv
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
from pymavlink import mavutil

SCALE = (1.5, 1.5, 1.0, 30.0)  # fwd m/s, lat m/s, up m/s, yaw deg/s
# Take off well above the mission altitude: the vertical velocity loop is lazy
# for the first couple of seconds after the take-off state machine ends and the
# vehicle settles ~1.5-3 m lower than the point where it stopped climbing.
HOVER_ALT = 8.0
RATE_HZ = 15.0

# Thrust per rotor is motorConstant * omega^2 with motorConstant = 8.54858e-06
# and maxRotVelocity = 1000 rad/s (PX4 x500_base/model.sdf). The motor model
# *clamps* at 1000 rad/s, so commanding 1500 only wasted the top third of the
# throttle range: every command above PWM ~1667 produced identical thrust, the
# controller lost authority exactly when it needed it and the altitude loop
# wound up. Map PWM 1000..2000 onto 0..1000 rad/s instead.
MAX_MOTOR_RAD_S = 1000.0
PWM_MIN, PWM_MAX = 1000, 2000


class MotorPublisher:
    """Publish ArduPilot servo outputs as Gazebo motor speeds.

    The x500 airframe listens on ``/<model>/command/motor_speed`` (gz.msgs.Actuators)
    and converts each value to thrust itself; this replaces the broken
    blade-element thrust model of the stock ArduPilot ``iris`` airframe.
    """

    def __init__(self, model: str):
        import gz.transport13 as gz_transport
        from gz.msgs10 import actuators_pb2

        self._pb = actuators_pb2
        self._node = gz_transport.Node()
        topic = f"/{model}/command/motor_speed"
        self._pub = self._node.advertise(topic, actuators_pb2.Actuators)
        if not self._pub.valid():
            raise RuntimeError(f"cannot advertise {topic}")
        self.speeds = [0.0, 0.0, 0.0, 0.0]
        print(f"motor bridge -> {topic}", flush=True)

    def update(self, msg) -> None:
        pwms = (msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw)
        self.speeds = [
            0.0 if pwm <= 0 else min(1.0, max(0.0, (pwm - PWM_MIN) / (PWM_MAX - PWM_MIN)))
            * MAX_MOTOR_RAD_S
            for pwm in pwms
        ]

    def publish(self) -> None:
        out = self._pb.Actuators()
        out.velocity.extend(self.speeds)
        self._pub.publish(out)

# Ignore position + accelerations; use velocity, hold yaw (heading is commanded
# explicitly, like the PX4 script). YAW_RATE is ignored because we send yaw.
TYPE_MASK_VEL_YAWD = 0b0000100111000111  # 2503

# Hold position + yaw, ignore velocity/acceleration/yaw-rate.
# bits: vx 8, vy 16, vz 32, ax 64, ay 128, az 256, yaw_rate 2048
TYPE_MASK_POS_YAWD = 0b100111111000  # 2552


def _ramp(x: float) -> float:
    """Smoothstep: 0 -> 1 with zero slope at both ends."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def pilot(t, T, rng):
    """Smooth demo mission: climb, cruise with a slow S-turn, then ease off.

    The previous version drew fresh random signs on *every* 15 Hz tick, so the
    velocity setpoint jumped between +0.75 and -0.75 m/s 15 times a second. The
    controller chased that noise, banking the airframe 45-60 degrees and drifting
    tens of metres, which is useless for a camera dataset. Everything here is a
    continuous function of mission progress instead, so the horizon stays level
    and the frames look like a real survey flight.
    """
    p = t / max(1, T - 1)  # mission progress in [0, 1]

    up = 0.0
    if p < 0.12:                      # smooth climb out
        up = 0.35 * _ramp(p / 0.12)
    elif p > 0.92:                    # gentle descent at the end
        up = -0.25 * _ramp((p - 0.92) / 0.08)

    # forward speed ramps in, holds, then eases off
    fwd = 0.9 * _ramp(p / 0.15) * (1.0 - _ramp((p - 0.85) / 0.15))

    # slow sinusoid => S-turn; amplitude follows the speed ramp
    s = math.sin(2.0 * math.pi * 1.5 * p)
    lat = 0.55 * s * _ramp(p / 0.20)
    yaw = 0.25 * math.sin(2.0 * math.pi * 1.5 * p + 0.6) * _ramp(p / 0.20)
    return [fwd, lat, up, yaw]


def current_heading(link, timeout=0.6, motor=None, state=None):
    """Vehicle yaw in degrees, NED convention (0 = north).

    The Gazebo world is ENU, so an x500 at rest reports yaw = 90 deg, not 0.
    Commanding yaw = 0 anyway asks for an immediate 90 deg turn; the yaw
    controller then saturates permanently (the gz motor model has no reaction
    torque) and starves roll/pitch of authority.

    This must stay short and keep the motor bridge fed: ArduPilot enters a LAND
    failsafe if GUIDED receives no setpoint for GUIDED_TIMEOUT seconds (default
    3 s). A blocking read here once made the vehicle land between take-off and
    the mission start.
    """
    deadline = time.time() + timeout
    yaw = None
    while time.time() < deadline:
        msg = link.recv_match(
            type=["ATTITUDE", "SERVO_OUTPUT_RAW", "LOCAL_POSITION_NED"],
            blocking=True,
            timeout=0.2,
        )
        if msg is None:
            continue
        kind = msg.get_type()
        if kind == "ATTITUDE":
            yaw = math.degrees(msg.yaw)
        elif kind == "SERVO_OUTPUT_RAW":
            if motor is not None:
                motor.update(msg)
                motor.publish()
        elif state is not None:
            state["x"], state["y"], state["z"] = msg.x, msg.y, msg.z
    if yaw is None:
        return 0.0
    return (yaw + 180.0) % 360.0 - 180.0


# --- plant parameters (applied over MAVLink before every flight) -------------
# These must match sim/ardupilot/models/x500_apm: thrust per rotor is
# motorConstant * omega^2 with omega = throttle * 1500 rad/s, so the vehicle
# hovers at motor output 0.5 (PWM ~1500).
#
# MOT_THST_HOVER is a *thrust fraction* (0..1 of maximum thrust), not a motor
# output. ArduPilot converts it with
#     actuator = spin_min + (spin_max - spin_min) * sqrt(thrust)   [expo 1.0]
# which exactly matches this plant (thrust ∝ omega², omega = throttle * 1000).
# Maximum thrust is 4 * 8.54858e-06 * 1000² = 34.2 N for the 2 kg airframe, so
# holding 19.6 N in the air needs
#     t = 19.6 / 34.2 = 0.573  ->  actuator 0.756  ->  omega 757 rad/s
# Feeding the actuator number (0.5) instead makes the vehicle sink at 0.25 g
# the moment the take-off state machine ends.
# MOT_HOVER_LEARN would adapt to that sinking and bake the error in, and
# MOT_THST_HOVER persists in SITL's eeprom, hence setting it every run.
PLANT_PARAMS = {
    "MOT_THST_HOVER": 0.573,
    "MOT_THST_EXPO": 1.0,
    "MOT_HOVER_LEARN": 0,
    # ArduPilot LANDs if GUIDED gets no setpoint for this many seconds. The
    # default is 3 s, which is shorter than a slow sensor read or a debugger
    # pause; raise it so only a real loss of control triggers the failsafe.
    "GUIDED_TIMEOUT": 20,
}


def configure_plant(link):
    """Push the plant-matching parameters and confirm they took."""
    for name, value in PLANT_PARAMS.items():
        got = set_param(link, name, value)
        print(f"{name} = {got}" if got is not None else f"{name} (no echo)", flush=True)


def wait_heartbeat(link, timeout=90.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = link.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg is not None and msg.get_srcSystem() == link.target_system:
            return msg
    raise RuntimeError("no heartbeat from SITL")


def wait_mode(link, mode_name, timeout=15.0):
    wanted = link.mode_mapping()[mode_name]
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = link.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg is not None and msg.custom_mode == wanted:
            return True
    return False


def set_mode(link, mode_name):
    link.set_mode_apm(mode_name)
    if not wait_mode(link, mode_name):
        raise RuntimeError(f"could not enter {mode_name}")


def request_message_interval(link, msg_id, hz):
    """Ask ArduPilot to stream one message at ``hz`` (best effort)."""
    link.mav.command_long_send(
        link.target_system,
        link.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        int(1e6 / hz) if hz > 0 else -1,
        0, 0, 0, 0, 0,
    )


def set_param(link, name, value, timeout=8.0):
    """Set a parameter over MAVLink and wait for the PARAM_VALUE echo."""
    link.mav.param_set_send(
        link.target_system,
        link.target_component,
        name.encode("utf-8"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = link.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg is not None and msg.param_id.rstrip("\x00") == name:
            return msg.param_value
    return None


def wait_ready(link, timeout=120.0):
    """Wait for a 3D GPS fix *and* an EKF position estimate.

    A GPS fix alone is not enough: ArduPilot refuses to enter GUIDED (silently)
    until the EKF reports absolute horizontal and vertical position, which is
    what ``EKF_STATUS_REPORT`` flags bit 3/4 mean.
    """
    deadline = time.time() + timeout
    fix = 0
    flags = 0
    need = (1 << 3) | (1 << 4)
    while time.time() < deadline:
        msg = link.recv_match(
            type=["GPS_RAW_INT", "STATUSTEXT", "HEARTBEAT", "EKF_STATUS_REPORT"],
            blocking=True,
            timeout=1,
        )
        if msg is None:
            continue
        kind = msg.get_type()
        if kind == "GPS_RAW_INT":
            fix = msg.fix_type
        elif kind == "EKF_STATUS_REPORT":
            flags = msg.flags
        elif kind == "STATUSTEXT":
            print(f"AP: {msg.text}", flush=True)
        if fix >= 3 and (flags & need) == need:
            print(f"ready: GPS fix={fix} ekf_flags={flags}", flush=True)
            return True
    print(
        f"warning: not fully ready (fix_type={fix}, ekf_flags={flags}); trying anyway",
        flush=True,
    )
    return False


def arm(link, timeout=45.0):
    deadline = time.time() + timeout
    next_send = 0.0
    while time.time() < deadline:
        now = time.time()
        if now >= next_send:
            link.mav.command_long_send(
                link.target_system,
                link.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1, 0, 0, 0, 0, 0, 0,
            )
            next_send = now + 3.0
        msg = link.recv_match(
            type=["HEARTBEAT", "STATUSTEXT", "COMMAND_ACK"], blocking=True, timeout=0.5
        )
        if msg is None:
            continue
        kind = msg.get_type()
        if kind == "HEARTBEAT":
            if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                return True
        elif kind == "STATUSTEXT":
            print(f"AP: {msg.text}", flush=True)
        elif kind == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if msg.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"arm refused (result={msg.result})", flush=True)
    raise RuntimeError("arming timed out")


def service(link, motor, state):
    """Read whatever telemetry is pending and forward servo frames immediately.

    Called both from the command loop and from the wait between command ticks.
    Sleeping straight after a command leaves the ESC setpoint stale for up to
    66 ms (one 15 Hz tick), which is enough for the rate loop to oscillate.
    """
    published = 0
    while True:
        msg = link.recv_match(
            type=["LOCAL_POSITION_NED", "SERVO_OUTPUT_RAW"], blocking=False
        )
        if msg is None:
            return published
        if msg.get_type() == "SERVO_OUTPUT_RAW":
            if state is not None:
                state["pwm"] = (
                    msg.servo1_raw,
                    msg.servo2_raw,
                    msg.servo3_raw,
                    msg.servo4_raw,
                )
            if motor is not None:
                motor.update(msg)
                motor.publish()
                published += 1
        elif state is not None:
            state["x"], state["y"], state["z"] = msg.x, msg.y, msg.z


def send_position(link, north, east, down, yaw_deg):
    link.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        link.target_system,
        link.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAWD,
        north, east, down,       # position m (NED, down positive)
        0, 0, 0,                 # velocity (ignored)
        0, 0, 0,                 # acceleration (ignored)
        yaw_deg, 0,
    )


def stabilize(link, motor, state, seconds=2.5, yaw_deg=0.0):
    """Hold the current position for a moment before the mission starts.

    Handing over to a zero-velocity setpoint while the vehicle is still climbing
    makes the velocity loop brake the climb it did not command and overshoot
    downwards (it fell from 5.6 m to 0.6 m in one episode). Holding position
    first lets the vertical velocity settle to zero, so the mission starts from a
    clean hover.
    """
    deadline = time.time() + seconds
    # let a fresh position estimate arrive before locking the hold target
    warm = time.time() + 0.4
    while time.time() < warm:
        service(link, motor, state)
        time.sleep(0.002)
    target = {"x": state.get("x", 0.0), "y": state.get("y", 0.0), "z": state.get("z", 0.0)}
    last = 0.0
    while time.time() < deadline:
        if time.time() - last > 0.5:
            send_position(link, target["x"], target["y"], target["z"], yaw_deg)
            last = time.time()
        service(link, motor, state)
        time.sleep(0.002)
    print(f"stabilised at alt={-state.get('z', 0.0):.2f} m", flush=True)


def takeoff(link, alt=HOVER_ALT, timeout=40.0, motor=None):
    """Command a GUIDED takeoff and wait until LOCAL_POSITION_NED reaches ``alt``.

    ``GLOBAL_POSITION_INT.relative_alt`` is not always streamed, so altitude is
    read from LOCAL_POSITION_NED (NED: z is negative upwards). When ``motor`` is
    given, servo outputs are forwarded to Gazebo on every iteration so the
    simulated rotors keep spinning during the climb.

    ArduPilot answers ``MAV_RESULT_FAILED`` while the EKF has no origin: the
    GUIDED takeoff path converts the target altitude with
    ``Location::get_alt_m(ABOVE_ORIGIN)``, which fails until the origin is set (a
    second or two after arming). So the command is retried until it is ACCEPTED,
    and never resent afterwards -- a second takeoff is refused because the
    vehicle is no longer "landed".
    """
    reached = 0.0
    accepted = False
    deadline = time.time() + timeout
    next_send = 0.0
    last_report = 0.0
    while time.time() < deadline:
        now = time.time()
        if not accepted and now >= next_send:
            link.mav.command_long_send(
                link.target_system,
                link.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0, 0, 0, 0, 0, 0, alt,
            )
            next_send = now + 1.5

        msg = link.recv_match(
            type=["LOCAL_POSITION_NED", "SERVO_OUTPUT_RAW", "COMMAND_ACK"],
            blocking=True,
            timeout=0.3,
        )
        if msg is None:
            continue
        if msg.get_type() == "COMMAND_ACK":
            if msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                accepted = msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
                print(
                    f"  takeoff ack result={msg.result}"
                    f"{' (accepted)' if accepted else ' (retrying)'}",
                    flush=True,
                )
            continue
        if msg.get_type() == "SERVO_OUTPUT_RAW":
            if motor is not None:
                motor.update(msg)
                motor.publish()
                if time.time() - last_report > 2.0:
                    last_report = time.time()
                    pwm = (msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw)
                    print(
                        f"  takeoff pwm={pwm} speeds="
                        + " ".join(f"{s:.0f}" for s in motor.speeds)
                        + f" alt={reached:.2f}",
                        flush=True,
                    )
            continue
        reached = -msg.z
        if accepted and reached >= alt * 0.7:
            break
    return reached


def send_velocity(link, north, east, down, yaw_deg):
    link.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        link.target_system,
        link.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_YAWD,
        0, 0, 0,                 # position (ignored)
        north, east, down,       # velocity m/s
        0, 0, 0,                 # acceleration (ignored)
        math.radians(yaw_deg),   # yaw
        0,                       # yaw rate (ignored)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--connection", default="tcp:127.0.0.1:5760")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="actions.csv")
    p.add_argument("--trigger", default=None, help="file to create when the flight starts")
    p.add_argument(
        "--motor-topic",
        default=None,
        help="Gazebo model name: forward servo outputs to /<model>/command/motor_speed "
        "(needed for the x500 airframe; omit for built-in SITL physics)",
    )
    args = p.parse_args()

    motor = MotorPublisher(args.motor_topic) if args.motor_topic else None

    print(f"connecting to {args.connection} ...", flush=True)
    link = mavutil.mavlink_connection(args.connection, dialect="ardupilotmega")
    wait_heartbeat(link)
    print(f"connected sys={link.target_system} comp={link.target_component}", flush=True)

    link.mav.request_data_stream_send(
        link.target_system,
        link.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        20,
        1,
    )
    # The motor bridge is only as fast as this stream: ask for the highest rate
    # ArduPilot will give us so the actuator setpoint is never the weak link.
    request_message_interval(link, 36, 100)  # SERVO_OUTPUT_RAW

    # The EKF must have a position estimate before ArduPilot will enter GUIDED,
    # otherwise the mode change is silently refused and every later takeoff
    # returns MAV_RESULT_FAILED.
    wait_ready(link)
    configure_plant(link)

    set_mode(link, "GUIDED")
    print("mode GUIDED", flush=True)

    arm(link)
    print("armed", flush=True)
    alt = takeoff(link, motor=motor)
    print(f"takeoff -> {alt:.2f} m", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []

    # Let the vehicle settle before the mission stream starts.
    state = {"x": 0.0, "y": 0.0, "z": -alt, "pwm": (0, 0, 0, 0)}
    heading = current_heading(link, motor=motor, state=state)
    print(f"initial heading {heading:.1f} deg", flush=True)
    stabilize(link, motor, state, yaw_deg=heading)
    T = args.steps
    t0 = time.time()
    period = 1.0 / RATE_HZ

    for t in range(T):
        if t == 0 and args.trigger:
            tf = Path(args.trigger)
            tf.unlink(missing_ok=True)
            tf.touch()
            print("trigger written", flush=True)

        a = pilot(t, T, rng)
        fwd, lat, up, yaw_rate = (a[i] * SCALE[i] for i in range(4))
        heading = (heading + yaw_rate / RATE_HZ) % 360.0
        rad = math.radians(heading)
        north = fwd * math.cos(rad) - lat * math.sin(rad)
        east = fwd * math.sin(rad) + lat * math.cos(rad)
        send_velocity(link, north, east, -up, heading)

        # Drain telemetry without blocking the 15 Hz command cadence. Every servo
        # frame is forwarded to Gazebo *immediately* (see ``service``).
        service(link, motor, state)

        if motor is not None:
            if t % 15 == 0:
                print(
                    f"t={t:3d} pwm={state['pwm']} speeds="
                    + " ".join(f"{s:.0f}" for s in motor.speeds)
                    + f" alt={-state['z']:.2f}",
                    flush=True,
                )

        rows.append(
            {
                "t": round(time.time() - t0, 3),
                "t_abs": round(time.time(), 4),
                "fwd": a[0],
                "lat": a[1],
                "up": a[2],
                "yaw": a[3],
                "x_m": round(state["x"], 3),
                "y_m": round(state["y"], 3),
                "up_m": round(-state["z"], 3),
            }
        )

        # Keep the motor bridge fed for the whole wait, not just once per tick.
        next_tick = time.time() + period
        while time.time() < next_tick:
            service(link, motor, state)
            time.sleep(0.002)

    print("landing ...", flush=True)
    link.mav.command_long_send(
        link.target_system,
        link.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, 0, 0, 0, 0,
    )
    time.sleep(10)
    link.mav.command_long_send(
        link.target_system,
        link.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0, 0, 0, 0, 0, 0, 0,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "t,t_abs,fwd,lat,up,yaw,x_m,y_m,up_m"
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write(header + "\n")
        for row in rows:
            fh.write(
                f"{row['t']},{row['t_abs']},{row['fwd']},{row['lat']},{row['up']},"
                f"{row['yaw']},{row['x_m']},{row['y_m']},{row['up_m']}\n"
            )
    if args.trigger:
        Path(str(args.trigger) + ".done").touch()
    print(f"saved {len(rows)} actions -> {out}", flush=True)
    print(
        f"final position x={state['x']:.2f} y={state['y']:.2f} up={-state['z']:.2f} m",
        flush=True,
    )


if __name__ == "__main__":
    main()
