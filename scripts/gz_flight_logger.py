"""Fly the PX4 x500_mono_cam in Gazebo with normalized velocity commands and log them.

Commands are [fwd, lat, up, yaw_rate] in [-1,1] at 15 Hz — the same action
contract as the rotor-format dataset — streamed as MAVSDK offboard
VelocityNedYaw setpoints (heading integrated from yaw_rate).

Usage (fresh SITL running):
    python gz_flight_logger.py --steps 300 --out run1_actions.csv
"""

import argparse
import asyncio
import math
from pathlib import Path
import time

import pandas as pd
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

SCALE = (1.5, 1.5, 1.0, 30.0)  # fwd m/s, lat m/s, up m/s, yaw deg/s
HOVER_ALT = 3.0


def pilot(t: int, T: int, rng) -> list[float]:
    phase_len = T // 5
    phase = min(t // phase_len, 4)
    sgn = [(rng.random() < 0.5) * -2.0 + 1.0 for _ in range(4)]
    if phase == 0:
        return [0.0, 0.0, 0.8, 0.0]
    if phase == 1:
        return [0.5 * sgn[0], 0.0, 0.0, 0.4 * sgn[3]]
    if phase == 2:
        return [0.0, 0.5 * sgn[1], 0.0, 0.0]
    if phase == 3:
        return [0.35 * sgn[0], 0.35 * sgn[1], 0.1 * sgn[2], 0.5 * sgn[3]]
    return [0.0, 0.0, 0.0, 0.0]


async def run(args):
    drone = System()
    print("connecting ...", flush=True)
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async def wait_connected():
        async for st in drone.core.connection_state():
            if st.is_connected:
                return
        raise RuntimeError("stream ended")

    await asyncio.wait_for(wait_connected(), timeout=30)
    print("connected", flush=True)

    alt = {"v": 0.0}

    async def poll_alt():
        async for p in drone.telemetry.position():
            alt["v"] = p.relative_altitude_m

    alt_task = asyncio.create_task(poll_alt())

    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
    )
    await drone.action.arm()
    print("armed, starting offboard", flush=True)
    await drone.offboard.start()

    import numpy as np

    rng = np.random.default_rng(args.seed)
    rows = []
    heading = 0.0
    T = args.steps
    t0 = time.time()

    for t in range(T):
        if t == 0 and args.trigger:
            tf = Path(args.trigger)
            tf.unlink(missing_ok=True)
            tf.touch()
            print("trigger written", flush=True)
        a = pilot(t, T, rng)
        # phase 0: climb to HOVER_ALT before following the script
        if alt["v"] < HOVER_ALT and t < T // 5:
            a = [0.0, 0.0, 0.8, 0.0]
        fwd, lat, up, yaw_rate = (a[i] * SCALE[i] for i in range(4))
        heading = (heading + yaw_rate / 15.0) % 360.0
        rad = math.radians(heading)
        north = fwd * math.cos(rad) - lat * math.sin(rad)
        east = fwd * math.sin(rad) + lat * math.cos(rad)
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(north, east, -up, heading)
        )
        rows.append(
            {
                "t": round(time.time() - t0, 3),
                "t_abs": round(time.time(), 4),
                "fwd": a[0],
                "lat": a[1],
                "up": a[2],
                "yaw": a[3],
            }
        )
        await asyncio.sleep(1 / 15)

    print("landing ...", flush=True)
    await drone.action.land()
    await asyncio.sleep(10)
    try:
        await drone.action.disarm()
    except Exception:
        pass
    alt_task.cancel()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    if args.trigger:
        Path(str(args.trigger) + ".done").touch()
    print(f"saved {len(df)} actions -> {args.out} | final alt {alt['v']:.2f} m", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="actions.csv")
    p.add_argument("--trigger", type=str, default=None, help="file to create when the flight starts")
    asyncio.run(run(p.parse_args()))
