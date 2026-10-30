"""Flight telemetry logger for PX4 (SITL or real drone) via MAVSDK.

Connects to a running PX4 instance (default SITL udp://:14540), optionally
flies a takeoff-hold-land mission, and logs telemetry to CSV.

Usage:
    # passive logging of an already-running SITL
    python mavsdk_flight_log.py --duration 60 --out telemetry.csv

    # fly a mission in SITL and log it
    python mavsdk_flight_log.py --fly --altitude 5 --duration 60 --out mission1.csv
"""

import argparse
import asyncio
import math
import time

import pandas as pd
from mavsdk import System


def euler_from_deg(q_wxyz):
    w, x, y, z = q_wxyz
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


async def run(args):
    drone = System()
    print("connecting to PX4 (udpin://0.0.0.0:14540) ...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async def wait_connected():
        async for st in drone.core.connection_state():
            if st.is_connected:
                return st
        raise RuntimeError("connection_state stream ended")

    state = await asyncio.wait_for(wait_connected(), timeout=30)
    print(f"connected: system {state.uuid}")

    rows = []
    done = asyncio.Event()

    async def collect_position():
        async for p in drone.telemetry.position_velocity_ned():
            rows.append(
                {
                    "t": time.time(),
                    "vel_n_mps": p.velocity_ned.north_m_s,
                    "vel_e_mps": p.velocity_ned.east_m_s,
                    "vel_d_mps": p.velocity_ned.down_m_s,
                    "lat_deg": p.position.latitude_deg,
                    "lon_deg": p.position.longitude_deg,
                    "rel_alt_m": p.position.relative_altitude_m,
                }
            )

    async def collect_attitude():
        async for a in drone.telemetry.attitude_euler():
            rows.append(
                {
                    "t": time.time(),
                    "roll_deg": a.roll_deg,
                    "pitch_deg": a.pitch_deg,
                    "yaw_deg": a.yaw_deg,
                }
            )

    async def mission():
        await asyncio.sleep(2)
        print("arming ...")
        await drone.action.arm()
        await drone.action.set_takeoff_altitude(args.altitude)
        print(f"takeoff to {args.altitude} m")
        await drone.action.takeoff()
        await asyncio.sleep(8)
        print("holding ...")
        await drone.hold()
        await asyncio.sleep(args.hold)
        print("landing ...")
        await drone.action.land()
        await asyncio.sleep(8)
        done.set()

    async def watcher():
        await asyncio.sleep(args.duration)
        done.set()

    tasks = [asyncio.create_task(collect_position()), asyncio.create_task(collect_attitude())]
    if args.fly:
        tasks.append(asyncio.create_task(mission()))
    tasks.append(asyncio.create_task(watcher()))

    await done.wait()
    for t in tasks:
        t.cancel()

    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"saved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fly", action="store_true", help="arm/takeoff/hold/land in SITL")
    p.add_argument("--altitude", type=float, default=5.0)
    p.add_argument("--hold", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--out", type=str, default="telemetry.csv")
    asyncio.run(run(p.parse_args()))
