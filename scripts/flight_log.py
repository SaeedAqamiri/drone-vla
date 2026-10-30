"""Flight telemetry logger for PX4 (SITL or real drone) via MAVSDK.

Connects to a running PX4 instance (SITL udpin://0.0.0.0:14540), optionally
flies a takeoff-hold-land mission, and logs telemetry to CSV at ~20 Hz.

Usage:
    python flight_log.py --fly --altitude 5 --duration 90 --out mission1.csv
    python flight_log.py --duration 60 --out passive.csv    # monitor only
"""

import argparse
import asyncio
import time

import pandas as pd
from mavsdk import System

COLS = [
    "t", "mode", "armed", "landed",
    "roll_deg", "pitch_deg", "yaw_deg",
    "vel_n_mps", "vel_e_mps", "vel_d_mps",
    "lat_deg", "lon_deg", "rel_alt_m", "abs_alt_m",
]


async def run(args):
    drone = System()
    print("connecting to PX4 ...", flush=True)
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async def wait_connected():
        async for st in drone.core.connection_state():
            if st.is_connected:
                return st
        raise RuntimeError("stream ended")

    await asyncio.wait_for(wait_connected(), timeout=30)
    print("connected", flush=True)

    latest = {c: None for c in COLS}
    rows = []
    done = asyncio.Event()

    async def poll(coro, keys, conv):
        async for v in coro:
            for k, val in zip(keys, conv(v)):
                latest[k] = val

    async def poll_mode():
        async for m in drone.telemetry.flight_mode():
            latest["mode"] = str(m).split(".")[-1]

    async def poll_armed():
        async for a in drone.telemetry.armed():
            latest["armed"] = bool(a)

    async def poll_landed():
        async for l in drone.telemetry.landed_state():
            latest["landed"] = str(l).split(".")[-1]

    async def poll_att():
        await poll(
            drone.telemetry.attitude_euler(), ["roll_deg", "pitch_deg", "yaw_deg"],
            lambda a: (round(a.roll_deg, 2), round(a.pitch_deg, 2), round(a.yaw_deg, 2)),
        )

    async def poll_vel():
        await poll(
            drone.telemetry.velocity_ned(), ["vel_n_mps", "vel_e_mps", "vel_d_mps"],
            lambda v: (round(v.north_m_s, 2), round(v.east_m_s, 2), round(v.down_m_s, 2)),
        )

    async def poll_pos():
        await poll(
            drone.telemetry.position(),
            ["lat_deg", "lon_deg", "rel_alt_m", "abs_alt_m"],
            lambda p: (
                round(p.latitude_deg, 7), round(p.longitude_deg, 7),
                round(p.relative_altitude_m, 2), round(p.absolute_altitude_m, 2),
            ),
        )

    async def sampler():
        t0 = time.time()
        while not (await asyncio.sleep(0.05)) and time.time() - t0 < args.duration:
            row = {**latest, "t": round(time.time() - t0, 2)}
            rows.append(row)

    async def watcher():
        await asyncio.sleep(args.duration)
        done.set()

    tasks = [
        asyncio.create_task(poll_mode()), asyncio.create_task(poll_armed()),
        asyncio.create_task(poll_landed()), asyncio.create_task(poll_att()),
        asyncio.create_task(poll_vel()), asyncio.create_task(poll_pos()),
    ]
    await asyncio.sleep(2)

    tasks += [asyncio.create_task(sampler()), asyncio.create_task(watcher())]

    if args.fly:
        await drone.action.set_takeoff_altitude(args.altitude)
        print("arming ...", flush=True)
        await drone.action.arm()
        print(f"takeoff to {args.altitude} m", flush=True)
        await drone.action.takeoff()
        t_to = time.time()
        while time.time() - t_to < 15:
            if latest["rel_alt_m"] and latest["rel_alt_m"] > args.altitude * 0.8:
                print(f"altitude confirmed: {latest['rel_alt_m']} m", flush=True)
                break
            await asyncio.sleep(1)
        else:
            print("WARNING: altitude not confirmed", flush=True)
        await asyncio.sleep(4)
        print("holding ...", flush=True)
        await drone.action.hold()
        await asyncio.sleep(args.hold)
        print("landing ...", flush=True)
        await drone.action.land()
        await asyncio.sleep(12)

    await done.wait()
    for t in tasks:
        t.cancel()

    df = pd.DataFrame([r for r in rows if r["mode"]], columns=COLS)
    df.to_csv(args.out, index=False)
    print(f"saved {len(df)} rows -> {args.out}", flush=True)
    if len(df):
        print("alt range:", df.rel_alt_m.min(), "->", df.rel_alt_m.max())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fly", action="store_true")
    p.add_argument("--altitude", type=float, default=5.0)
    p.add_argument("--hold", type=float, default=8.0)
    p.add_argument("--duration", type=float, default=90.0)
    p.add_argument("--out", type=str, default="telemetry.csv")
    asyncio.run(run(p.parse_args()))
