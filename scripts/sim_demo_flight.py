#!/usr/bin/env python3
"""پرواز نمایشی PX4: arm -> takeoff -> hover (چاپ زنده) -> land.

هم روی SITL و هم روی پهپاد واقعی (Pixhawk) کار می‌کند؛ فقط آدرس فرق دارد.

SITL:   ~/env_flight/bin/python scripts/sim_demo_flight.py
واقعی:  ... sim_demo_flight.py --address serial:///dev/ttyACM0:921600
        ... sim_demo_flight.py --address udpout://192.168.1.50:14540

⚠ روی پهپاد واقعی: پروانه نصب نباشد و پهپاد مهار شده باشد.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from mavsdk import System


def log(msg: str) -> None:
    print(msg, flush=True)


class Telemetry:
    """استریم‌های MAVSDK را یک‌بار باز می‌کند و آخرین مقدار را نگه می‌دارد."""

    def __init__(self, drone: System) -> None:
        self.drone = drone
        self.position = None
        self.health = None
        self.in_air = None
        self.battery = None
        self._tasks: list[asyncio.Task] = []

    async def _watch_position(self) -> None:
        async for p in self.drone.telemetry.position():
            self.position = p

    async def _watch_health(self) -> None:
        async for h in self.drone.telemetry.health():
            self.health = h

    async def _watch_in_air(self) -> None:
        async for v in self.drone.telemetry.in_air():
            self.in_air = v

    async def _watch_battery(self) -> None:
        async for b in self.drone.telemetry.battery():
            self.battery = b

    def start(self) -> None:
        for coro in (
            self._watch_position(),
            self._watch_health(),
            self._watch_in_air(),
            self._watch_battery(),
        ):
            self._tasks.append(asyncio.ensure_future(coro))

    def stop(self) -> None:
        for t in self._tasks:
            t.cancel()

    async def wait_position(self, timeout: float = 30.0) -> bool:
        loop = asyncio.get_event_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            if self.position is not None:
                return True
            await asyncio.sleep(0.2)
        return False

    async def wait_armable(self, timeout: float = 30.0) -> bool:
        loop = asyncio.get_event_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            h = self.health
            if h is not None and getattr(h, "is_armable", False):
                return True
            await asyncio.sleep(0.25)
        return False


async def wait_connection(drone: System, timeout: float) -> None:
    log("در حال اتصال به PX4 ...")

    async def _wait() -> None:
        async for st in drone.core.connection_state():
            if st.is_connected:
                return

    await asyncio.wait_for(_wait(), timeout=timeout)
    log("اتصال برقرار شد ✓")


async def climb_to(tel: Telemetry, target: float, timeout: float = 30.0) -> None:
    loop = asyncio.get_event_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        p = tel.position
        if p is not None:
            log(f"  صعود: alt={p.relative_altitude_m:5.2f} m")
            if p.relative_altitude_m >= target * 0.85:
                return
        await asyncio.sleep(1.0)


async def descend_and_wait(tel: Telemetry, timeout: float = 45.0) -> None:
    loop = asyncio.get_event_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        p = tel.position
        if p is not None:
            log(f"  فرود: alt={p.relative_altitude_m:5.2f} m")
            if p.relative_altitude_m < 0.30:
                return
        await asyncio.sleep(1.5)


async def hover_report(tel: Telemetry, seconds: float) -> None:
    loop = asyncio.get_event_loop()
    end = loop.time() + seconds
    while loop.time() < end:
        p = tel.position
        if p is not None:
            log(
                f"  [hover] alt={p.relative_altitude_m:5.2f} m  "
                f"lat={p.latitude_deg:.6f}  lon={p.longitude_deg:.6f}"
            )
        await asyncio.sleep(1.0)


async def run(args) -> int:
    drone = System()
    await drone.connect(system_address=args.address)
    await wait_connection(drone, args.connect_timeout)

    tel = Telemetry(drone)
    tel.start()

    if not await tel.wait_position(30):
        log("داده‌ی موقعیت نیامد — متوقف می‌شوم")
        tel.stop()
        return 2

    if await tel.wait_armable(args.health_timeout):
        log("EKF آماده است ✓")
    else:
        log("⚠ EKF آماده نشد؛ تلاش برای آرم با احتیاط")

    log(f">>> آرم (ارتفاع هدف {args.altitude} m)")
    armed = False
    for attempt in range(1, 4):
        try:
            await drone.action.arm()
            armed = True
            break
        except Exception as e:  # noqa: BLE001
            log(f"  آرم تلاش {attempt} ناموفق: {e}")
            await asyncio.sleep(2.0)
    if not armed:
        log("آرم نشد — پایان")
        tel.stop()
        return 2
    log("آرم شد ✓")

    await drone.action.set_takeoff_altitude(args.altitude)
    log(">>> takeoff")
    await drone.action.takeoff()
    await climb_to(tel, args.altitude)

    log(">>> hover (بدون ورودی)")
    await hover_report(tel, args.hover)

    log(">>> land")
    await drone.action.land()
    await descend_and_wait(tel)

    log(">>> پایان — فرود کامل ✓")
    tel.stop()
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="udpin://0.0.0.0:14540",
                   help="آدرس MAVSDK (پیش‌فرض: SITL)")
    p.add_argument("--altitude", type=float, default=5.0)
    p.add_argument("--hover", type=float, default=10.0)
    p.add_argument("--connect-timeout", type=float, default=40.0)
    p.add_argument("--health-timeout", type=float, default=30.0)
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
