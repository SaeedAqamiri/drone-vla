from __future__ import annotations

import asyncio
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .camera import LATEST as CAMERA_LATEST, CameraStream
from .config import CAMERA_MODEL, PX4_DIR
from .localizer import MapLocalizer
from .truth import TruthBridge
from .geo import GEOD, distance_to_polyline_m, local_ned, mission_route_points


class SimManager:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.telemetry: dict[str, Any] = {}
        self.connected = False
        self.mission_status = "idle"
        self._telemetry_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._log_lines: deque[str] = deque(maxlen=500)
        self._stdin: Any = None
        self._reader_thread: threading.Thread | None = None
        self.camera = CameraStream()
        self.truth = TruthBridge()
        self._camera_retry_at = 0.0
        self._truth_retry_at = 0.0
        self._drone: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ev_task: asyncio.Task | None = None
        self._ev_active = False
        self._ev_z0: float | None = None
        self._ev_last: dict[str, Any] = {}
        self.local_ned: dict[str, float] = {}
        self._ev_error = ""
        self._model_name = CAMERA_MODEL
        self.ev_source = "truth"
        self.use_offboard = False
        self.localizer: MapLocalizer | None = None
        self._map_prior: tuple[float, float] = (0.0, 0.0)
        self._map_last: dict[str, float] | None = None
        self._map_kf = -1
        self.health: dict[str, Any] = {}

    def _log_tail(self, lines: int = 30) -> list[str]:
        return list(self._log_lines)[-lines:]

    def _reader(self, process: subprocess.Popen) -> None:
        if process.stdout is None:
            return
        try:
            for raw in iter(process.stdout.readline, b""):
                text = raw.decode("utf-8", errors="replace")
                for line in text.replace("\r", "\n").splitlines():
                    if line.strip():
                        self._log_lines.append(line)
        except Exception:
            pass

    def start(self, mission: Any, timeout: float = 120.0) -> dict:
        with self._lock:
            if self.process and self.process.poll() is None:
                return {"ok": True, "message": "already running", "pid": self.process.pid}
            self._loop = None
            self._drone = None
            subprocess.run(
                ["pkill", "-f", "mavsdk_server.*udpin://0.0.0.0:14540"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            env = os.environ.copy()
            env["HEADLESS"] = "1"
            env["PX4_GZ_WORLD"] = "baylands"
            env["PX4_HOME_LAT"] = str(mission.start_lat)
            env["PX4_HOME_LON"] = str(mission.start_lon)
            env["PX4_HOME_ALT"] = str(mission.altitude_m)
            self.process = subprocess.Popen(
                ["make", "px4_sitl", "gz_x500_mono_cam_down"],
                cwd=str(PX4_DIR),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._stdin = self.process.stdin
            self._log_lines.clear()
            self._reader_thread = threading.Thread(
                target=self._reader, args=(self.process,), daemon=True, name="navlab-px4-log"
            )
            self._reader_thread.start()
            self.camera.start()
            self._stop.clear()
            self.mission_status = "starting"
            self.connected = False
            self.telemetry = {}
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop, daemon=True, name="navlab-mavsdk"
            )
            self._telemetry_thread.start()

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is None or self.process.poll() is not None:
                return {"ok": False, "message": "PX4 process exited", "log": self._log_tail()}
            if self.connected or any(
                "Startup script returned successfully" in line for line in self._log_tail(400)
            ):
                wait_until = time.time() + 10
                while time.time() < wait_until and (
                    self._loop is None or self._loop.is_closed() or self._drone is None
                ):
                    time.sleep(0.2)
                self.truth.start()
                self.mission_status = "ready"
                return {"ok": True, "message": "SITL ready", "pid": self.process.pid}
            time.sleep(1.0)
        self.mission_status = "timeout"
        return {"ok": False, "message": "SITL startup timeout", "log": self._log_tail()}

    def stop(self) -> dict:
        with self._lock:
            self._stop.set()
            self._ev_active = False
            self.mission_status = "stopping"
            if self._stdin is not None:
                try:
                    self._stdin.close()
                except Exception:
                    pass
                self._stdin = None
            if self.process and self.process.poll() is None:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            self.camera.stop()
            self.truth.stop()
            subprocess.run(
                ["pkill", "-f", "mavsdk_server.*udpin://0.0.0.0:14540"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.process = None
            self.connected = False
            self.telemetry = {}
            self.mission_status = "idle"
            return {"ok": True, "message": "stopped"}

    def status(self) -> dict:
        running = bool(self.process and self.process.poll() is None)
        camera = self.camera.status()
        if running and not camera["running"] and time.time() >= self._camera_retry_at:
            self.camera.start()
            self._camera_retry_at = time.time() + 10.0
            camera = self.camera.status()
        truth = self.truth.status()
        if running and not truth["running"] and time.time() >= self._truth_retry_at:
            self.truth.start()
            self._truth_retry_at = time.time() + 10.0
            truth = self.truth.status()
        return {
            "running": running,
            "pid": self.process.pid if running and self.process else None,
            "connected": self.connected,
            "mission_status": self.mission_status,
            "telemetry": self.telemetry,
            "local_ned": self.local_ned,
            "camera": camera,
            "truth": truth,
            "health": self.health,
            "ev": {
                "active": self._ev_active,
                "last": self._ev_last,
                "error": self._ev_error,
                "source": self.ev_source,
            },
            "log": self._log_tail(40),
        }

    def _telemetry_loop(self) -> None:
        try:
            asyncio.run(self._telemetry())
        except Exception:
            self.connected = False
        finally:
            self._loop = None
            self._drone = None

    async def _telemetry(self) -> None:
        from mavsdk import System

        drone = System()
        while not self._stop.is_set():
            try:
                await drone.connect(system_address="udpin://0.0.0.0:14540")
                break
            except Exception:
                await asyncio.sleep(2.0)
        if self._stop.is_set():
            return

        try:
            await drone.telemetry.set_rate_health(1.0)
            await drone.telemetry.set_rate_position_velocity_ned(10.0)
        except Exception:
            pass

        async def connection() -> None:
            async for state in drone.core.connection_state():
                if self._stop.is_set():
                    return
                self.connected = state.is_connected

        async def position() -> None:
            async for value in drone.telemetry.position():
                if self._stop.is_set():
                    return
                self.telemetry.update(
                    {
                        "lat": value.latitude_deg,
                        "lon": value.longitude_deg,
                        "alt_msl": value.absolute_altitude_m,
                        "alt_rel": value.relative_altitude_m,
                        "updated": time.time(),
                    }
                )

        async def attitude() -> None:
            async for value in drone.telemetry.attitude_euler():
                if self._stop.is_set():
                    return
                self.telemetry.update(
                    {
                        "roll": value.roll_deg,
                        "pitch": value.pitch_deg,
                        "yaw": value.yaw_deg,
                        "updated": time.time(),
                    }
                )

        self._drone = drone
        self._loop = asyncio.get_running_loop()
        await asyncio.gather(connection(), position(), attitude())

    def _run(self, coro):
        if self._loop is None or self._drone is None:
            raise RuntimeError("MAVSDK is not connected")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def enable_gps_denied(
        self,
        mission: Any,
        source: str = "truth",
        map_dir: str | None = None,
        disable_gps: bool = True,
        init_prior: bool = False,
    ) -> dict:
        try:
            if source == "map":
                if not map_dir:
                    return {"ok": False, "message": "map_dir is required"}
                self.localizer = MapLocalizer(map_dir)
                self.ev_source = "map"
                if init_prior and self.local_ned:
                    self._map_prior = (
                        float(self.local_ned.get("east", 0.0)),
                        float(self.local_ned.get("north", 0.0)),
                    )
                elif init_prior and self.telemetry.get("lat") is not None:
                    north, east = local_ned(
                        mission.start_lat,
                        mission.start_lon,
                        float(self.telemetry["lat"]),
                        float(self.telemetry["lon"]),
                    )
                    self._map_prior = (east, north)
                else:
                    self._map_prior = (0.0, 0.0)
                self._map_last = None
                self._map_kf = -1
            else:
                self.ev_source = "truth"
                self.localizer = None
            self._run(self._enable_gps_denied(mission, disable_gps))
            stage = "GPS off" if disable_gps else "GPS on (validation)"
            return {"ok": True, "message": f"EV starting ({source}, {stage})"}
        except Exception as error:
            return {"ok": False, "message": str(error)}

    def restore_gps_async(self) -> bool:
        try:
            self._run(self._restore_gps())
            return True
        except Exception:
            return False

    def disable_gps_denied(self) -> dict:
        self._ev_active = False
        if self._ev_task is not None:
            self._ev_task.cancel()
            self._ev_task = None
        try:
            self._run(self._restore_gps())
            return {"ok": True, "message": "GPS restored"}
        except Exception as error:
            return {"ok": False, "message": str(error)}

    async def _enable_gps_denied(self, mission: Any, disable_gps: bool = True) -> None:
        drone = self._drone
        if drone is None:
            raise RuntimeError("MAVSDK is not connected")
        self._ev_active = True
        self._ev_error = ""
        self._ev_z0 = None
        if self._ev_task is None or self._ev_task.done():
            self._ev_task = asyncio.create_task(self._ev_loop())
        await asyncio.sleep(1.0)
        overrides = (
            {
                "EKF2_EV_CTRL": 11,
                "EKF2_HGT_REF": 3,
                "EKF2_GPS_CTRL": 0,
                "COM_ARM_WO_GPS": 1,
            }
            if disable_gps
            else {
                "EKF2_EV_CTRL": 1,
                "EKF2_HGT_REF": 0,
                "EKF2_GPS_CTRL": 7,
                "COM_ARM_WO_GPS": 0,
            }
        )
        for name, value in [("EKF2_EV_NOISE_MD", 1), *overrides.items()]:
            try:
                await drone.param.set_param_int(name, value)
            except Exception as error:
                self._ev_error = f"{name}: {error}"
        try:
            await drone.param.set_param_float("EKF2_EVP_NOISE", 0.5)
            await drone.param.set_param_float("EKF2_EVV_NOISE", 0.5)
        except Exception:
            pass
        try:
            await drone.action.set_gps_global_origin(
                mission.start_lat,
                mission.start_lon,
                float(mission.altitude_m),
            )
        except Exception as error:
            self._ev_error = f"origin: {error}"

    def set_param(self, name: str, value: str, value_type: str = "int") -> dict:
        try:
            self._run(self._set_param(name, value, value_type))
            return {"ok": True, "message": f"set {name}={value}"}
        except Exception as error:
            return {"ok": False, "message": str(error)}

    async def _set_param(self, name: str, value: str, value_type: str) -> None:
        drone = self._drone
        if drone is None:
            raise RuntimeError("MAVSDK is not connected")
        if value_type == "float":
            await drone.param.set_param_float(name, float(value))
        else:
            await drone.param.set_param_int(name, int(value))

    def disable_gps(self) -> dict:
        try:
            self._run(self._disable_gps())
            return {"ok": True, "message": "GPS disabled"}
        except Exception as error:
            return {"ok": False, "message": str(error)}

    async def _disable_gps(self) -> None:
        drone = self._drone
        if drone is None:
            return
        for name, value in [
            ("EKF2_EV_CTRL", 1),
            ("EKF2_HGT_REF", 0),
            ("EKF2_GPS_CTRL", 0),
            ("COM_ARM_WO_GPS", 1),
        ]:
            try:
                await drone.param.set_param_int(name, value)
            except Exception:
                pass

    async def _restore_gps(self) -> None:
        drone = self._drone
        if drone is None:
            return
        for name, value in [
            ("EKF2_EV_CTRL", 0),
            ("EKF2_HGT_REF", 1),
            ("EKF2_GPS_CTRL", 7),
            ("COM_ARM_WO_GPS", 0),
        ]:
            try:
                await drone.param.set_param_int(name, value)
            except Exception:
                pass

    def _truth_pose(self) -> list[float] | None:
        data = self.truth.read()
        if data:
            return [
                float(data["x"]),
                float(data["y"]),
                float(data["z"]),
                float(data.get("yaw", 0.0)),
            ]
        try:
            out = subprocess.check_output(
                ["gz", "model", "-m", self._model_name, "-p"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            arrays = re.findall(r"\[([^\]]+)\]", out)
            return [float(value) for value in arrays[-2].split()]
        except Exception:
            return None

    async def _ev_loop(self) -> None:
        from mavsdk.mavlink_direct import MavlinkMessage

        while self._ev_active and not self._stop.is_set():
            try:
                meta: dict[str, Any] = {"source": self.ev_source, "updated": time.time()}
                yaw = 0.0
                if self.ev_source == "map" and self.localizer is not None:
                    located = await asyncio.to_thread(
                        self.localizer.locate,
                        str(CAMERA_LATEST),
                        self._map_prior[0],
                        self._map_prior[1],
                        20.0,
                        self._map_kf,
                    )
                    if located is not None:
                        east = located["east"]
                        north = located["north"]
                        z_down = -located["z"]
                        yaw = float(located.get("yaw", 0.0))
                        self._map_prior = (east, north)
                        self._map_kf = int(located["keyframe"])
                        truth = self.truth.read()
                        if truth:
                            meta["truth_error_m"] = round(
                                math.hypot(
                                    east - float(truth["x"]),
                                    north - float(truth["y"]),
                                ),
                                2,
                            )
                        self._map_last = {"east": east, "north": north, "z_down": z_down}
                        meta.update(
                            {
                                "matched": True,
                                "inliers": located["inliers"],
                                "coverage": located.get("coverage"),
                                "keyframe": located["keyframe"],
                                "latency_ms": located["latency_ms"],
                            }
                        )
                    elif self._map_last is not None:
                        east = self._map_last["east"]
                        north = self._map_last["north"]
                        z_down = self._map_last["z_down"]
                        meta.update({"matched": False, "hold": True})
                    else:
                        east, north, z_down = 0.0, 0.0, 0.0
                        meta.update({"matched": False, "hold": True})
                else:
                    xyz = await asyncio.to_thread(self._truth_pose)
                    if xyz is None:
                        await asyncio.sleep(0.2)
                        continue
                    if self._ev_z0 is None:
                        self._ev_z0 = xyz[2]
                    east = xyz[0]
                    north = xyz[1]
                    z_down = -(xyz[2] - self._ev_z0)
                    if len(xyz) > 3:
                        yaw = float(xyz[3])
                    meta["matched"] = True
                message = MavlinkMessage(
                    message_name="VISION_POSITION_ESTIMATE",
                    system_id=1,
                    component_id=197,
                    target_system_id=1,
                    target_component_id=1,
                    fields_json=json.dumps(
                        {
                            "usec": 0,
                            "x": east,
                            "y": north,
                            "z": z_down,
                            "roll": 0.0,
                            "pitch": 0.0,
                            "yaw": yaw,
                            "covariance": [0.5] * 21,
                            "reset_counter": 0,
                        }
                    ),
                )
                if self._drone is not None:
                    await self._drone.mavlink_direct.send_message(message)
                meta.update({"x": east, "y": north, "z": z_down})
                self._ev_last = meta
            except Exception as error:
                self._ev_error = str(error)
            await asyncio.sleep(0.2)

    async def _fly_offboard(
        self,
        drone: Any,
        mission: Any,
        points: list[tuple[float, float]],
        target: float,
        state: dict[str, Any],
        geofence_limit: float,
    ) -> None:
        from mavsdk.offboard import PositionNedYaw

        async def track_ned() -> None:
            async for value in drone.telemetry.position_velocity_ned():
                state["ned"] = value.position

        ned_tracker = asyncio.create_task(track_ned())
        try:
            await drone.offboard.set_position_ned(
                PositionNedYaw(0.0, 0.0, -target, 0.0)
            )
            await asyncio.sleep(0.3)
            await drone.offboard.start()
            for lat, lon in points[1:]:
                north, east = local_ned(
                    mission.start_lat, mission.start_lon, lat, lon
                )
                setpoint = PositionNedYaw(north, east, -target, 0.0)
                leg_deadline = time.time() + 90
                while time.time() < leg_deadline:
                    await drone.offboard.set_position_ned(setpoint)
                    position = state.get("pos")
                    if position is not None:
                        distance = distance_to_polyline_m(
                            points, position.latitude_deg, position.longitude_deg
                        )
                        if distance > geofence_limit:
                            await drone.offboard.stop()
                            await drone.action.land()
                            raise RuntimeError(
                                f"geofence breach: {distance:.0f} m from route"
                            )
                    ned = state.get("ned")
                    if ned is not None:
                        remaining = (
                            (ned.north_m - north) ** 2
                            + (ned.east_m - east) ** 2
                        ) ** 0.5
                        if remaining < 8.0:
                            break
                    await asyncio.sleep(0.1)
            await drone.offboard.stop()
            await drone.action.land()
            await asyncio.sleep(8)
        finally:
            ned_tracker.cancel()

    def fly_to_end(self, mission: Any) -> dict:
        if self.process is None or self.process.poll() is not None:
            return {"ok": False, "message": "SITL is not running"}
        if self.mission_status != "ready":
            return {"ok": False, "message": "SITL is not ready"}
        points = mission_route_points(mission)
        if len(points) < 2:
            return {"ok": False, "message": "route needs at least two points"}
        self.mission_status = "flying"
        thread = threading.Thread(
            target=self._flight_thread,
            args=(mission, points),
            daemon=True,
            name="navlab-flight",
        )
        thread.start()
        return {"ok": True, "message": "flight started", "waypoints": len(points)}

    def _flight_thread(self, mission: Any, points: list[tuple[float, float]]) -> None:
        try:
            asyncio.run(self._flight(mission, points))
            self.mission_status = "landed"
        except Exception as error:
            self.mission_status = f"flight error: {error}"

    async def _flight(self, mission: Any, points: list[tuple[float, float]]) -> None:
        from mavsdk import System

        drone = System()
        await drone.connect(system_address="udpin://0.0.0.0:14540")
        async for state in drone.core.connection_state():
            if state.is_connected:
                break

        async def wait_armable() -> None:
            async for health in drone.telemetry.health():
                if health.is_armable:
                    return

        try:
            await asyncio.wait_for(wait_armable(), timeout=40)
        except Exception:
            pass

        target = min(max(float(mission.altitude_m), 10.0), 50.0)
        try:
            await drone.action.set_takeoff_altitude(target)
        except Exception:
            pass
        await drone.action.arm()
        await drone.action.takeoff()

        state: dict[str, Any] = {}

        async def track() -> None:
            async for value in drone.telemetry.position():
                state["pos"] = value

        tracker = asyncio.create_task(track())
        try:
            deadline = time.time() + 60
            while time.time() < deadline:
                position = state.get("pos")
                if position and position.relative_altitude_m >= target * 0.8:
                    break
                await asyncio.sleep(1.0)

            position = state.get("pos")
            home_alt = 0.0
            if position is not None:
                home_alt = position.absolute_altitude_m - position.relative_altitude_m

            geofence_limit = max(float(mission.corridor_km) * 500.0 + 150.0, 200.0)
            if self._ev_active and self.use_offboard:
                try:
                    await self._fly_offboard(
                        drone, mission, points, target, state, geofence_limit
                    )
                    return
                except Exception as error:
                    self._ev_error = f"offboard: {error}"
            for lat, lon in points[1:]:
                await drone.action.goto_location(lat, lon, home_alt + target, 0.0)
                leg_deadline = time.time() + 120
                while time.time() < leg_deadline:
                    position = state.get("pos")
                    if position is not None:
                        distance = distance_to_polyline_m(
                            points, position.latitude_deg, position.longitude_deg
                        )
                        if distance > geofence_limit:
                            await drone.action.land()
                            raise RuntimeError(
                                f"geofence breach: {distance:.0f} m from route"
                            )
                        _, _, remaining = GEOD.inv(
                            position.longitude_deg,
                            position.latitude_deg,
                            lon,
                            lat,
                        )
                        if remaining < 20.0:
                            break
                    await asyncio.sleep(1.0)
            await drone.action.land()
            await asyncio.sleep(8)
        finally:
            tracker.cancel()


SIM = SimManager()
