#!/usr/bin/env python3
"""Run the whole live loop: Gazebo + camera pipe + ArduPilot SITL + pose bridge + vision loop.

Split simulation (see ``scripts/gz_pose_bridge.py``): ArduPilot SITL integrates a
quadcopter with its real EKF, arming checks and GUIDED controller, while Gazebo
renders only the orthophoto ground. A teleport bridge moves a static downward
camera to the vehicle's reported pose every tick, so the vision pipeline sees
real renders of real imagery and the autopilot stays the real one.

    /usr/bin/python3 scripts/run_live.py --run outputs/geo/boulder --alt 100 \
        --waypoints "80,0|80,-80|0,-80"

Processes and ports::

    gz sim -s -r ............ world boulder_probe.sdf, camera topic /probe/image
    gz_frame_pipe.py ........ /probe/image -> /dev/shm/geo_frame.jpg
    arducopter (SITL) ....... SERIAL0  tcp:127.0.0.1:5760   (pose bridge)
                              SERIAL1  udp:127.0.0.1:14650   (vision loop)
    gz_pose_bridge.py ....... MAVLink pose -> /world/geo_boulder/set_pose
    geo_vision_loop.py ...... image -> map match -> VISION_POSITION_ESTIMATE + setpoints

The bridge gets the blocking TCP port because it is the latency-critical path;
the loop gets a UDP port so both can talk to the autopilot at once (SITL's TCP
server accepts a single client).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GZ_PYTHON = "/usr/bin/python3"  # has gz.transport13; the project venv does not


def default_loop_python() -> str:
    """The interpreter with rasterio/pyproj (usually the project venv, never
    the one that owns the gz bindings)."""
    for candidate in (
        os.environ.get("GEO_VISION_PYTHON"),
        "/home/drone/env_flight/bin/python",
        sys.executable,
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return sys.executable


class Proc:
    """A child process whose output goes to ``<logs>/<name>.log``."""

    def __init__(self, name: str, argv: list[str], logs: Path, env: dict | None = None, cwd: Path | None = None) -> None:
        self.name = name
        self.argv = argv
        self.log = logs / f"{name}.log"
        self.handle = self.log.open("w", encoding="utf-8", errors="replace")
        print(f"[{name}] {' '.join(argv)}", flush=True)
        self.process = subprocess.Popen(
            argv,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(cwd) if cwd else None,
            start_new_session=True,
        )

    def alive(self) -> bool:
        return self.process.poll() is None

    def wait_for_log(self, needle: str, timeout: float) -> bool:
        """Block until the child prints ``needle`` (or dies / times out)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive():
                return False
            try:
                if needle in self.log.read_text(encoding="utf-8", errors="replace"):
                    return True
            except FileNotFoundError:
                pass
            time.sleep(0.2)
        return False

    def tail(self, lines: int = 15) -> str:
        try:
            return "\n".join(self.log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except FileNotFoundError:
            return "<no log>"

    def stop(self, timeout: float = 10.0) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self.process.kill()
                self.process.wait(timeout=5)
        self.handle.close()


def wait_tcp(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.3)
    return False


def build_gz_env(run_dir: Path, partition: str) -> dict:
    env = dict(os.environ)
    paths = []
    env_sh = run_dir / "env.sh"
    if env_sh.is_file():
        for line in env_sh.read_text(encoding="utf-8").splitlines():
            if line.startswith("export GZ_SIM_RESOURCE_PATH="):
                paths.append(line.split("=", 1)[1].removesuffix(":${GZ_SIM_RESOURCE_PATH:-}"))
    existing = env.get("GZ_SIM_RESOURCE_PATH")
    if existing:
        paths.append(existing)
    env["GZ_SIM_RESOURCE_PATH"] = ":".join(p for p in paths if p)
    env["GZ_PARTITION"] = partition
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    return env


def vision_params(loop_python: str) -> dict[str, float]:
    """Read the parameter table from the package, without importing it here.

    ``run_live`` itself runs under the interpreter that owns the gz bindings,
    which has no pyproj/rasterio, so the import has to happen in the loop's
    interpreter. One source of truth, two processes.
    """
    code = (
        "import json,sys;"
        f"sys.path.insert(0, {str(REPO)!r});"
        "from sat2aerial.closure import VISION_PARAMS;"
        "print(json.dumps(VISION_PARAMS))"
    )
    result = subprocess.run([loop_python, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("could not read VISION_PARAMS:\n" + result.stderr)
    return {name: float(value) for name, value in json.loads(result.stdout).items()}


def write_defaults(path: Path, params: dict[str, float]) -> None:
    """ArduPilot defaults file: applied before the EKF starts.

    Needed because ``VISO_TYPE`` is ``@RebootRequired``: a live PARAM_SET after
    boot is silently too late, and the EKF then ignores every vision fix.
    """
    lines = ["# GPS-denied map-matching flight (written by scripts/run_live.py)"]
    lines += [f"{name} {value:g}" for name, value in params.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="outputs/geo/boulder", help="output dir of make_geo_world")
    parser.add_argument("--out", default=None, help="where to write logs/report (default <run>/live)")
    parser.add_argument("--alt", type=float, default=100.0, help="flight altitude, metres AGL")
    parser.add_argument("--waypoints", default="80,0|80,-80|0,-80")
    parser.add_argument("--start", default="0,0")
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--reach", type=float, default=6.0)
    parser.add_argument("--rate", type=float, default=2.0, help="localization rate, Hz")
    parser.add_argument("--hold-s", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=300.0, help="wall-clock cap for the flight, seconds")
    parser.add_argument("--progress", type=int, default=10)
    parser.add_argument("--pose-source", choices=["simstate", "ekf"], default="simstate",
                        help="where the Gazebo camera gets its pose; simstate is the vehicle's truth")
    parser.add_argument("--land-at-end", action="store_true")
    parser.add_argument("--bootstrap-s", type=float, default=4.0)
    parser.add_argument("--frame-file", default="/dev/shm/geo_frame.jpg")
    parser.add_argument("--mavlink", default="tcp:127.0.0.1:5760", help="pose bridge link (SERIAL0)")
    parser.add_argument("--telemetry", default="udpin:127.0.0.1:14650", help="vision loop link (SERIAL1)")
    parser.add_argument("--sitl-binary", default=str(Path.home() / "ardupilot/build/sitl/bin/arducopter"))
    parser.add_argument("--loop-python", default=None, help="interpreter for the vision loop (needs rasterio/pyproj)")
    parser.add_argument("--param", action="append", default=[], help="override a SITL default, NAME=VALUE (repeatable)")
    parser.add_argument("--wipe-eeprom", action="store_true", help="start SITL from factory defaults")
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    args = parser.parse_args(argv)

    run_dir = (REPO / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"{meta_path} missing — run scripts/make_geo_world.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    world_sdf = run_dir / "worlds" / f"{meta['name']}_probe.sdf"
    if not world_sdf.is_file():
        raise SystemExit(
            f"{world_sdf} missing — regenerate it with\n"
            f"  python scripts/check_geo_world.py --run {args.run} --alt {args.alt:.0f} --write-only"
        )

    out_dir = Path(args.out) if args.out else run_dir / "live"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    logs = out_dir / "logs"
    logs.mkdir(parents=True)
    sitl_dir = out_dir / "sitl"
    sitl_dir.mkdir()
    if args.wipe_eeprom:
        for stale in ("eeprom.bin", "logs"):
            (sitl_dir / stale).unlink(missing_ok=True)

    spawn_name = meta.get("spawn", {}).get("name", "probe_cam")
    origin = meta["origin"]
    partition = f"geo_live_{os.getpid()}"
    gz_env = build_gz_env(run_dir, partition)
    procs: list[Proc] = []
    vision_loop: subprocess.CompletedProcess | None = None
    report: dict = {"run": str(run_dir), "out": str(out_dir), "altitude_m": args.alt}
    started = time.time()

    def shutdown() -> None:
        for proc in reversed(procs):
            proc.stop()

    loop_python = args.loop_python or default_loop_python()

    try:
        # -- 1. Gazebo: the ground, and one static camera we teleport ----------
        procs.append(
            Proc(
                "gz",
                ["gz", "sim", "-s", "-r", "--headless-rendering", str(world_sdf)],
                logs,
                env=gz_env,
            )
        )

        # -- 2. camera frames -> a file the vision loop can read ---------------
        procs.append(
            Proc(
                "frame_pipe",
                [GZ_PYTHON, str(REPO / "scripts/gz_frame_pipe.py"), "--topic", "/probe/image",
                 "--out", args.frame_file, "--max-hz", "10"],
                logs,
                env=gz_env,
            )
        )
        frame_path = Path(args.frame_file)
        if frame_path.exists():
            frame_path.unlink()
        deadline = time.time() + args.startup_timeout
        while time.time() < deadline and not frame_path.exists():
            if not procs[-1].alive():
                raise SystemExit("frame pipe died:\n" + procs[-1].tail())
            time.sleep(0.3)
        if not frame_path.exists():
            raise SystemExit("no camera frames arrived — is the probe camera in the world?")
        print(f"[frame_pipe] first frame after {time.time() - started:.1f}s", flush=True)

        # -- 3. ArduPilot SITL: physics, EKF, GUIDED ---------------------------
        params = vision_params(loop_python)
        # GUIDED position targets fly at ArduPilot's own WP_SPD (default 10 m/s),
        # so --speed has to reach the autopilot, not just the local controller
        params.setdefault("WP_SPD", args.speed)
        for override in args.param:
            name, _, value = override.partition("=")
            params[name.strip()] = float(value)
        defaults_path = sitl_dir / "geo_live.parm"
        write_defaults(defaults_path, params)
        sitl_cmd = [
            args.sitl_binary,
            "--model", "quad",
            "--speedup", "1",
            "--slave", "0",
            "--sim-address=127.0.0.1",
            "-I0",
            f"--defaults={defaults_path}",
            f"--home={origin['lat']:.7f},{origin['lon']:.7f},0,0",
            f"--serial1=udpclient:{args.telemetry.split(':', 1)[1]}",
        ]
        procs.append(Proc("sitl", sitl_cmd, logs, cwd=sitl_dir))
        if not wait_tcp("127.0.0.1", 5760, args.startup_timeout):
            raise SystemExit("SITL never opened tcp:5760:\n" + procs[-1].tail())
        print("[sitl] listening on tcp:5760", flush=True)

        # -- 4. pose bridge: SITL's true pose -> Gazebo set_pose ---------------
        procs.append(
            Proc(
                "pose_bridge",
                [GZ_PYTHON, str(REPO / "scripts/gz_pose_bridge.py"),
                 "--world", f"geo_{meta['name']}", "--model", spawn_name,
                 "--mavlink", args.mavlink,
                 "--origin-lat", f"{origin['lat']:.10f}",
                 "--origin-lon", f"{origin['lon']:.10f}",
                 "--pose-source", args.pose_source,
                 "--default-z", f"{args.alt:g}"],
                logs,
                env=gz_env,
            )
        )
        if not procs[-1].wait_for_log("heartbeat from system", args.startup_timeout):
            raise SystemExit("pose bridge never got a heartbeat:\n" + procs[-1].tail())

        # -- 5. vision loop ----------------------------------------------------
        loop_cmd = [
            loop_python, str(REPO / "scripts/geo_vision_loop.py"),
            "--map", meta["ortho"],
            "--origin-lat", f"{origin['lat']:.10f}",
            "--origin-lon", f"{origin['lon']:.10f}",
            "--source", f"file:{args.frame_file}",
            "--mavlink", args.telemetry,
            "--width", "512", "--height", "384", "--hfov", "60",
            "--alt", f"{args.alt:g}",
            "--waypoints", args.waypoints,
            "--start", args.start,
            "--speed", f"{args.speed:g}",
            "--reach", f"{args.reach:g}",
            "--rate", f"{args.rate:g}",
            "--bootstrap-s", f"{args.bootstrap_s:g}",
            "--hold-s", f"{args.hold_s:g}",
            "--setup-params", "--send-origin", "--arm",
            "--takeoff-alt", f"{args.alt:g}",
            "--takeoff-timeout", "120",
            "--max-duration", f"{args.max_duration:g}",
            "--progress", str(args.progress),
            "--log", str(out_dir / "fixes.csv"),
            "--plot", str(out_dir / "trajectory.png"),
        ]
        if args.land_at_end:
            loop_cmd.append("--land-at-end")
        print(f"[vision_loop] {' '.join(loop_cmd)}", flush=True)
        loop_log = (logs / "vision_loop.log").open("w", encoding="utf-8")
        chunks: list[str] = []
        loop_proc = subprocess.Popen(
            loop_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert loop_proc.stdout is not None
        for line in loop_proc.stdout:  # stream: a killed run must still leave evidence
            chunks.append(line)
            loop_log.write(line)
            loop_log.flush()
            print(line, end="", flush=True)
        loop_proc.wait(timeout=60)
        loop_log.close()
        text = "".join(chunks)
        report["vision_loop_returncode"] = loop_proc.returncode

        summary = {}
        if text.strip().startswith("{"):
            summary = json.loads(text)
        else:
            start = text.find("{")
            if start >= 0:
                try:
                    summary = json.loads(text[start:])
                except json.JSONDecodeError:
                    summary = {}
        report["summary"] = summary
    finally:
        shutdown()

    report["wall_time_s"] = round(time.time() - started, 1)
    report["passed"] = bool(
        report.get("summary", {}).get("waypoints_reached") == report.get("summary", {}).get("total_waypoints")
        and report.get("summary", {}).get("fixes", 0) > 0
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "wall_time_s": report["wall_time_s"], "out": str(out_dir)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
