from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import MAPS_DIR, OUTPUTS, ROOT

SCRIPT = ROOT / "scripts" / "record_map.py"


class MapRecorder:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.out_dir: Path | None = None
        self.started_at: float | None = None
        self.log_path = OUTPUTS / "map_record.log"

    def start(
        self, mission_id: int, altitude: float, duration: float = 120.0
    ) -> dict:
        if self.process and self.process.poll() is None:
            return {"ok": False, "message": "recording already running"}
        out_dir = MAPS_DIR / f"mission{mission_id}_alt{int(round(altitude))}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "--out",
                str(out_dir),
                "--duration",
                str(duration),
                "--altitude",
                str(altitude),
            ],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.out_dir = out_dir
        self.started_at = time.time()
        return {"ok": True, "message": "recording started", "out": str(out_dir)}

    def stop(self) -> dict:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.process = None
        return {"ok": True, "message": "recording stopped"}

    def status(self) -> dict[str, Any]:
        running = bool(self.process and self.process.poll() is None)
        out_dir = self.out_dir
        frames = 0
        altitude = None
        if out_dir is not None:
            manifest = out_dir / "manifest.json"
            if manifest.exists():
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    frames = len(data.get("frames", []))
                    altitude = data.get("altitude")
                except Exception:
                    pass
            else:
                frames = len(list((out_dir / "frames").glob("*.jpg")))
        return {
            "running": running,
            "out_dir": str(out_dir) if out_dir else None,
            "frames": frames,
            "altitude": altitude,
            "elapsed_s": round(time.time() - self.started_at, 1)
            if self.started_at and running
            else None,
        }
