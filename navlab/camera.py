from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import CAMERA_DIR, CAMERA_TOPIC, ROOT

SCRIPT = ROOT / "scripts" / "gz_latest_frame.py"
LATEST = CAMERA_DIR / "latest.jpg"
META = CAMERA_DIR / "latest.json"


class CameraStream:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        CAMERA_DIR.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                "/usr/bin/python3",
                str(SCRIPT),
                "--topic",
                CAMERA_TOPIC,
                "--out",
                str(CAMERA_DIR),
                "--fps",
                "5",
            ],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def stop(self) -> None:
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

    def status(self) -> dict[str, Any]:
        running = bool(self.process and self.process.poll() is None)
        meta: dict[str, Any] = {}
        if META.exists():
            try:
                meta = json.loads(META.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        age = None
        if LATEST.exists():
            age = round(time.time() - LATEST.stat().st_mtime, 2)
        return {
            "running": running,
            "available": bool(age is not None and age < 4.0),
            "age_s": age,
            "meta": meta,
            "url": "/api/sim/camera.jpg",
        }
