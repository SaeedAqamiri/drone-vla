from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import CAMERA_MODEL, ROOT, TRUTH_FILE

SCRIPT = ROOT / "scripts" / "gz_truth_bridge.py"


class TruthBridge:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                "/usr/bin/python3",
                str(SCRIPT),
                "--topic",
                "auto",
                "--model",
                CAMERA_MODEL,
                "--out",
                str(TRUTH_FILE),
                "--hz",
                "20",
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

    def read(self) -> dict[str, Any] | None:
        if not TRUTH_FILE.exists():
            return None
        try:
            return json.loads(TRUTH_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None

    def status(self) -> dict[str, Any]:
        data = self.read() or {}
        age = None
        if TRUTH_FILE.exists():
            age = round(time.time() - TRUTH_FILE.stat().st_mtime, 2)
        return {
            "running": bool(self.process and self.process.poll() is None),
            "age_s": age,
            "available": bool(age is not None and age < 1.0),
            "data": data,
        }
