from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs" / "navlab"
MISSIONS_DIR = OUTPUTS / "missions"
QUICK_DIR = OUTPUTS / "quick"
RUNS_DIR = OUTPUTS / "runs"
DB_PATH = OUTPUTS / "navlab.db"
PX4_DIR = ROOT / "PX4-Autopilot"
COMFY_INPUT = ROOT / "ComfyUI" / "input"
COMFY_URL = os.environ.get("NAVLAB_COMFY_URL", "http://127.0.0.1:8188")
CAMERA_DIR = OUTPUTS / "camera"
CAMERA_TOPIC = os.environ.get("NAVLAB_CAMERA_TOPIC", "auto")
CAMERA_MODEL = os.environ.get("NAVLAB_CAMERA_MODEL", "x500_mono_cam_down_0")
TRUTH_DIR = OUTPUTS / "truth"
TRUTH_FILE = TRUTH_DIR / "truth.json"
MAPS_DIR = OUTPUTS / "maps"
DEFAULT_BACKEND = os.environ.get("NAVLAB_BACKEND", "dummy")
BFL_API_KEY = os.environ.get("BFL_API_KEY", "")

for path in (OUTPUTS, MISSIONS_DIR, QUICK_DIR, RUNS_DIR, CAMERA_DIR, TRUTH_DIR, MAPS_DIR):
    path.mkdir(parents=True, exist_ok=True)
